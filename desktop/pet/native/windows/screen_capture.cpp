#include "screen_capture.h"
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wincodec.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <memory>

namespace tulpa::windows {
using namespace winrt::Windows::Graphics::Capture;
using namespace winrt::Windows::Graphics::DirectX;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;

json frame_metadata(const Frame& frame) {
  return {{"frameId", frame.id}, {"revision", frame.revision}, {"observedAt", frame.observed_at},
    {"expiresAt", frame.expires_at}, {"displayID", frame.display}, {"pixelWidth", frame.width},
    {"pixelHeight", frame.height}, {"byteLength", frame.jpeg.size()}};
}
bool ScreenCapture::supported() { try { return GraphicsCaptureSession::IsSupported(); } catch (...) { return false; } }

bool ScreenCapture::host_excluded() const {
  if (!exclude_) return true;
  if (!host_) return false;
  struct Check { DWORD host; bool protected_windows = true; } check{host_};
  EnumWindows([](HWND window, LPARAM data) -> BOOL {
    auto& check = *reinterpret_cast<Check*>(data); DWORD pid = 0;
    GetWindowThreadProcessId(window, &pid);
    if (pid == check.host && IsWindowVisible(window)) {
      DWORD affinity = 0;
      if (!GetWindowDisplayAffinity(window, &affinity) || affinity != WDA_EXCLUDEFROMCAPTURE) check.protected_windows = false;
    }
    return TRUE;
  }, reinterpret_cast<LPARAM>(&check));
  return check.protected_windows;
}

static std::vector<std::uint8_t> jpeg_image(const std::vector<std::uint8_t>& pixels, UINT width, UINT height, UINT out_width, UINT out_height, float quality) {
  winrt::com_ptr<IWICImagingFactory> factory;
  winrt::check_hresult(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(factory.put())));
  winrt::com_ptr<IWICBitmap> bitmap;
  winrt::check_hresult(factory->CreateBitmapFromMemory(width, height, GUID_WICPixelFormat32bppBGRA, width * 4,
    static_cast<UINT>(pixels.size()), const_cast<BYTE*>(pixels.data()), bitmap.put()));
  winrt::com_ptr<IWICBitmapScaler> scaler;
  winrt::check_hresult(factory->CreateBitmapScaler(scaler.put()));
  winrt::check_hresult(scaler->Initialize(bitmap.get(), out_width, out_height, WICBitmapInterpolationModeFant));
  winrt::com_ptr<IWICFormatConverter> converter;
  winrt::check_hresult(factory->CreateFormatConverter(converter.put()));
  winrt::check_hresult(converter->Initialize(scaler.get(), GUID_WICPixelFormat24bppBGR, WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom));
  winrt::com_ptr<IStream> stream;
  winrt::check_hresult(CreateStreamOnHGlobal(nullptr, TRUE, stream.put()));
  winrt::com_ptr<IWICBitmapEncoder> encoder;
  winrt::check_hresult(factory->CreateEncoder(GUID_ContainerFormatJpeg, nullptr, encoder.put()));
  winrt::check_hresult(encoder->Initialize(stream.get(), WICBitmapEncoderNoCache));
  winrt::com_ptr<IWICBitmapFrameEncode> frame;
  winrt::com_ptr<IPropertyBag2> properties;
  winrt::check_hresult(encoder->CreateNewFrame(frame.put(), properties.put()));
  PROPBAG2 property{}; property.pstrName = const_cast<wchar_t*>(L"ImageQuality");
  VARIANT value{}; value.vt = VT_R4; value.fltVal = quality;
  winrt::check_hresult(properties->Write(1, &property, &value));
  winrt::check_hresult(frame->Initialize(properties.get()));
  winrt::check_hresult(frame->SetSize(out_width, out_height));
  WICPixelFormatGUID format = GUID_WICPixelFormat24bppBGR;
  winrt::check_hresult(frame->SetPixelFormat(&format));
  winrt::check_hresult(frame->WriteSource(converter.get(), nullptr));
  winrt::check_hresult(frame->Commit()); winrt::check_hresult(encoder->Commit());
  STATSTG info{}; winrt::check_hresult(stream->Stat(&info, STATFLAG_NONAME));
  if (info.cbSize.QuadPart > 4 * 1024 * 1024) throw std::runtime_error("jpeg_too_large");
  std::vector<std::uint8_t> bytes(static_cast<size_t>(info.cbSize.QuadPart));
  LARGE_INTEGER zero{}; winrt::check_hresult(stream->Seek(zero, STREAM_SEEK_SET, nullptr));
  ULONG count = 0; winrt::check_hresult(stream->Read(bytes.data(), static_cast<ULONG>(bytes.size()), &count));
  bytes.resize(count); return bytes;
}
Frame synthetic_protocol_frame() {
  Frame frame; frame.id = unique_id(); frame.width = 2; frame.height = 2; frame.observed_at = unix_time();
  frame.jpeg = jpeg_image(std::vector<std::uint8_t>(16, 255), 2, 2, 2, 2, 0.8f);
  FrameSlot slot; return *slot.store(std::move(frame), unix_time());
}

struct PersistentCapture {
  struct Latest {
    std::mutex mutex;
    std::condition_variable ready;
    winrt::com_ptr<ID3D11Device> device;
    winrt::com_ptr<ID3D11DeviceContext> context;
    winrt::com_ptr<ID3D11Texture2D> texture;
    winrt::Windows::Graphics::SizeInt32 size{};
    HRESULT error = S_OK;
  };
  HMONITOR monitor;
  std::shared_ptr<Latest> latest = std::make_shared<Latest>();
  Direct3D11CaptureFramePool pool{nullptr};
  GraphicsCaptureSession session{nullptr};
  GraphicsCaptureItem item{nullptr};
  winrt::event_token token{}, closed_token{};
  explicit PersistentCapture(HMONITOR target) : monitor(target) {
    winrt::check_hresult(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
      nullptr, 0, D3D11_SDK_VERSION, latest->device.put(), nullptr, latest->context.put()));
    auto dxgi = latest->device.as<IDXGIDevice>();
    winrt::com_ptr<::IInspectable> inspectable;
    winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgi.get(), inspectable.put()));
    auto runtime_device = inspectable.as<IDirect3DDevice>();
    auto factory = winrt::get_activation_factory<GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
    winrt::check_hresult(factory->CreateForMonitor(monitor, winrt::guid_of<GraphicsCaptureItem>(), winrt::put_abi(item)));
    pool = Direct3D11CaptureFramePool::CreateFreeThreaded(runtime_device, DirectXPixelFormat::R16G16B16A16Float, 2, item.Size());
    session = pool.CreateCaptureSession(item);
    session.IsCursorCaptureEnabled(false);
    closed_token = item.Closed([value = latest](auto const&, auto const&) {
      std::lock_guard lock(value->mutex); value->error = HRESULT_FROM_WIN32(ERROR_INVALID_MONITOR_HANDLE); value->ready.notify_all();
    });
    token = pool.FrameArrived([value = latest, runtime_device](const Direct3D11CaptureFramePool& sender, auto const&) {
      try {
        auto frame = sender.TryGetNextFrame(); if (!frame) return;
        auto access = frame.Surface().as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
        winrt::com_ptr<ID3D11Texture2D> source;
        winrt::check_hresult(access->GetInterface(__uuidof(ID3D11Texture2D), source.put_void()));
        D3D11_TEXTURE2D_DESC description{}; source->GetDesc(&description);
        const auto size = frame.ContentSize(); bool resized = false;
        { std::lock_guard lock(value->mutex);
          D3D11_TEXTURE2D_DESC previous{}; if (value->texture) value->texture->GetDesc(&previous);
          if (!value->texture || previous.Width != description.Width || previous.Height != description.Height || previous.Format != description.Format) {
            description.Usage = D3D11_USAGE_DEFAULT; description.BindFlags = 0; description.CPUAccessFlags = 0; description.MiscFlags = 0;
            value->texture = nullptr; winrt::check_hresult(value->device->CreateTexture2D(&description, nullptr, value->texture.put()));
          }
          value->context->CopyResource(value->texture.get(), source.get());
          resized = value->size.Width != 0 && (value->size.Width != size.Width || value->size.Height != size.Height);
          value->size = size;
        }
        // Release pool-owned frames immediately, so an idle desktop has an
        // independent last-known texture and new frames never block on sampling.
        frame.Close(); value->ready.notify_all();
        if (resized && size.Width > 0 && size.Height > 0) sender.Recreate(runtime_device, DirectXPixelFormat::R16G16B16A16Float, 2, size);
      } catch (const winrt::hresult_error& error) { std::lock_guard lock(value->mutex); value->error = error.code(); value->ready.notify_all(); }
        catch (...) { std::lock_guard lock(value->mutex); value->error = E_FAIL; value->ready.notify_all(); }
    });
    try { session.StartCapture(); }
    catch (...) { item.Closed(closed_token); pool.FrameArrived(token); session.Close(); pool.Close(); throw; }
  }
  ~PersistentCapture() {
    try { if (item) item.Closed(closed_token); if (pool) pool.FrameArrived(token); if (session) session.Close(); if (pool) pool.Close(); } catch (...) {}
  }
};

Frame ScreenCapture::capture() {
  const HMONITOR monitor = MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY);
  if (!session_ || session_->monitor != monitor) session_ = std::make_shared<PersistentCapture>(monitor);
  auto pending = session_->latest;
  std::unique_lock lock(pending->mutex);
  if (!pending->ready.wait_for(lock, std::chrono::seconds(2), [&] { return static_cast<bool>(pending->texture) || FAILED(pending->error); })) throw std::runtime_error("screen_frame_timeout");
  winrt::check_hresult(pending->error);
  auto device = pending->device;
  auto context = pending->context;
  auto texture = pending->texture;
  const auto size = pending->size;
  const double observed_at = unix_time();
  /* Keep one continuous WGC session. Sampling/encoding is throttled, not the
     capture lifetime; static desktops reuse the independent latest texture. */
    D3D11_TEXTURE2D_DESC description{}; texture->GetDesc(&description);
    const UINT width = std::min(description.Width, static_cast<UINT>(std::max(0, size.Width)));
    const UINT height = std::min(description.Height, static_cast<UINT>(std::max(0, size.Height)));
    if (!width || !height || static_cast<std::uint64_t>(width) * height > 64'000'000) throw std::runtime_error("invalid_screen_dimensions");
    description.Usage = D3D11_USAGE_STAGING; description.BindFlags = 0; description.CPUAccessFlags = D3D11_CPU_ACCESS_READ; description.MiscFlags = 0;
    winrt::com_ptr<ID3D11Texture2D> staging;
    winrt::check_hresult(device->CreateTexture2D(&description, nullptr, staging.put()));
    context->CopyResource(staging.get(), texture.get());
    std::vector<std::uint8_t> pixels(static_cast<size_t>(width) * height * 4);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    winrt::check_hresult(context->Map(staging.get(), 0, D3D11_MAP_READ, 0, &mapped));
    float peak = 1;
    for (UINT y = 0; y < height; ++y) {
      const auto* row = reinterpret_cast<const std::uint16_t*>(static_cast<const BYTE*>(mapped.pData) + y * mapped.RowPitch);
      for (UINT x = 0; x < width; ++x) peak = std::max(peak, 0.2126f * half_to_float(row[x * 4]) + 0.7152f * half_to_float(row[x * 4 + 1]) + 0.0722f * half_to_float(row[x * 4 + 2]));
    }
    for (UINT y = 0; y < height; ++y) {
      const auto* row = reinterpret_cast<const std::uint16_t*>(static_cast<const BYTE*>(mapped.pData) + y * mapped.RowPitch);
      for (UINT x = 0; x < width; ++x) {
        float r = half_to_float(row[x * 4]), g = half_to_float(row[x * 4 + 1]), b = half_to_float(row[x * 4 + 2]);
        const float luminance = std::max(0.0f, 0.2126f * r + 0.7152f * g + 0.0722f * b);
        const float scale = peak > 1.01f ? (1 + luminance / (peak * peak)) / (1 + luminance) : 1;
        const size_t offset = (static_cast<size_t>(y) * width + x) * 4;
        pixels[offset] = linear_to_srgb(b * scale); pixels[offset + 1] = linear_to_srgb(g * scale);
        pixels[offset + 2] = linear_to_srgb(r * scale); pixels[offset + 3] = 255;
      }
    }
    context->Unmap(staging.get(), 0);
    lock.unlock();
    const double shrink = std::min(1.0, 960.0 / std::max(width, height));
    Frame result; result.id = unique_id(); result.observed_at = observed_at;
    result.width = std::max(1U, static_cast<UINT>(width * shrink)); result.height = std::max(1U, static_cast<UINT>(height * shrink));
    for (const float quality : {0.8f, 0.6f, 0.4f, 0.25f}) {
      result.jpeg = jpeg_image(pixels, width, height, result.width, result.height, quality);
      if (result.jpeg.size() <= 200 * 1024) return result;
    }
    throw std::runtime_error("screen_jpeg_budget_exceeded");
}

void ScreenCapture::start(int interval, bool exclude) {
  stop(); interval_ = interval; exclude_ = exclude;
  if (!supported()) { report_("unavailable", "windows_graphics_capture_unavailable"); return; }
  worker_ = std::jthread([this](std::stop_token stop) {
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    while (!stop.stop_requested()) {
      try {
        if (!desktop_unlocked()) { session_.reset(); slot_.clear(); report_("unavailable", "desktop_locked_or_unavailable"); }
        else if (!host_excluded()) { session_.reset(); slot_.clear(); report_("unavailable", "self_exclusion_unavailable"); }
        else {
          auto frame = capture();
          if (!stop.stop_requested() && desktop_unlocked() && host_excluded()) {
            auto saved = slot_.store(std::move(frame), unix_time());
            if (saved) { report_("running", ""); emit_("environment_frame_available", frame_metadata(*saved)); }
          }
        }
      } catch (const winrt::hresult_error& error) { session_.reset(); slot_.clear(); report_("failed", "screen_capture_" + error_code(error.code())); }
        catch (const std::exception& error) { session_.reset(); slot_.clear(); report_("failed", error.what()); }
      std::unique_lock lock(wait_mutex_);
      wait_.wait_for(lock, std::chrono::milliseconds(interval_.load()), [&] { return stop.stop_requested(); });
    }
    session_.reset(); slot_.clear(); winrt::uninit_apartment();
  });
}
void ScreenCapture::stop() { if (worker_.joinable()) { worker_.request_stop(); wait_.notify_all(); worker_.join(); } slot_.clear(); }
void ScreenCapture::update_interval(int interval) { interval_ = interval; wait_.notify_all(); }
std::optional<Frame> ScreenCapture::claim(const std::string& id) {
  if (!desktop_unlocked() || !host_excluded()) { slot_.clear(); return std::nullopt; }
  return slot_.claim(id, unix_time());
}
}
