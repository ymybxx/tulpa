// WASAPI process-loopback activation follows Microsoft's ApplicationLoopback
// sample (MIT). See audio-LICENSE.txt for attribution and license.
#include "audio.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <audioclient.h>
#include <audioclientactivationparams.h>
#include <mmdeviceapi.h>
#include <wrl.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace tulpa::windows {
namespace {
using Microsoft::WRL::ComPtr;
constexpr DWORD kSampleRate = 16'000;

struct Handle {
  HANDLE value = nullptr;
  explicit Handle(bool manualReset = false)
      : value(CreateEventW(nullptr, manualReset, FALSE, nullptr)) {
    if (!value) throw std::runtime_error("system_audio_event_creation_failed");
  }
  ~Handle() { CloseHandle(value); }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
};

struct AudioError : std::exception {
  std::string reason;
  AudioError(const char* stage, HRESULT result) {
    char code[16];
    std::snprintf(code, sizeof(code), "0x%08lX", static_cast<unsigned long>(result));
    reason = std::string(stage) + ":" + code;
  }
  const char* what() const noexcept override { return reason.c_str(); }
};

void check(HRESULT result, const char* stage) {
  if (FAILED(result)) throw AudioError(stage, result);
}

struct Activation {
  Handle ready{true};
  std::mutex mutex;
  HRESULT result = E_PENDING;
  ComPtr<IAudioClient> client;
};

// The activation may complete after stop/timeout. Its owned state must outlive
// the worker's wait and must not retain the capture object or call its callbacks.
class ActivationHandler final
    : public Microsoft::WRL::RuntimeClass<
          Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
          IActivateAudioInterfaceCompletionHandler, Microsoft::WRL::FtmBase> {
 public:
  explicit ActivationHandler(std::shared_ptr<Activation> state)
      : state_(std::move(state)) {}
  STDMETHOD(ActivateCompleted)(IActivateAudioInterfaceAsyncOperation* operation)
      override {
    HRESULT activationResult = E_FAIL;
    ComPtr<IUnknown> audio;
    HRESULT result = operation->GetActivateResult(&activationResult, &audio);
    if (SUCCEEDED(result)) result = activationResult;
    {
      std::lock_guard lock(state_->mutex);
      if (SUCCEEDED(result)) result = audio.As(&state_->client);
      state_->result = result;
    }
    SetEvent(state_->ready.value);
    return S_OK;
  }

 private:
  std::shared_ptr<Activation> state_;
};

std::uint64_t monotonicNs() {
  LARGE_INTEGER count{}, frequency{};
  QueryPerformanceCounter(&count);
  QueryPerformanceFrequency(&frequency);
  return static_cast<std::uint64_t>(
      static_cast<long double>(count.QuadPart) * 1'000'000'000 / frequency.QuadPart);
}
}  // namespace

struct SystemAudioCapture::Impl {
  std::uint32_t hostProcessId;
  PcmCallback onPcm;
  StatusCallback onStatus;
  Handle stopEvent{true};
  std::thread worker;

  Impl(std::uint32_t processId, PcmCallback pcm, StatusCallback status)
      : hostProcessId(processId), onPcm(std::move(pcm)),
        onStatus(std::move(status)) {
    if (!hostProcessId || !onPcm) throw std::invalid_argument("invalid_audio_capture_host");
  }

  bool stopped() const { return WaitForSingleObject(stopEvent.value, 0) == WAIT_OBJECT_0; }
  void status(const std::string& state, const std::string& reason = {}) noexcept {
    try { if (onStatus) onStatus(state, reason); } catch (...) { }
  }

  void capture() {
    auto activation = std::make_shared<Activation>();
    auto completion = Microsoft::WRL::Make<ActivationHandler>(activation);
    if (!completion) throw AudioError("system_audio_activation_allocation", E_OUTOFMEMORY);
    AUDIOCLIENT_ACTIVATION_PARAMS parameters{};
    parameters.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK;
    parameters.ProcessLoopbackParams.TargetProcessId = hostProcessId;
    parameters.ProcessLoopbackParams.ProcessLoopbackMode =
        PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE;
    PROPVARIANT property{};
    property.vt = VT_BLOB;
    property.blob.cbSize = static_cast<ULONG>(sizeof(parameters));
    property.blob.pBlobData = reinterpret_cast<BYTE*>(&parameters);
    ComPtr<IActivateAudioInterfaceAsyncOperation> operation;
    check(ActivateAudioInterfaceAsync(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
          __uuidof(IAudioClient), &property, completion.Get(), &operation),
          "system_audio_activation_failed");
    HANDLE activationEvents[] = {stopEvent.value, activation->ready.value};
    DWORD wait = WaitForMultipleObjects(2, activationEvents, FALSE, 10'000);
    if (wait == WAIT_OBJECT_0) return;
    if (wait != WAIT_OBJECT_0 + 1)
      throw AudioError("system_audio_activation_timeout", HRESULT_FROM_WIN32(ERROR_TIMEOUT));
    ComPtr<IAudioClient> client;
    {
      std::lock_guard lock(activation->mutex);
      check(activation->result, "system_audio_activation_failed");
      client = activation->client;
    }
    Handle samplesReady;
    WAVEFORMATEX format{};
    format.wFormatTag = WAVE_FORMAT_PCM;
    format.nChannels = 1;
    format.nSamplesPerSec = kSampleRate;
    format.wBitsPerSample = 16;
    format.nBlockAlign = 2;
    format.nAvgBytesPerSec = kSampleRate * format.nBlockAlign;
    // The shared audio engine converts the application mix to our wire format.
    // Failure stays explicit; never relabel native 44.1/48 kHz as 16 kHz PCM.
    check(client->Initialize(AUDCLNT_SHAREMODE_SHARED,
          AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK |
          AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
          0, 0, &format, nullptr), "system_audio_format_failed");
    check(client->SetEventHandle(samplesReady.value), "system_audio_event_failed");
    ComPtr<IAudioCaptureClient> captureClient;
    check(client->GetService(IID_PPV_ARGS(&captureClient)), "system_audio_service_failed");
    check(client->Start(), "system_audio_start_failed");
    struct StopClient {
      IAudioClient* client;
      ~StopClient() { client->Stop(); }
    } stopClient{client.Get()};
    status("running");
    HANDLE events[] = {stopEvent.value, samplesReady.value};
    bool firstPacket = true;
    while (!stopped()) {
      const DWORD result = WaitForMultipleObjects(2, events, FALSE, 1'000);
      if (result == WAIT_OBJECT_0) return;
      // An idle system may produce no packets. Do not invent captured silence.
      if (result == WAIT_TIMEOUT) continue;
      if (result != WAIT_OBJECT_0 + 1)
        throw AudioError("system_audio_wait_failed", HRESULT_FROM_WIN32(GetLastError()));
      UINT32 packetFrames = 0;
      check(captureClient->GetNextPacketSize(&packetFrames), "system_audio_device_lost");
      while (packetFrames && !stopped()) {
        BYTE* data = nullptr;
        UINT32 frames = 0;
        DWORD flags = 0;
        UINT64 devicePosition = 0, qpcPosition = 0;
        check(captureClient->GetBuffer(&data, &frames, &flags, &devicePosition,
                                      &qpcPosition), "system_audio_read_failed");
        std::vector<std::int16_t> pcm;
        try {
          pcm.resize(frames, 0);
          if (!(flags & AUDCLNT_BUFFERFLAGS_SILENT) && data)
            std::memcpy(pcm.data(), data, frames * sizeof(std::int16_t));
        } catch (...) {
          captureClient->ReleaseBuffer(frames);
          throw;
        }
        check(captureClient->ReleaseBuffer(frames), "system_audio_release_failed");
        if (!firstPacket && (flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY))
          status("running", "capture_discontinuity");
        firstPacket = false;
        // WASAPI returns QPC position in 100 ns units, not performance-counter ticks.
        auto timestamp = qpcPosition * 100;
        if (flags & AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR) {
          timestamp = monotonicNs();
          const auto duration = static_cast<std::uint64_t>(frames) * 1'000'000'000 / kSampleRate;
          timestamp = timestamp > duration ? timestamp - duration : 0;
        }
        if (!stopped() && !pcm.empty()) onPcm(timestamp, pcm);
        check(captureClient->GetNextPacketSize(&packetFrames), "system_audio_device_lost");
      }
    }
  }

  void run() noexcept {
    const HRESULT comResult = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(comResult)) {
      status("failed", AudioError("system_audio_com_failed", comResult).what());
      return;
    }
    DWORD retryMs = 500;
    while (!stopped()) {
      try { capture(); }
      catch (const std::exception& error) { if (!stopped()) status("failed", error.what()); }
      catch (...) { if (!stopped()) status("failed", "system_audio_capture_failed"); }
      if (WaitForSingleObject(stopEvent.value, retryMs) == WAIT_OBJECT_0) break;
      retryMs = std::min<DWORD>(5'000, retryMs * 2);
    }
    CoUninitialize();
    status("stopped");
  }
};

SystemAudioCapture::SystemAudioCapture(std::uint32_t hostProcessId,
                                     PcmCallback onPcm, StatusCallback onStatus)
    : impl_(std::make_unique<Impl>(hostProcessId, std::move(onPcm), std::move(onStatus))) {}
SystemAudioCapture::~SystemAudioCapture() { stop(); }
void SystemAudioCapture::start() {
  if (impl_->worker.joinable()) return;
  ResetEvent(impl_->stopEvent.value);
  impl_->worker = std::thread([this] { impl_->run(); });
}
void SystemAudioCapture::stop() {
  SetEvent(impl_->stopEvent.value);
  if (impl_->worker.joinable()) impl_->worker.join();
}

}  // namespace tulpa::windows
