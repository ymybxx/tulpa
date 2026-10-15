# 桌面端开发、调试与测试方法

[README](../README.md) | [Troubleshooting](troubleshooting.md)

这份文档总结在 `desktop/pet`（tulpa 桌面端）上改界面、查问题、做验证的做法。配套工具在 `desktop/pet/scripts/dev/`。

## 原则

- **开发时用开发版，不打包。** 改完只重建改动的部分，然后重启或刷新开发版。只有要交付、或要验证打包本身时才运行 `npm run dist:mac:local`。
- **先确认触发条件，再改。** 先复现，再加记录找到具体是哪段代码、哪一帧出了问题，然后改根因。修完用同一种测量方法再测一遍，确认问题消失。
- **像新手一样用真实鼠标试。** 调试接口发出的点击走的是页面事件，碰不到 macOS 层面的行为（桌面空间切换、App 激活、窗口层级、点击穿透）。界面改完，最后要用真实鼠标移动、悬停、点击、右键、拖动、滚轮试一遍。
- **用真实组件验证。** 真实模型、真实形象文件、真实服务；不在界面输入框里填真实 API Key；不去改系统权限和安全设置。
- **用完关掉 tulpa**，避免后台继续消耗 token。

## 构建与启动

| 改了什么 | 重建 | 生效方式 |
|---|---|---|
| 页面（`src/`） | `npm run build:renderer` | 刷新对应页面即可（见下文 `eval … location.reload()`） |
| 主进程（`electron/`） | `npm run build:electron` | 重启开发版 |
| 原生模块（`native/pet-input`） | `npm run build:pet-input:mac` | 重启开发版（已加载的 `.node` 不会重新加载） |

启动（或重启）开发版：

```bash
desktop/pet/scripts/dev/start-dev-app.sh
```

- 数据目录默认是 `~/Library/Application Support/tulpa-dev`，和平时用的 tulpa 分开。也可以传入别的目录作参数。不要放在 `$TMPDIR`：macOS 会清掉几天没访问的文件，模型和设置会丢。
- 用 `open -n -g` 在后台启动，不抢当前 App 的焦点；开发版是独立进程，不是终端的子进程。
- 页面调试端口 `:9230`，主进程调试端口 `:9231`，输出写到 `<数据目录>/app.log`。
- 同一时间只保留一个开发版：脚本会先关掉旧实例和它的本地服务。

打包前必须退出所有 tulpa，包括开发版和平时用的打包版。`scripts/assert-packaging-idle.mjs` 会检查这一点：运行中覆盖 PyInstaller 的文件，会让后续模块解压失败。

## 用 CDP 查看和操作页面

`scripts/dev/cdp.mjs` 里的 `<page>` 写页面网址的结尾：`main.html` 是主窗口，`index.html` 是桌面上的 TA，`pet-chat.html` 是 TA 旁边的悬浮层。

```bash
node scripts/dev/cdp.mjs pages
node scripts/dev/cdp.mjs eval index.html 'window.desktopPet.showMainView("looks")'
node scripts/dev/cdp.mjs eval main.html 'location.reload()'
node scripts/dev/cdp.mjs eval main.html 'document.querySelector(".le-tabs button.active")?.textContent'
node scripts/dev/cdp.mjs shot main.html /tmp/main.png
node scripts/dev/cdp.mjs click main.html 640 420
node scripts/dev/cdp.mjs drag index.html 900 400 -120 0
node scripts/dev/cdp.mjs wheel index.html 900 400 -8 30 16
```

- `eval` 适合读取界面状态，例如当前页签、元素是否隐藏、滚动位置、`localStorage`。判断「某个操作后页面有没有跳」，就在操作前后各读一次 `scrollTop`。
- 页面里模块内部的变量读不到。需要观察时，在页面里临时挂监听，例如 `window.desktopPet.onPetPanel(p => window.__log.push(p))`，操作完再读 `window.__log`。
- `click`、`drag`、`wheel` 走 Chromium 的输入事件，页面处理和真实鼠标一样，但不经过 macOS。

## 调试主进程

```bash
node scripts/dev/cdp.mjs main 'require("electron").BrowserWindow.getAllWindows().map(w => [w.webContents.getURL().split("/").pop(), w.getBounds(), w.isVisible(), w.isFocused()])'
```

- 主进程里 `require("electron")` 可以用，但 `main.ts` 模块内部的变量拿不到。
- **挂事件看时序**：`app.on("did-become-active")`、窗口的 `focus`/`blur`/`show`/`hide`，把时间戳推进 `globalThis.__log`，操作完读出来。
- **包装方法看调用**：把某个窗口的 `setBounds`、`setPosition` 包一层并记录时间和参数，就能看出一次操作里窗口动了几次。缩放抖动就是这样查出来的：每一步窗口被挪了两次。
- 需要看模块内部状态时，就临时加一行 `console.log`，从 `app.log` 里读。查完删掉，不要提交。

## 什么时候必须用真实鼠标

下面这些只有真实鼠标（computer-use）能验证：

- 悬停出现、点击穿透到后面的 App、右键菜单、拖动、滚轮缩放；
- 在别的 App 全屏时的表现、桌面空间切换、tulpa 何时变成前台；
- 首次点击、焦点、键盘输入去了哪个窗口。

注意事项：

- **按下之前先悬停一下。** TA 的窗口平时让鼠标穿透，需要收到一次移动事件才会切换成「这里是 TA，接住点击」。直接按下的拖动工具会穿透到后面。
- **窗口在别的桌面空间上时：** 后台点击会被拒绝，截图可能是旧的；Chromium 会把页面当成隐藏，停止 `requestAnimationFrame`，预览不播，CDP 的滚轮事件也可能卡住。先把窗口带回当前空间，或者用 `cdp.mjs shot`（它会让页面临时重画一帧）。
- 模拟「别的 App 在前台全屏」：打开一个全屏的「文本编辑」，并用 `osascript -e 'tell application "TextEdit" to activate'` 让它真正成为前台。只从后台菜单切换全屏，前台还是原来的 App，结论会错。
- 确认谁在前台：`lsappinfo info -only name $(lsappinfo front)`。
- 用户正在用电脑、前台是别的 App 时，不要抢鼠标，改用 CDP 验证页面逻辑，再请用户亲手试。

## 测量流畅度和抖动

截图只能看到某一瞬间，看不出抖动。要逐帧记录：

```bash
# 连续 2 秒，每次屏幕刷新都记录 TA 框的位置和高度
node scripts/dev/cdp.mjs frames index.html '[document.getElementById("pet-panel").style.left, document.getElementById("pet-panel").style.height]' 2000
```

- 边记录边在另一个终端用 `cdp.mjs wheel` 或 `drag` 施加操作，然后看数据：有没有来回（数值回退），有没有丢帧（连续几帧不变、下一帧跳一大步），每步增量是否均匀。
- 屏幕刷新率可能是 120 Hz（相邻两次记录约 8 ms）。
- 查「窗口变化」和「页面重画」的时间差：主进程包装 `setBounds` 记时间，页面里监听 `resize` 记时间，两边对比。差一帧以上，就会看到位置不对的旧画面。
- `Page.startScreencast` 可以抓页面画面再算物体的边界框，但它自己会丢帧，只能看趋势，不能判断是否均匀。
- 录屏（`screencapture -v`）需要屏幕录制权限，不要为此去改系统设置。

## macOS 桌宠相关结论

这些是查问题时确认过的系统行为，改桌面端相关代码前先看一遍：

- **窗口移动或改大小时，会先显示上一帧在错的位置，等页面重画后才纠正。** 所以 TA 的窗口铺满屏幕、固定不动（`electron/pet-stage.ts`）。拖动和缩放只改页面里的框，位置、大小和画面在同一帧内更新。
- **触控板捏合只发给真正在前台的 App。** 没有「输入监控」权限时，系统级事件监听也只能收到发给自己的事件。所以 TA 只用双指滑动或鼠标滚轮缩放。
- **tulpa 被激活时，如果主窗口开在别的桌面，macOS 会把人拉到那个桌面。** 所以不要在悬停或点 TA 时激活 tulpa；原生模块里的激活在这种情况下会主动拒绝。
- **TA 的窗口是不会激活 App 的面板。** 点它能接收输入，但 `NSApp` 不会变成激活状态（`lsappinfo` 显示它在前台，Electron 却收不到 `did-become-active`）。
- **面板窗口隐藏后再显示时，如果当前是别的 App 的全屏空间，它不会出现在那里。** 所以悬浮层窗口只要 TA 在就一直显示，没内容时透明并让鼠标穿透。
- **摆放规则不能在自己引起的变化上重复套用。** 「至少露出一部分」的规则曾在框每次移动后重新执行，每次都给同一个偏移，TA 就一路滑到屏幕边。现在只有模型边界真的变了才重新摆放。

## 排查流程

1. 用开发版复现，记下精确的复现步骤（例如「点一下后面的 Claude，再在 TA 上捏合」）。
2. 在关键环节加记录：页面事件、主进程调用、窗口事件。确认问题出在哪一环，例如「事件根本没进来」「进来了但判断错了」「判断对了但画错了」。
3. 改根因，不用牺牲已有功能来绕开问题。
4. 用第 2 步的同一套记录再测，确认数据正常。
5. 用真实鼠标按用户的方式走一遍。
6. 跑测试（`npx vitest run`、`npx tsc --noEmit -p .`、`-p tsconfig.electron.json`），提交，关掉开发版。
