# 参与贡献

感谢你愿意改进 tulpa。请先阅读[行为准则](CODE_OF_CONDUCT.md)；发现安全问题请按 [SECURITY.md](SECURITY.md) 私下报告，不要开公开 Issue。

## 改动之前

大的功能、存储结构变化、新的模型或服务接入，请先开 Issue 说明：

- 用户遇到的问题是什么（而不只是想怎么实现）；
- 对回复速度、记忆准确度、token 用量和已有用户数据的影响；
- 怎样复现、怎样验证效果。

小的修复、测试和文档改动可以直接提 Pull Request。

## 开发环境

| 需要 | 说明 |
| --- | --- |
| 系统 | macOS 12.3+（Apple 芯片）或 Windows 11 x64 |
| Python | 3.11，用 [uv](https://docs.astral.sh/uv/) 管理 |
| Node.js | 18 或更高 |
| macOS 原生组件 | Xcode Command Line Tools（构建音频、屏幕感知等 Swift 助手） |

```bash
git clone https://github.com/ymybxx/tulpa.git
cd tulpa
uv sync --extra voice --dev
cd desktop/pet
npm ci
npm run dev        # Windows: npm run dev:win
```

开发模式下，Electron 直接用仓库里的 `uv` 环境启动本地 Python 服务，改界面不需要重新打包 Python。第一次启动会进入引导，按提示填自己的模型服务即可。语音和 Live2D 运行时会下载到应用数据目录，不会放进仓库。

更多说明：

- 桌面端结构、打包与签名：[desktop/pet/README.md](desktop/pet/README.md)
- 本地调试与自动化测试：[docs/desktop-dev-testing.md](docs/desktop-dev-testing.md)
- Windows：[docs/windows-desktop.md](docs/windows-desktop.md)
- 本地服务的架构与接口：[docs/architecture.md](docs/architecture.md)、[docs/api.md](docs/api.md)

## 提交前检查

```bash
# Python 本地服务
uv run ruff format --check src tests
uv run ruff check .
uv run pytest -q

# 桌面端
cd desktop/pet
npm run typecheck
npm test
```

需要真实 PostgreSQL 或真实模型服务的测试默认会跳过；单元测试不能依赖真实的模型服务或密钥。

## Pull Request

- 一个 PR 只做一件事，说明改了什么、为什么改、怎么验证的。
- 界面改动附截图；macOS 和 Windows 都受影响的改动，两边都要顾到。
- 不要提交密钥、`.env`、聊天记录、录音、截图里的个人信息，也不要提交 Live2D 模型等你没有再分发授权的素材。
- 改了用户能感知的行为，在 `CHANGELOG.md` 的 `[Unreleased]` 里写一条。
