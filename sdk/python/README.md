# Fish Python Plugin SDK · 0.1

宿主可以闭源。开发插件只需要这个 SDK 和安装好的 Fish Memory App，无需数据库、模型密钥、音频库或宿主源码。

## 开始开发

当前 SDK 尚未发布到 PyPI。先安装开发者分发的 `fish_plugin_sdk-0.1.3-py3-none-any.whl`，需要 CLI 构建工具时安装其 `[tools]` extra。

```sh
fish-plugin init my-plugin
fish-plugin check my-plugin
fish-plugin dev my-plugin
```

App → 插件 → 加载开发目录，选择宠物和正在播放的会话。在“开发与事件测试”中启用开发模式，再执行 `dev`。也可先在本体加载目录、填写配置，然后从 CLI 热重载。

`dev` 只重启当前插件，关闭 CLI 会停止此实例。默认发现文件在 macOS 的 `~/Library/Application Support/fish-memory-desktop/plugin-dev.json`、Windows 的 `%APPDATA%/fish-memory-desktop/plugin-dev.json`；可用 `--app-file` 指定其他 App 安装位置。此文件是短期开发凭据，关闭开发模式或退出 App 就失效，不要分享。

## 一份可运行的插件

```python
import asyncio
from pydantic import BaseModel, Field
from fish_plugin_sdk import Event, Pet, Plugin

class Notice(BaseModel):
    text: str = Field(max_length=500, description="这次发生的事情")

NOTICE = Event("example.notice", Notice, "结合当前情境表达自己的看法。", speak=True)

class MyPlugin(Plugin):
    events = (NOTICE,)

    async def run(self, pet: Pet):
        await pet.push(NOTICE, Notice(text="窗外下雨了"))
        await asyncio.Event().wait()
```

`fish-plugin.json` 声明 ID、名称、版本和入口 `my_plugin:MyPlugin`；`pyproject.toml` 声明依赖。模块导入、类声明、`describe()` 不做网络、线程或其他业务操作。连接与任务全部放在 `run()`，用 `async with` / `finally` 清理。停止时收到取消，不能吞掉取消继续运行。

`Plugin.Config` 是 Pydantic 配置模型，App 按 JSON Schema 生成表单。首版 UI 支持字符串、数字、布尔值和枚举；复杂业务数据放在事件 Schema 中。秘密字段使用 `Field(json_schema_extra={"writeOnly": True})`，App 加密保存，后续显示“已保存”，不会读回明文。配置通过进程标准输入交付，不放命令行或事件数据中。插件可使用 `FISH_PLUGIN_DATA_DIR` 保存自己的平台游标等小量状态。

## 核心接口

持续活动可在插件类上声明一次（SDK 0.1.2）：

```python
from fish_plugin_sdk import Activity

class MyPlugin(Plugin):
    activity = Activity(
        name="直播主持",
        role="正在主持自己直播的主播",
        audience="观众；本机用户是搭档",
        objective="自己选择话题、表达观点，并根据回应继续展开。",
    )
    events = (NOTICE,)
```

`activity` 说明现在在做什么，宠物原来的名称、性格和记忆继续保留。SDK 连接时与事件一起注册，断线重连会重新注册。对于 `autonomous:true` 的插件，这份情景也进入绑定宠物、绑定会话的普通聊天和语音预生成；停止或断线立即失效，不修改长期性格，也不影响其他会话。普通临时事件插件仍可省略它。

事件 Schema 的描述在本次事件任务中传递；历史作为已发生的对话记录，最新用户安排单列。调用 `pet.state` 可查看实际注册的 `activity`。这些行为均不需要额外的 LLM 调用。

| 接口 | 返回/含义 |
|---|---|
| `await pet.push(EVENT, data, event_id=...)` | 已落库的事件回执；不等于已说完。自动分配 ID，传输结果不确定时最多重试一次且使用同一 ID |
| `await pet.context("history", "memory")` | 已完成的同源上下文，不生成新摘要。字段可选 personality / history / memory / perception / events |
| `pet.state` / `await pet.refresh_state()` | 最近状态/主动查询。`can_respond` 仅是提示，提交时仍由宿主原子判断 |
| `async for update in pet.notifications()` | state / context_changed / response；单消费者。通知可能因断线丢失 |
| `await pet.result(event_id)` | 从持久化事件恢复权威结果；完成时 text 只包含实际播出的部分 |

消息与通知为字典，字段见 [协议](PROTOCOL.md)。context 的 `revision` 是内容摘要，`as_of` 是读取时刻，不代表每份感知刚刚生成。历史只含已播出或正常可见的消息；感知仍服从用户现有权限与开关。大上下文超出单帧限额时返回错误，可按字段分开读取；不发送图片/音频字节。

`speak` 只接受布尔值，默认 true；false 只记录，true 在宿主允许响应时调用模型生成发言。auto 已移除。该参数只在宿主调度时使用，不传给模型；模型输出固定为 m（情绪）、a（台词），不再判断是否发言。生成、准备 TTS 或实际播放占用期间，新事件只记录，之后不会补播。中断后的未播放内容不会当成说过的话。一个持续自主插件设置 `autonomous:true`，临时接管此宠物的自主机会；停止后原陪伴按用户现有开关恢复。

## 检查和发行

```sh
fish-plugin check .
fish-plugin build . --sdk-wheel /path/to/fish_plugin_sdk-0.1.3-py3-none-any.whl
```

`check` 检查静态元信息和当前环境中的实际导入/事件声明。它明确输出 `live_app=not_run` 和 `other_platforms=not_run`，不把本地导入当成实机发声或全平台认证。本体事件测试走正式受理、LLM 和原生播放链路，可重发同一 ID 检查去重；发送 speak 事件会使用宠物已配置的模型服务。

`build` 在当前平台下载仅 wheel 的依赖集合，封入 `.fish-plugin`，记录每个文件的 SHA-256、当前平台/Python 版本和检查报告。安装端离线安装这些文件，不重新解析、源码编译或运行系统 pip。当前为本机产物，跨平台发布须在各目标平台分别构建、导入和运行契约用例，再发布对应产物索引。不能把 macOS 的原生 wheel 标成 Windows 可用；市场及多平台 App 联调尚未包含在这个实验版。

插件是当前用户运行的原生 Python 代码，独立进程和 venv 用于依赖、生命周期隔离，不是阻止恶意代码访问文件的安全沙箱。只安装信任的代码。SDK 不提供宿主私有 Python 对象、模型钥匙或管理 API；停止后实例凭据撤销。

## 运行建议（可选，SDK 0.1.1）

插件可以声明评估目标，宿主用配置中的推理模型分析启动后的真实运行片段：

```python
from fish_plugin_sdk import Plugin, ReviewCriterion, ReviewProfile

class MyPlugin(Plugin):
    review = ReviewProfile(
        title="互动效果",
        goal="围绕具体内容表达观点，并接住用户的回应。",
        criteria=[ReviewCriterion(name="话题推进", description="同一话题的后续发言需要有新增内容。")],
    )
```

`review` 通过 `describe()` 交给 App，不加入事件 `description` 或宠物发言提示词。没有声明的插件继续按原方式运行。可选 `references` 保存参考来源，`mechanism` 描述开发者希望评估者了解的实现机制；不要放连接密钥或秘密配置。

宿主约每三分钟检查一次，至少六个终态事件后生成建议。也可以在插件页点“分析当前片段”（至少两个终态事件）。只把设备确认的已播前缀当作发言；生成失败、打断和存储事件分别统计。最多读取本窗口 40 个插件事件、20 条用户消息，模型分析有大小限制的近期样本。不会读取整个数据库。

建议包含时间范围、插件版本、运行编号、问题、证据和改进建议，保存于宿主插件目录的 `reviews/`，不会进入真实聊天或 Fact。插件页直接查看历史和更早记录；停止或重启 App 后历史仍在。该能力只提供建议，不修改插件代码、事件描述、参数、性格或记忆。


## 插件辅助推理（SDK 0.1.3）

插件可以使用 App 当前配置的**推理模型**，无需拿到模型密钥。插件决定推理任务、结果结构以及后续动作，宿主只负责同源上下文、模型请求、校验、取消和记录。

```python
from pydantic import BaseModel

class Selection(BaseModel):
    subject: str
    opinion: str

selection = await pet.reason(
    "从当前素材选一个还没谈过的具体点，形成自己的判断和依据。",
    Selection,
    fields=("personality", "history", "perception"),
)
# 如何使用 selection 由插件决定；reason 本身不会说话或操作设备。
```

`data=` 可以带插件自己的状态；`fields=` 指定使用哪些已发布上下文。返回值是经过 Schema 校验的 Pydantic 对象，不是聊天消息。调用会产生模型费用；最多一个推理请求在途，全宿主最多两个，不自动重试。取消此协程、断开插件、停止插件或进入用户语音交流会取消尚未完成的推理。较慢的推理不阻塞同一连接上的事件推送、状态查询和取消请求。输入与结果记录在已有 observations 的 `plugin.reasoning` 中，截图只保存元信息。

直播插件在两轮之间的等待时间里提前选择内容，准备完成后才推送 `live.idle`。准备耗时仍取决于配置的推理模型，等待时间不足以覆盖它时，下一轮会晚一些开始。

两个可选事件配置用于避免重复传递中间状态：

```python
IDLE = Event(
    "live.idle", PreparedContent, "用自己的口吻表达这份已经准备好的内容。",
    speak=True,
    include_in_context=False,
)
```

`include_in_context=False` 仍然存储事件并保留诊断，但不把调度信号递归加入之后的事件历史。最终发言由宿主固定组装：角色、最近最多 50 条真实消息、Static/Dynamic 摘要、感知，以及当前事件。SDK 只定义事件数据和任务含义，不能覆盖发言格式或裁掉基础上下文。旧 `context_fields` 参数只为加载旧插件而接受，不再生效。插件辅助分析仍可使用 `pet.context(fields)` 和 `pet.reason(fields=...)` 选择并自行组织材料。

字段的 `json_schema_extra={"x-fish-context": "identity"}` 将原始身份投影为稳定的匿名标识。同一插件的多种事件需要关联同一个人物时，可在对应字段增加相同的 `"x-fish-identity-scope"`（例如 `"live.reply"`）；未声明时仍按事件类型隔离。该范围标记和原始身份仅用于宿主投影，不进入模型可见的 Schema。

事件数据模型可通过 `ConfigDict(json_schema_extra={"x-fish-history-template": "观众「${viewer_name}」说：${text}"})` 声明自然语言 History 格式。宿主接收事件时按 `string.Template` 做一次字段替换，将结果保存为 `context_text`；不调用模型，原始事件数据不变。占位符只能引用已声明的内容字段，不能引用 `metadata` 或 `identity` 字段；替换值中的占位符保持原样。未声明模板的插件仍使用原有 History 格式，模板本身不进入模型可见的 Schema。


## 后台 Task

使用本体已配置的 Agent。无需向插件提供模型或 Agent 密钥；插件自己组织取材。

```python
from uuid import uuid4

context = await pet.context("history", "memory", "perception")
if pet.tasks.available:
    request_id = uuid4()  # 保存此 ID；回执丢失时查它，不创建另一个请求
    job = await pet.tasks.start(
        "核实当前产品已公布的功能和版本，给出简短结论及来源。",
        request_id=request_id,
        context_key="product-A:activation-1",
        reference={"subject": "产品 A", "question": "是否支持离线使用"},
        context_revision=context["revision"],
    )
    status = await pet.tasks.get(job.run_id)
    # 收到 task_changed 提示后查询；不要在发言路径等待。
    if status.result_available:
        page = await pet.tasks.result(job.run_id)
    # 需要放弃时：await pet.tasks.cancel(job.run_id)
```

`start` 只等待持久化排队，不等待网络探测或任务完成；普通主持和用户聊天继续。
同一实例最多一个任务在途，用户任务可并发。未配置 Agent 返回 `task_unavailable`。
回执不明时 `get(request_id=request_id)`，或使用完全相同的参数和 ID 重交；不同内容会报冲突。
`get` 是权威状态；`task_changed` 与其他通知共用现有通知流，只使用一个消费循环分发。
`result` 每页最多 16 KB，使用 `next_cursor` 读后续；它不表示资料已被核实，也不会触发说话。
任务结果的 `context_key` 由宿主原样回传，插件须确认还属于当前话题后再使用。
取消只确认本地停止意图；`stop_requested` 不代表远端已停止。状态 unknown 仍占在途槽。
需要授权、登录或用户输入时研究停止；宿主不会替自主任务答复审批，也不向普通聊天插入任务结果。
插件停止/重启、换宠物或会话后，旧任务不能被新实例读取、取消或自动采用。
