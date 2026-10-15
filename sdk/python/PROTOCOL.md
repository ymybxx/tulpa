# External event protocol v1

连接：`ws://127.0.0.1:<app-port>/v1/extensions/ws?protocol=1`，`Authorization: Bearer <instance-token>`。默认禁用压缩，UTF-8 JSON 单帧 64 KiB，最多 32 个注册类型。拒绝浏览器 Origin，不接受实例 token 调用管理接口。App 为启动实例提供连接信息，开发者不使用内部 API key。

请求：`{"id":"unique-request-id","method":"push","params":{...}}`。
成功：`{"id":"...","result":{...}}`。
失败：`{"id":"...","error":{"code":"invalid_data","message":"字段错误说明"}}`。
通知：`{"method":"response","params":{...}}`。

## 请求

- `register`: `events: [{type, schema, speak, mergeable?}]`，可选 `activity: {name, role, audience, objective}`。Schema 为 Draft 2020-12 对象结构，只支持本地 `$ref`；定义持久化一次并绑定修订摘要。activity 是当前实例生命周期内的活动，不写入宠物长期设定；重连重新注册，省略或 null 则不声明活动。
- `push`: `event_id: UUID, type: string, data: object`。先落库，再确认回执；ID 重复返回原结果，内容冲突报错。同一 ID 不重复响应。
- `context`: 可选 `fields: string[]`。连接绑定唯一宠物/会话，参数不能另选别人的上下文。
- `state`: 当前可响应状态及实际注册的 activity。自主插件活动只对绑定宠物/会话生效，断线、停止后不再进入普通回复。
- `result`: `event_id: UUID`。只查询当前实例下的事件。

## 结果

### App 手动输入（SDK 0.1.6）

`register` 可附加 `inputs: [{type, title, description, submit_label, schema}]`，最多 8 个。
这些是当前插件的表单声明，不是发言事件；Schema 使用与事件一致的本地引用和大小约束。
App 管理接口将用户输入发送为通知 `input`，params 为 `{event_id, type, data}`。
SDK 在单独的输入队列中调用 `Plugin.input(pet, type, data, request_id=event_id)`，
插件自行处理并决定是否 `pet.push()`。普通状态/播放通知不被输入处理器抢占。

SDK 用请求 `input_result` 回传 `{request_id, result}`。宿主只把回执交给 App，
不从表单内容直接建立表达请求。每运行最多 4 条未确认输入，保留最近 64 条回执；
相同输入 ID 重试不会再次投递，内容冲突拒绝，断线/停止后旧回执不能进入新运行。
15 秒未收到回执标记结果未知，不自动重发，也不伪造发言完成。

表单由已验证的管理身份提交，不开放给插件实例 token 或直播平台。
输入成功只表示插件接收；实际播放仍依据原事件的 `completed/interrupted` 通知。

回执 `disposition` 为 `stored_only` / `responding`，包含 `event_id`、`observation_id`、`duplicate` 和 `reason`。

`status`：`stored_only` → 无响应；`pending` → `generating` → `generated` → `completed` / `interrupted`；模型选择不说为 `silent`，异常为 `failed`，已请求停止但设备尚未确认是 `cancelling`。

`generated` 只代表文本生成结束。`completed` / `interrupted` 才包含已确认 `consumed_offset` 和实际播出的 `text`。不能把生成进度当作扬声器输出。断线通知不保证补发，恢复后按 event_id 查询；SDK不排队重放过去的声音。

常用 reason：`silent_event`、`disconnected`、`no_playback_consumer`、`user_busy`、`pet_busy`。新事件在忙时只记录，不会等空闲再播。无事件重放队列。

常用错误：`invalid_request`、`invalid_definition`、`invalid_data`、`unregistered_event`、`id_conflict`、`capacity`、`response_too_large`、`stopped`、`internal_error`。

实例、事件、定义修订、实际消息和结果独立关联。未来新增结构化动作结果或优化版本可以追加可选字段；破坏性改动增加协议主版本，本次实验接口收敛为布尔 speak；auto 请求明确拒绝，不作隐式转换。消费进度仍以设备实际播放为准。


### `reason`（SDK 0.1.3）

请求 `params`：`request_id`（UUID）、`prompt`（最多4000字符）、`schema`（对象JSON Schema，最多16000字符，仅本地引用）、`data`、`fields`。字段可选 personality/memory/history/perception/audiovisual/events。宠物与会话来自连接凭据，不能由调用者切换。

SDK 0.1.7 增加 `input_text`（最多16000字符），与非空 `data` 互斥。提供时改用普通文字分段：只包含 `fields` 明确选择的材料及原始 `input_text`，不自动注入活动职责、对话进度、上次用户消息或完整 Schema；Schema 仍在宿主和 SDK 本地校验。`fields=[]` 表示只发送 `input_text`，不读取快照、历史、感知或截图。未提供 `input_text` 时保持原有 JSON 封装，兼容旧插件。

`audiovisual` 独立读取已采集的影音理解和音频转写，按来源排除 ACS 操作记录；它不修改全局感知设置。推理选择此字段时，在模型支持图片且当前有可用截图的情况下附带截图，与原 `perception` 的截图路径一致。

响应 `result`：`value`（校验后的JSON对象）、`id`（推理记录ID）、`context_revision`。SDK直接返回经过模型类校验的 `value`。推理有45秒上限、16KiB结果上限、每实例一个在途任务、全宿主两个并发任务；不调用TTS、工具或消息生成。请求超过容量返回 `reasoning_busy`，失败返回 `reasoning_failed`，取消返回 `reasoning_cancelled`。

`cancel_reason` 接收同一个 `request_id`，只取消该实例中匹配的任务；旧请求的迟到取消不会影响新请求。SDK协程取消会发送它。推理处理与该连接上的短请求独立，断开连接会回收推理任务。

事件定义的 `include_in_context`（默认 true）控制调度信号是否进入后续事件资料，不改变存储。历史 `context_fields` 字段仅兼容接收，不控制最终发言。宿主共用聊天上下文构建器，固定保留最近最多 50 条真实消息、Static/Dynamic 摘要、角色和开启的感知，并将本次数据置于 `event.data`。SDK 辅助调用可单独选择材料；事件发言封装不能由 SDK 改写。

事件 Schema 的顶层属性可用 `x-fish-context` 标明呈现方式：`metadata` 属性完整保存在事件诊断中，但不进入宠物的发言、事件历史和辅助推理，其 Schema 也不进入发言提示；`identity` 字符串以稳定的不含来源前缀的身份标识呈现。未标注属性保持原样。SDK `context` 使用同一呈现，`result` 仍保留完整诊断数据。这不裁剪普通聊天历史、记忆或感知，也不把外部事件提升为主人指令。

`speak` 默认为 true，只接受 JSON true/false。false 只记录，不调用模型；true 进入当前宠物的先入先出队列，忙时或播放端未连接时返回 `disposition=queued`。宿主收到实际播放完成或空闲状态变化后处理下一条；不新增轮询，不以模型生成完成代替播放完成。插件立即上报独立事件，不必管理宿主忙闲。

SDK 0.1.8 增加 `mergeable: bool`，默认 false。宿主只合并等待队列中相邻、同实例且同定义修订的可合并事件，每次最多 32 条、合并数据最多 16000 字符；超出的事件留待下一次回应。不同类型、定义修订或实例均隔开批次，不设置优先级。合并发言的本轮数据为 `event.items`，每一项分别按原 Schema 呈现并保留身份；单条仍为 `event.data`。每条原事件独立保存状态，合并时共享 `message_id` 与 `response_event_ids`，分别通知实际播放结果。

每只宠物最多等待 256 条发言事件；满时新事件明确返回 `capacity`，不伪装成已接受。相同 ID 的重试仍返回原记录，不重复排队。主动停止实例取消其待处理事件；连接短暂断开时暂停，重新连接后继续。宿主重启把未完成事件标为 interrupted/host_restarted，不伪造播完回执。发言请求不含 speak/mergeable，输出只含 m、a；SDK 数据 Schema 不决定宿主上下文和输出格式。


### Task 增量能力

注册回复的 `capabilities.tasks` 表示宿主支持通用 Agent Task（目标仍需在本体配置）。
`tasks.start` 参数为 `{prompt, request_id, context_key, reference?, context_revision?}`；
固定实例/宠物/会话/目标由宿主解析，拒绝这些身份或端点的调用者覆盖。
`tasks.get` 以 `run_id` 或 `request_id` 查询，`tasks.result` 接受 `run_id,cursor`，
`tasks.cancel` 以原 Run 身份记录停止意图。响应含原 Task/Run、request_id、context_key、状态、
停止标志与结果是否可读；结果分块不超 16 KB。`task_changed` 不携带正文，通知可丢失，查询可恢复。
同一 request_id 与输入摘要绑定；实例 generation 不同不能接管旧任务。

宿主热更新 Task 配置时，通过原连接发送 `capabilities` 通知：
`{"tasks": true, "task_revision": 1}`。SDK 在读取通知时同步更新 `pet.tasks.available`；
注册响应也包含当前值。开关、目标或密钥变动才更新 revision，不重连插件。
关闭后拒绝新请求，已有任务仍可查询、取消，丢失回执的原 request_id 仍可查询或重提。
已派发的远端工作不因关闭开关自动取消；本地等待派发的工作暂停至重新开启。
