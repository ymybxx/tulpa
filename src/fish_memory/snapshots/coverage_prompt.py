SNAPSHOT_SUPPORT_COVERAGE_CONTRACT = (
    "## Summary 与 support coverage\n"
    "选择一个 sourceFactIndex 或 newSupportIndex，表示最终 Summary 承诺覆盖该 Fact "
    "仍会影响后续回答的核心命题。仅提及相同主题或实体不算覆盖。\n"
    "对象、关系、否定、条件、状态、未完成动作、数量、时间和顺序不得丢失。\n"
    "多个 Facts 可共享一句等价表达，但同主题下无法无损合并的独立命题必须拆成多个 "
    "Create。\n"
    "输出前逐条复核所有被选择或保留的 support；无法保证覆盖时不得把它们合并到同一 "
    "Summary。\n"
)
