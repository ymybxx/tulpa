import json
from collections.abc import Sequence

from fish_memory.answer.context import ChatMessage

RQUERY_REPAIR_SYSTEM_PROMPT = """你负责修复一个长期记忆检索 Query。

原始 Query 中的实体不能在修复后的 Query 中丢失。仅修复缺失实体，保留生成 Query 的
其余检索意图；不要添加原始 Query 没有的实体或条件。只返回 JSON 对象
{"q":"修复后的非空 Query"}，不要输出解释、Markdown 或其他字段。"""


def build_rquery_repair_messages(
    original_query: str,
    generated_query: str,
    missing_aliases: Sequence[str],
    error: str,
) -> list[ChatMessage]:
    payload = {
        "originalQuery": original_query,
        "generatedQuery": generated_query,
        "missingAliases": list(missing_aliases),
        "error": error,
    }
    return [
        {"role": "system", "content": RQUERY_REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
