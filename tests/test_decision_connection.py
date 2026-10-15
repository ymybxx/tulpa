from fish_memory.experimental.config import JEV_ENDPOINT
from fish_memory.experimental.decision import decision_connection

BAILIAN = {
    "id": "bailian",
    "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}
DOUBAO = {"id": "doubao", "baseUrl": "https://ark.cn-beijing.volces.com/api/v3"}
VOICE = {"id": "voice", "provider": "dashscope"}


def configuration(**companion):
    return {
        "experimentalCompanion": {"model": "jev-1.13.0", **companion},
        "models": {"text": [DOUBAO, BAILIAN], "voice": [VOICE]},
    }


def test_bailian_uses_the_workspace_address_and_the_saved_bailian_key():
    connection = decision_connection(
        configuration(bailianWorkspaceId="ws-example123"),
        {"modelApiKeys": {"doubao": "d", "bailian": "b"}, "jevApiKey": "j"},
    )
    assert connection is not None
    assert connection.provider == "bailian"
    assert connection.endpoint == (
        "https://ws-example123.cn-beijing.maas.aliyuncs.com"
        "/compatible-mode/v1/systemone"
    )
    assert connection.model == "decision-model-preview"
    assert connection.api_key == "b"


def test_the_voice_key_serves_when_no_bailian_text_model_has_one():
    connection = decision_connection(
        configuration(bailianWorkspaceId="llm-abc123"), {"modelApiKeys": {"voice": "v"}}
    )
    assert connection is not None and connection.api_key == "v"


def test_without_a_choice_only_jev_ready_keeps_jev():
    connection = decision_connection(configuration(), {"jevApiKey": "j"})
    assert connection is not None
    assert (connection.provider, connection.endpoint, connection.api_key) == (
        "jev",
        JEV_ENDPOINT,
        "j",
    )


def test_a_chosen_model_that_cannot_work_connects_nothing():
    assert (
        decision_connection(
            configuration(decisionProvider="bailian"),
            {"modelApiKeys": {"bailian": "b"}},
        )
        is None
    )
    assert (
        decision_connection(
            configuration(decisionProvider="jev", bailianWorkspaceId="ws-1234"),
            {"modelApiKeys": {"bailian": "b"}},
        )
        is None
    )
    assert (
        decision_connection(
            configuration(bailianWorkspaceId="bad id!"),
            {"modelApiKeys": {"bailian": "b"}},
        )
        is None
    )


async def test_more_than_sixteen_questions_go_in_parallel_calls_to_the_endpoint():
    import json

    import httpx
    from pydantic import SecretStr

    from fish_memory.experimental.config import ExperimentalConfiguration
    from fish_memory.experimental.jev import JevClient

    seen = []

    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((str(request.url), body["model"], len(body["questions"])))
        return httpx.Response(
            200,
            json={
                "answers": {
                    name: {"type": "noul", "noul": 0.5} for name in body["questions"]
                }
            },
        )

    client = JevClient(client=httpx.AsyncClient(transport=httpx.MockTransport(answer)))
    endpoint = (
        "https://ws-1234.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/systemone"
    )
    questions = {f"q{index}": {"type": "noul"} for index in range(20)}
    result = await client.evaluate(
        ExperimentalConfiguration(
            api_key=SecretStr("b"), model="decision-model-preview", endpoint=endpoint
        ),
        "state",
        questions,
        kind="test",
    )
    await client.close()
    assert sorted(result["answers"]) == sorted(questions)
    assert sorted(seen) == [
        (endpoint, "decision-model-preview", 4),
        (endpoint, "decision-model-preview", 16),
    ]
