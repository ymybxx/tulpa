import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

from fish_memory.answer.context_loader import LoadedAnswerContext
from fish_memory.answer.output_sink import PromotableAnswerSink
from fish_memory.experimental.config import ExperimentalConfiguration
from fish_memory.experimental.jev import ACTIONS, JevClient
from fish_memory.experimental.prepared import ExperimentalPreparedAnswer
from fish_memory.history.models import HistoryUnit
from fish_memory.performance.tracker import MessagePerformanceTracker
from tests.test_experimental_companion import runtime, settle_work, setup


def response(choice, research):
    return {
        "answers": {
            "next_action": {"type": "choice", "choice": choice},
            "should_research": research,
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ACTIONS)
@pytest.mark.parametrize("probability", [0, 0.49, 0.5, 1])
async def test_research_question_shares_request_and_resolves_one_action(
    choice, probability
):
    requests = []

    async def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, json=response(choice, {"type": "noul", "noul": probability})
        )

    client = JevClient(client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    config = ExperimentalConfiguration(enabled=True, api_key=SecretStr("test-only"))
    state = "History:\nu: 你先去学习一下\na: 我去研究。"
    try:
        action = await client.decide(config, state)
        assert action == ("t" if probability >= 0.5 else choice)
        assert len(requests) == 1
        body = requests[0]
        assert body["state"] == state
        assert list(body["questions"]) == ["next_action", "should_research"]
        assert list(body["questions"]["next_action"]["criteria"]) == list(ACTIONS)
        assert body["questions"]["should_research"]["type"] == "noul"
        assert set(body["questions"]["should_research"]["criteria"]) == {
            "true",
            "false",
        }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_unavailable_agent_never_adds_or_selects_research():
    requests = []

    async def handle(request):
        requests.append(json.loads(request.content))
        # An unsolicited extra answer cannot enable an unavailable capability.
        return httpx.Response(200, json=response("s", {"type": "noul", "noul": 1}))

    client = JevClient(client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    config = ExperimentalConfiguration(enabled=True, api_key=SecretStr("test-only"))
    try:
        assert await client.decide(config, "研究这件事", ("s", "wait")) == "s"
        assert list(requests[0]["questions"]) == ["next_action"]
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "research",
    [
        None,
        {},
        {"type": "choice", "noul": 1},
        {"noul": 1},
        *[
            {"type": "noul", "noul": value}
            for value in (
                True,
                False,
                -0.1,
                1.1,
                "0.9",
                None,
                float("nan"),
                float("inf"),
            )
        ],
    ],
)
async def test_invalid_research_answer_cannot_start_work(research):
    async def handle(request):
        return httpx.Response(200, content=json.dumps(response("s", research)))

    client = JevClient(client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    config = ExperimentalConfiguration(enabled=True, api_key=SecretStr("test-only"))
    try:
        with pytest.raises(ValueError, match="Invalid research probability"):
            await client.decide(config, "研究这件事")
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["s", "t", "wait"])
@pytest.mark.parametrize("entry", ["user", "event", "timer"])
async def test_research_true_uses_existing_t_flow_once(choice, entry):
    requests = []

    async def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response(choice, {"type": "noul", "noul": 0.8}))

    f = setup()
    f.service.judge = JevClient(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    )
    f.models.query = "研究脱口秀的结构与练习方法"
    f.service.tasks = SimpleNamespace(
        available=AsyncMock(return_value=True),
        execute=AsyncMock(return_value=SimpleNamespace(text="已整理研究结果。")),
    )
    context = LoadedAnswerContext(
        [
            {"role": "system", "content": "你正在陪伴用户。"},
            {"role": "user", "content": "History:\nu: 你先去研究下脱口秀怎么说"},
        ],
        (),
    )
    owner = prepared = None
    try:
        if entry == "user":
            prepared = ExperimentalPreparedAnswer(
                f.service,
                uuid4(),
                context,
                PromotableAnswerSink(),
                MessagePerformanceTracker(),
            )
            selected = await prepared.initial_ready()
            assert selected.name == "t"
            f.service.tasks.execute.assert_not_awaited()
            public = PromotableAnswerSink()
            await prepared.release_to_public(
                public,
                source_message_id=uuid4(),
                answer_message_id=uuid4(),
                runtime=None,
                session_id=uuid4(),
            )
            assert public.snapshot().content == "让我回想一下。"
        else:
            owner, contexts, delivery = runtime(f)
            contexts.loaded = context
            if entry == "event":
                unit = HistoryUnit(
                    uuid4(),
                    "event",
                    "直播观众要求先学习脱口秀。",
                    f.clock.now,
                    f.clock.now,
                )
                owner.observe(*owner.binding, unit, received_at=f.clock.now)
            else:
                f.clock.now += timedelta(seconds=6)
                await owner.tick()
            assert owner._decision is not None
            await owner._decision
            assert f.models.channels == ["t"]
            assert len(delivery.streams) == 1
            assert delivery.streams[0].text == "让我回想一下。"

        await settle_work(f)
        assert len(requests) == 1
        assert list(requests[0]["questions"]) == ["next_action", "should_research"]
        f.service.tasks.execute.assert_awaited_once()
        intent, _ = f.service.tasks.execute.call_args.args
        assert intent.purpose == f.models.query
        assert [event.observation_type for _, event in f.observations.events] == [
            "experimental.started",
            "experimental.result",
        ]
        assert "研究结果出来了" in f.observations.events[-1][1].payload["context_text"]
    finally:
        if prepared is not None:
            await prepared.cancel()
        if owner is not None:
            await owner.close()
        await f.service.close()
