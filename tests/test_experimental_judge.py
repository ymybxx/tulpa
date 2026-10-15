import json

import pytest
from pydantic import SecretStr

from fish_memory.experimental.config import ExperimentalConfiguration
from fish_memory.experimental.jev import ACTIONS, JevClient
from fish_memory.experimental.judge import CompanionJudge, ReasoningJudge
from fish_memory.llm.types import ChatStreamChunk


class Reasoning:
    def __init__(self, *answers: dict):
        self.answers = list(answers)
        self.calls: list[dict] = []

    async def stream_json(self, messages, **options):
        self.calls.append({"messages": messages, **options})
        text = json.dumps({"answers": self.answers.pop(0)})
        yield ChatStreamChunk(content=text[:5])
        yield ChatStreamChunk(content=text[5:], finish_reason="stop")


WITHOUT_JEV = ExperimentalConfiguration(enabled=True)


def test_companion_runs_without_a_jev_key():
    assert WITHOUT_JEV.enabled and WITHOUT_JEV.api_key is None


@pytest.mark.asyncio
async def test_reasoning_model_answers_the_same_questions_as_jev():
    reasoning = Reasoning({"next_action": "w", "should_research": 0.2})
    action = await ReasoningJudge(reasoning).decide(
        WITHOUT_JEV, "History:\nu: 这是谁演的"
    )
    assert action == "w"
    call = reasoning.calls[0]
    assert "History:\nu: 这是谁演的" in call["messages"][1]["content"]
    answers = call["response_schema"]["properties"]["answers"]
    assert answers["properties"]["next_action"]["enum"] == list(ACTIONS)
    assert answers["properties"]["should_research"] == {"type": "number"}
    assert answers["required"] == ["next_action", "should_research"]


@pytest.mark.asyncio
async def test_reasoning_answers_are_validated_like_jev():
    research = ReasoningJudge(Reasoning({"next_action": "s", "should_research": 0.8}))
    assert await research.decide(WITHOUT_JEV, "state") == "t"
    unavailable = ReasoningJudge(Reasoning({"next_action": "t"}))
    with pytest.raises(ValueError, match="unavailable action"):
        await unavailable.decide(WITHOUT_JEV, "state", ("s", "wait"))
    out_of_range = ReasoningJudge(Reasoning({"new_observation": 3}))
    with pytest.raises(ValueError, match="observation probability"):
        await out_of_range.admit(WITHOUT_JEV, {"audiovisualSummary": "x"})


@pytest.mark.asyncio
async def test_thought_and_observation_use_the_reasoning_model():
    reasoning = Reasoning({"thought": "2"}, {"new_observation": 0.7})
    judge = ReasoningJudge(reasoning)
    assert await judge.select_thought(WITHOUT_JEV, "state", ("甲", "乙")) == 1
    assert await judge.admit(WITHOUT_JEV, {"audiovisualSummary": "新剧情"})
    assert "新剧情" in reasoning.calls[1]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_jev_is_used_whenever_its_key_is_configured():
    class Jev(JevClient):
        def __init__(self):
            self.calls = 0

        async def evaluate(self, configuration, state, questions, *, kind):
            self.calls += 1
            assert kind == "experimental.jev.decision"
            return {"answers": {"next_action": {"type": "choice", "choice": "s"}}}

        async def close(self):
            pass

    jev, reasoning = Jev(), Reasoning({"next_action": "wait"})
    judge = CompanionJudge(jev, ReasoningJudge(reasoning))
    keyed = WITHOUT_JEV.model_copy(update={"api_key": SecretStr("jev-key")})
    blank = WITHOUT_JEV.model_copy(update={"api_key": SecretStr("  ")})
    assert await judge.decide(keyed, "state", ("s", "wait")) == "s"
    assert await judge.decide(blank, "state", ("s", "wait")) == "wait"
    assert jev.calls == 1 and len(reasoning.calls) == 1
