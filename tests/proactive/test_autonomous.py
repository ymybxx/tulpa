import asyncio
from dataclasses import replace

import pytest

from fish_memory.agent.autonomous import AutonomousRunPolicy, autonomous_run_input
from fish_memory.agent.connector import (
    AgentProviderEvent,
    AgentProviderEventKind,
    AgentRunSnapshot,
    ConnectorRunState,
)
from fish_memory.agent.models import AgentRunStatus
from fish_memory.agent.processor import AgentRunProcessor
from tests.agent.test_processor import FakeStore, work


class Records:
    def __init__(self):
        item = work(status=AgentRunStatus.RUNNING, external_run_id="remote")
        self.work = replace(
            item,
            run=replace(
                item.run, origin="proactive", source_message_id=None, origin_context={}
            ),
        )

    async def request_stop(self, run_id):
        self.work = replace(
            self.work,
            run=replace(self.work.run, origin_context={"stopRequested": True}),
        )
        return self.work

    async def get(self, run_id):
        return self.work

    async def active(self):
        return () if self.work.run.status.is_terminal else (self.work,)

    async def mark_run_unknown(self, run_id, *args):
        self.work = replace(
            self.work, run=replace(self.work.run, status=AgentRunStatus.UNKNOWN)
        )

    async def finish_run(self, run_id, status, *args):
        self.work = replace(self.work, run=replace(self.work.run, status=status))


class Remote:
    def __init__(self, state):
        self.state = state
        self.stops = 0

    async def stop_run(self, *args):
        self.stops += 1

    async def get_run(self, *args):
        return AgentRunSnapshot(self.state)


def test_autonomous_goal_carries_observed_subject_without_a_fake_user_message():
    records = Records()
    item = replace(
        records.work,
        run=replace(
            records.work.run,
            input="核实影片拍摄地",
            origin_context={
                "reference": {
                    "topic": "一部名为测试电影的影片",
                    "progress": "片尾提到蓝色山谷",
                }
            },
        ),
    )
    request = autonomous_run_input(item)
    assert "核实影片拍摄地" in request and "蓝色山谷" in request
    assert item.run.source_message_id is None
    ordinary = replace(item, run=replace(item.run, origin="user"))
    assert autonomous_run_input(ordinary) == "核实影片拍摄地"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "remote_state,expected",
    [
        (ConnectorRunState.RUNNING, AgentRunStatus.UNKNOWN),
        (ConnectorRunState.CANCELLED, AgentRunStatus.CANCELLED),
    ],
)
async def test_external_stop_must_be_confirmed_before_releasing_slot(
    remote_state, expected
):
    records, remote = Records(), Remote(remote_state)
    policy = AutonomousRunPolicy(
        records, records, {"hermes": remote}, timeout_seconds=900
    )
    confirmed = await policy.stop(records.work)
    assert remote.stops == 1
    assert records.work.run.status == expected
    assert confirmed == expected.is_terminal
    assert records.work.run.origin_context["stopRequested"]


@pytest.mark.asyncio
async def test_approval_event_stops_autonomous_run_without_materializing_interaction():
    records, remote = Records(), Remote(ConnectorRunState.CANCELLED)
    policy = AutonomousRunPolicy(
        records, records, {"hermes": remote}, timeout_seconds=900
    )
    ordinary = FakeStore()
    processor = AgentRunProcessor(
        ordinary, {"hermes": remote}, autonomous_policy=policy
    )
    try:
        terminal = await processor._apply_event(
            records.work,
            AgentProviderEvent(
                kind=AgentProviderEventKind.APPROVAL,
                state=ConnectorRunState.WAITING_USER,
                approval_choices=("allow", "deny"),
            ),
        )
        assert terminal and remote.stops == 1
        assert not ordinary.approval_is_open
        assert not any("approval" in name for name, _ in ordinary.audit)
    finally:
        await processor.close()


@pytest.mark.asyncio
async def test_unknown_dispatch_identity_never_fakes_a_successful_stop():
    records, remote = Records(), Remote(ConnectorRunState.CANCELLED)
    records.work = replace(
        records.work,
        run=replace(
            records.work.run, status=AgentRunStatus.DISPATCHING, external_run_id=None
        ),
    )
    policy = AutonomousRunPolicy(
        records, records, {"hermes": remote}, timeout_seconds=900
    )
    assert not await policy.stop(records.work)
    assert records.work.run.origin_context["stopRequested"]
    assert not records.work.run.status.is_terminal
    assert remote.stops == 0


@pytest.mark.asyncio
async def test_recovery_records_stop_intent_without_waiting_for_remote_network():
    records, release = Records(), asyncio.Event()

    class GatedRemote(Remote):
        async def stop_run(self, *args):
            await release.wait()
            await super().stop_run(*args)

    remote = GatedRemote(ConnectorRunState.CANCELLED)
    policy = AutonomousRunPolicy(
        records, records, {"hermes": remote}, timeout_seconds=900
    )
    try:
        await asyncio.wait_for(policy.recover(), 0.1)
        assert records.work.run.origin_context["stopRequested"]
        assert not records.work.run.status.is_terminal
    finally:
        release.set()
        await policy.close()
    assert remote.stops == 1
