"""Bounded App-to-plugin inputs. Only the plugin decides which events to push."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from jsonschema import Draft202012Validator

from fish_memory.extensions.models import (
    MAX_FRAME_BYTES,
    ExtensionError,
    Instance,
    PushEvent,
)

INPUT_TIMEOUT_SECONDS = 15


@dataclass
class PendingInput:
    request: PushEvent
    future: asyncio.Future[dict[str, Any]]
    timer: asyncio.TimerHandle | None = None


class PluginInputs:
    def __init__(self) -> None:
        self._records: dict[UUID, OrderedDict[UUID, PendingInput]] = {}

    async def send(
        self, item: Instance, body: dict[str, Any], emit: Any
    ) -> dict[str, Any]:
        if item.stopped or not item.connected:
            raise ExtensionError("disconnected", "请先启动插件，再发送测试输入")
        request = PushEvent.model_validate(body)
        if len(request.model_dump_json().encode()) > MAX_FRAME_BYTES - 256:
            raise ExtensionError("input_too_large", "输入内容过大")
        definition = item.inputs.get(request.type)
        if definition is None:
            raise ExtensionError(
                "unknown_input", "当前插件版本不支持此输入，请更新插件"
            )
        errors = Draft202012Validator(definition.schema_).iter_errors(request.data)
        error = next(iter(errors), None)
        if error is not None:
            raise ExtensionError("invalid_input", error.message)
        records = self._records.setdefault(item.generation, OrderedDict())
        record = records.get(request.event_id)
        if record is not None:
            if record.request != request:
                raise ExtensionError("id_conflict", "同一输入 ID 不能对应不同内容")
        else:
            if sum(not value.future.done() for value in records.values()) >= 4:
                raise ExtensionError("input_busy", "插件正在处理输入，请稍后再试")
            while len(records) >= 64:
                finished = next(
                    key for key, value in records.items() if value.future.done()
                )
                records.pop(finished)
            record = PendingInput(request, asyncio.get_running_loop().create_future())
            records[request.event_id] = record
            record.timer = asyncio.get_running_loop().call_later(
                INPUT_TIMEOUT_SECONDS, self._timeout, record.future
            )
            emit(item, "input", request.model_dump(mode="json"))
        # A cancelled HTTP waiter must not cancel processing or a duplicate waiter.
        return await asyncio.shield(record.future)

    @staticmethod
    def _timeout(future: asyncio.Future[dict[str, Any]]) -> None:
        if not future.done():
            future.set_result({"disposition": "unknown", "reason": "input_timeout"})

    def complete(self, item: Instance, body: dict[str, Any]) -> dict[str, bool]:
        identifier = UUID(body["request_id"])
        result = body.get("result")
        if not isinstance(result, dict):
            raise ExtensionError("invalid_receipt", "Input receipt must be an object")
        record = self._records.get(item.generation, {}).get(identifier)
        if record is None or record.future.done():
            return {"received": False}
        if record.timer is not None:
            record.timer.cancel()
        record.future.set_result(result)
        return {"received": True}

    def disconnect(self, item: Instance) -> None:
        for record in self._records.pop(item.generation, {}).values():
            if record.timer is not None:
                record.timer.cancel()
            if not record.future.done():
                record.future.set_result(
                    {"disposition": "unknown", "reason": "disconnected"}
                )
