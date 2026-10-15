"""Small process entrypoint. No host modules, API keys, or audio libraries."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import signal
import sys

from .client import Pet, PluginError
from .plugin import Plugin


def load(entrypoint: str) -> type[Plugin]:
    module, separator, name = entrypoint.partition(":")
    if not separator:
        raise ValueError("entrypoint must be module:PluginClass")
    plugin = getattr(importlib.import_module(module), name)
    if not isinstance(plugin, type) or not issubclass(plugin, Plugin):
        raise ValueError("entrypoint must inherit fish_plugin_sdk.Plugin")
    return plugin


async def run(plugin: type[Plugin], startup: dict) -> None:
    instance = plugin(startup["config"])
    description = plugin.describe()
    async with Pet(
        startup["url"],
        startup["token"],
        description["events"],
        activity=description.get("activity"),
        inputs=description.get("inputs"),
    ) as pet:
        work = asyncio.create_task(instance.run(pet))
        connection = asyncio.create_task(pet.wait_closed())
        inputs = asyncio.create_task(deliver_inputs(instance, pet))

        async def parent_lifetime() -> None:
            parent = startup.get("parent_pid", os.getppid())
            while os.getppid() == parent:  # noqa: ASYNC110 - parent death has no portable event source
                await asyncio.sleep(2)

        parent = asyncio.create_task(parent_lifetime())
        loop = asyncio.get_running_loop()
        if sys.platform != "win32":
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, work.cancel)
        try:
            done, _ = await asyncio.wait(
                [work, connection, parent, inputs], return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
        finally:
            for task in (work, connection, parent, inputs):
                task.cancel()
            await asyncio.gather(
                work, connection, parent, inputs, return_exceptions=True
            )


async def deliver_inputs(instance: Plugin, pet: Pet) -> None:
    definitions = {value.type: value for value in instance.inputs}
    async for request in pet.input_requests():
        try:
            definition = definitions[request["type"]]
            data = definition.data.model_validate(request["data"])
            result = await instance.input(
                pet, definition.type, data, request_id=request["event_id"]
            )
        except (KeyError, ValueError):
            result = {"disposition": "rejected", "reason": "invalid_input"}
        except Exception:
            logging.getLogger(__name__).exception("Plugin input failed")
            result = {"disposition": "rejected", "reason": "input_failed"}
        try:
            await pet.input_result(request["event_id"], result)
        except (PluginError, TimeoutError):
            logging.getLogger(__name__).warning("Input receipt not delivered")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("entrypoint")
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s"
    )
    plugin = load(args.entrypoint)
    if args.describe:
        print(json.dumps(plugin.describe(), ensure_ascii=False))
        return
    startup = json.loads(sys.stdin.readline())
    try:
        asyncio.run(run(plugin, startup))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except PluginError as error:
        if error.code not in {"stopped", "closed"}:
            raise


if __name__ == "__main__":
    main()
