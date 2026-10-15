"""Safe release of unused native allocator pages outside real-time paths."""

from __future__ import annotations

import ctypes
import gc
import logging
import sys

logger = logging.getLogger(__name__)


def release_unused_memory(reason: str) -> int:
    """Collect unreachable Python objects and ask macOS malloc to return idle pages."""
    gc.collect()
    if sys.platform != "darwin":
        return 0
    try:
        libc = ctypes.CDLL(None)
        pressure_relief = libc.malloc_zone_pressure_relief
        pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        pressure_relief.restype = ctypes.c_size_t
        reclaimed = int(pressure_relief(None, 0))
    except (AttributeError, OSError, TypeError, ValueError):
        logger.debug("Native allocator pressure relief unavailable", exc_info=True)
        return 0
    logger.info(
        "Released unused native allocator pages: reason=%s bytes=%d",
        reason,
        reclaimed,
    )
    return reclaimed
