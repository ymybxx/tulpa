"""Develop against this package, without importing the closed-source host."""

from .client import Pet, PluginError
from .plugin import Activity, Event, Input, Plugin
from .review import ReviewCriterion, ReviewProfile, ReviewReference

__all__ = [
    "Activity",
    "Event",
    "Input",
    "Pet",
    "Plugin",
    "PluginError",
    "ReviewCriterion",
    "ReviewProfile",
    "ReviewReference",
]
__version__ = "0.1.8"
