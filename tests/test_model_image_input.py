import pytest

from fish_memory.app import image_input_capability


class Model:
    def __init__(self, capability, probed="supported"):
        self.image_input_capability = capability
        self.probed = probed
        self.probes = 0

    async def probe_image_input_capability(self):
        self.probes += 1
        if isinstance(self.probed, Exception):
            raise self.probed
        return self.probed


@pytest.mark.asyncio
async def test_a_known_answer_is_not_asked_again():
    model = Model("unsupported")
    assert await image_input_capability(model) == "unsupported"
    assert model.probes == 0


@pytest.mark.asyncio
async def test_an_unknown_model_is_sent_the_probe_image():
    model = Model("unknown", probed="supported")
    assert await image_input_capability(model) == "supported"
    assert model.probes == 1


@pytest.mark.asyncio
async def test_a_failed_probe_or_missing_model_reads_as_unknown():
    failing = Model("unknown", probed=RuntimeError())
    assert await image_input_capability(failing) == "unknown"
    assert await image_input_capability(None) == "unknown"
