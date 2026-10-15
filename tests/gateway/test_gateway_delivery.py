from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fish_memory.gateway.delivery import EVENT_TYPE, GatewayDelivery, GatewayEvent
from fish_memory.gateway.routes import router


def fixture(recipient=("fish", "session")):
    events = SimpleNamespace(
        store=SimpleNamespace(receive_host_event=AsyncMock()), changed=Mock()
    )
    target = AsyncMock(return_value=recipient)
    return GatewayDelivery(events, target), events


@pytest.mark.parametrize("speak", [True, False])
async def test_an_event_is_kept_like_any_other_and_wakes_nobody(speak):
    deliver, events = fixture()
    event = GatewayEvent(source="快递柜", title="取件", text="包裹到了", speak=speak)
    identifier = await deliver(event)
    events.store.receive_host_event.assert_awaited_once_with(
        "fish",
        "session",
        identifier,
        EVENT_TYPE,
        {"source": "快递柜", "title": "取件", "text": "包裹到了"},
        wakes=False,
        reports_work=speak,
    )
    events.changed.assert_called_once_with("fish", "session")


async def test_an_unnamed_sender_is_called_an_outside_program():
    deliver, events = fixture()
    await deliver(GatewayEvent(text="今晚有雨", speak=False))
    data = events.store.receive_host_event.call_args.args[4]
    assert data == {"source": "外部程序", "text": "今晚有雨"}


async def test_the_senders_id_makes_a_resend_the_same_event():
    first = GatewayDelivery.event_id(GatewayEvent(id="42", source="ci", text="a"))
    again = GatewayDelivery.event_id(GatewayEvent(id="42", source="ci", text="b"))
    other = GatewayDelivery.event_id(GatewayEvent(id="42", source="nas", text="a"))
    assert first == again != other
    assert GatewayDelivery.event_id(GatewayEvent(text="a")) != GatewayDelivery.event_id(
        GatewayEvent(text="a")
    )


async def test_no_character_means_not_accepted():
    deliver, events = fixture(recipient=None)
    assert await deliver(GatewayEvent(text="hi")) is None
    events.store.receive_host_event.assert_not_called()


def test_event_fields_are_bounded():
    with pytest.raises(ValidationError):
        GatewayEvent(text="")
    with pytest.raises(ValidationError):
        GatewayEvent(text="x" * 8001)
    with pytest.raises(ValidationError):
        GatewayEvent(text="x", command="rm")


def test_route_reports_accepted_unready_and_no_character():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    events = "/internal/gateway/events"
    assert client.post(events, json={"text": "hi"}).status_code == 503
    identifier = uuid4()
    app.state.gateway_delivery = AsyncMock(return_value=identifier)
    response = client.post(events, json={"text": "hi"})
    assert response.status_code == 202
    assert response.json() == {"accepted": True, "id": str(identifier)}
    app.state.gateway_delivery = AsyncMock(return_value=None)
    assert client.post(events, json={"text": "hi"}).status_code == 409
    assert client.post(events, json={}).status_code == 422
