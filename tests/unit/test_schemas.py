import pytest
from pydantic import ValidationError

from packages.core.models.migrations import migrate
from packages.core.models.schemas import Claim, ClaimStatus, Episode, Event, EventType


def test_versioned_and_strict():
    ep = Episode(goal="x", domain_pack="demo", agent_id="mock")
    assert ep.schema_version == "1.0"
    with pytest.raises(ValidationError):
        Episode(goal="x", domain_pack="demo", agent_id="a", schema_version="2")
    with pytest.raises(ValueError):
        migrate({"schema_version": "99"})


def test_event_and_claim():
    event = Event(episode_id="e", branch_id="b", event_type=EventType.AGENT_ACTION)
    assert Event.model_validate_json(event.model_dump_json()) == event
    assert Claim(episode_id="e", text="x", created_by_event_id="x").status == ClaimStatus.UNVERIFIED
