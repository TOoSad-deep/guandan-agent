"""Reject contradictory review claims at the immutable DTO boundary."""
import pytest
from pydantic import ValidationError

from guandan_agent.config import RunConfig
from guandan_agent.contracts import ModelDecision, ReviewResult, ErrorInfo


def outcome(**updates):
    values = dict(kind="review", status="changed", trigger_codes=("USE_BOMB",),
                  before=ModelDecision(action_id="L0001", reason="main"),
                  after=ModelDecision(action_id="L0002", reason="review"),
                  call_id="request:m2", error=None)
    return {**values, **updates}


def test_review_outcome_is_frozen_and_owns_copied_trigger_list():
    triggers = ["USE_BOMB"]
    value = ReviewResult(**outcome(trigger_codes=triggers))
    triggers.append("BEAT_PARTNER")
    assert value.trigger_codes == ("USE_BOMB",)
    assert ReviewResult.model_validate_json(value.model_dump_json()) == value
    with pytest.raises(ValidationError):
        value.status = "kept"


@pytest.mark.parametrize("updates", [
    {"status": "kept"}, {"before": None}, {"after": None}, {"call_id": None},
    {"kind": "repair"}, {"status": "repaired"}, {"trigger_codes": ("USE_BOMB", "USE_BOMB")},
    {"error": ErrorInfo(code="MODEL_SCHEMA", component="test", safe_message="Invalid", related_event_id=None)},
    {"status": "skipped"}, {"status": "failed"}, {"extra": True},
])
def test_review_rejects_false_status_or_unattributed_success(updates):
    with pytest.raises(ValidationError):
        ReviewResult(**outcome(**updates))


def test_skipped_ticket_retains_id_and_successful_repair_has_no_valid_main():
    error = ErrorInfo(code="NO_REVIEW_TIME", component="test", safe_message="No time", related_event_id=None)
    value = ReviewResult(**outcome(status="skipped", after=None, error=error))
    assert value.call_id == "request:m2"
    repaired = ReviewResult(**outcome(kind="repair", status="repaired", before=None,
                                     trigger_codes=("MODEL_SCHEMA",)))
    assert repaired.after.action_id == "L0002"


def test_uncalibrated_split_cost_risk_is_explicitly_rejected():
    with pytest.raises(ValidationError, match="SPLIT_COST"):
        RunConfig.model_validate({"review": {"split_cost_threshold": 1.0}})
    assert RunConfig.model_validate({"run": {"policy": "V2"}}).review.enabled
