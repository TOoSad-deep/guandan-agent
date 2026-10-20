"""Rule shortcuts and bounded V1/V2 graphs; Session retains action authority."""

from collections.abc import Callable

from guandan_agent.config import ReviewConfig
from guandan_agent.contracts import (
    DecisionInput,
    DecisionOutcome,
    DecisionProposal,
    ErrorInfo,
    LegalActionSet,
    VisibleState,
)
from guandan_agent.hand import CHEAP_RULE_VERSION, cheap_choice
from guandan_agent.model.gateway import ModelGateway
from guandan_agent.strategy.graph import V1Graph, V2Graph


Chooser = Callable[[VisibleState, LegalActionSet], str]


def _error(code: str, message: str) -> ErrorInfo:
    return ErrorInfo(code=code, component="strategy", safe_message=message, related_event_id=None)


class DecisionPolicy:
    """A proposal producer, with no platform write or session mutation access."""

    rule_version = CHEAP_RULE_VERSION

    def __init__(self, chooser: Chooser = cheap_choice, *, gateway: ModelGateway | None = None,
                 review_config: ReviewConfig | None = None) -> None:
        self._chooser = chooser
        self._graph = None if gateway is None else V1Graph(gateway)
        self._review_graph = (None if gateway is None else
                              V2Graph(gateway, review_config or ReviewConfig()))

    async def decide(self, input: DecisionInput, calls: object | None = None) -> DecisionOutcome:
        context = input.context
        if context.state.binding != context.legal.binding:
            return DecisionOutcome(
                status="invalidated", proposal=None,
                error=_error("STATE_CONFLICT", "State and legal actions belong to different requests."),
            )
        legal_ids = {action.action_id for action in context.legal.actions}
        if context.fallback_action_id not in legal_ids:
            return DecisionOutcome(
                status="unavailable", proposal=None,
                error=_error("INVALID_REQUEST", "The prevalidated fallback is not in the legal set."),
            )
        cause = None
        if len(context.legal.actions) == 1:
            action_id = context.legal.actions[0].action_id
            source = "unique"
            reason = "Only one distinct action in the complete legal set."
        elif context.policy in ("V1", "V2") and self._graph is None:
            return DecisionOutcome(
                status="unavailable", proposal=None,
                error=_error("POLICY_UNAVAILABLE", "The requested model policy is not configured in this batch."),
            )
        elif context.policy in ("V1", "V2"):
            if calls is None:
                return DecisionOutcome(status="unavailable", proposal=None,
                                       error=_error("POLICY_UNAVAILABLE", "A model call port is required."))
            if input.candidates is None or context.state.facts_status != "trusted":
                cause = _error("TOOL_INVALID_RESULT", "Trusted state and registered candidates are required.")
                action_id, source = context.fallback_action_id, "fallback"
                reason = "Model input unavailable; using the prevalidated fallback."
            elif input.candidates.binding != context.state.binding:
                return DecisionOutcome(status="invalidated", proposal=None,
                                       error=_error("REQUEST_STALE", "Candidates belong to a different request."))
            else:
                graph = self._review_graph if context.policy == "V2" else self._graph
                return await graph.decide(input, calls)
        else:
            try:
                if input.candidates is not None:
                    if input.candidates.binding != context.state.binding:
                        return DecisionOutcome(status="invalidated", proposal=None,
                                               error=_error("REQUEST_STALE", "Candidates belong to a different request."))
                    action_id = input.candidates.rule_action_id
                else:
                    action_id = self._chooser(context.state, context.legal)
                if not isinstance(action_id, str) or action_id not in legal_ids:
                    cause = _error("TOOL_INVALID_RESULT", "Rule selector returned an invalid action ID.")
            except Exception:
                cause = _error("INTERNAL_ERROR", "Rule selector failed; using the validated fallback.")
            if cause is None:
                source = "rule"
                reason = ("Prepared V0 candidate rule." if input.candidates is not None
                          else f"Deterministic temporary V0 rule ({self.rule_version}).")
            else:
                action_id = context.fallback_action_id
                source = "fallback"
                reason = "Rule failure; using the request's prevalidated fallback."
        return DecisionOutcome(
            status="proposed",
            proposal=DecisionProposal(
                binding=context.state.binding,
                candidate_snapshot_id=None,
                action_id=action_id,
                source=source,
                reason=reason,
                cause_code=None if cause is None else cause.code,
                review=None,
            ),
            error=cause,
        )
