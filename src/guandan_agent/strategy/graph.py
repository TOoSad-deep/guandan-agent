"""Acyclic V1/V2 LangGraphs with request authority outside graph state."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from guandan_agent.config import ReviewConfig
from guandan_agent.contracts import (
    DecisionInput, DecisionOutcome, DecisionProposal, ErrorInfo, MainReady, ModelCallResult,
    ReviewResult,
)
from guandan_agent.model.gateway import ModelGateway, build_model_input
from guandan_agent.model.provider import safe_error
from guandan_agent.strategy.risk import detect_risks

if TYPE_CHECKING:
    from guandan_agent.runtime.budget import CallPort


class GraphState(TypedDict, total=False):
    input: DecisionInput
    main_result: ModelCallResult
    proposal: DecisionProposal | None
    error: ErrorInfo | None


class ReviewGraphState(GraphState, total=False):
    trigger_codes: tuple[str, ...]
    review_result: ReviewResult | None


class V1Graph:
    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway

    async def decide(self, input: DecisionInput, calls: CallPort) -> DecisionOutcome:
        """Capabilities live in closures, never in checkpointable graph state."""
        def valid() -> bool:
            view = calls.status(input.context.state.binding)
            return view.active and view.suggestions_open and view.remaining_s > 0

        async def main(state: GraphState):
            model_input = build_model_input(state["input"])
            reply = await self.gateway.choose(model_input, calls)
            return {"main_result": reply, "error": reply.error}

        def route(state: GraphState):
            result = state["main_result"]
            if not valid():
                return {"error": safe_error("REQUEST_STALE", "Request no longer accepts this suggestion.")}
            if result.decision is None or result.call_id is None:
                return {}
            snapshot = state["input"].candidates
            assert snapshot is not None
            registered = calls.register_main(MainReady(
                binding=input.context.state.binding,
                candidate_snapshot_id=snapshot.candidate_snapshot_id,
                call_id=result.call_id, decision=result.decision, trigger_codes=(),
            ))
            if not registered:
                return {"error": safe_error("REQUEST_STALE", "Main suggestion registration was rejected.")}
            return {}

        def route_target(state: GraphState) -> str:
            if state.get("error") is not None and state["error"].code == "REQUEST_STALE":
                return "end"
            return "finalize" if state["main_result"].decision is not None else "fallback"

        def finalize(state: GraphState):
            if not valid():
                return {"error": safe_error("REQUEST_STALE", "Request expired before finalization.")}
            decision = state["main_result"].decision
            assert decision is not None and input.candidates is not None
            return {"proposal": DecisionProposal(
                binding=input.context.state.binding,
                candidate_snapshot_id=input.candidates.candidate_snapshot_id,
                action_id=decision.action_id, source="main", reason=decision.reason,
                cause_code=None, review=None,
            )}

        def fallback(state: GraphState):
            if not valid():
                return {"error": safe_error("REQUEST_STALE", "Request expired before fallback.")}
            error = state.get("error")
            return {"proposal": DecisionProposal(
                binding=input.context.state.binding,
                candidate_snapshot_id=None,
                action_id=input.context.fallback_action_id, source="fallback",
                reason="Model decision unavailable; using the prevalidated fallback.",
                cause_code=None if error is None else error.code, review=None,
            )}

        graph = StateGraph(GraphState)
        graph.add_node("main", main)
        graph.add_node("route", route)
        graph.add_node("finalize", finalize)
        graph.add_node("fallback", fallback)
        graph.add_edge(START, "main")
        graph.add_edge("main", "route")
        graph.add_conditional_edges("route", route_target,
                                    {"finalize": "finalize", "fallback": "fallback", "end": END})
        graph.add_edge("finalize", END)
        graph.add_edge("fallback", END)
        with tracing_context(enabled=False):
            state = await graph.compile().ainvoke(
                {"input": input, "proposal": None, "error": None},
                config={"callbacks": [], "recursion_limit": 8},
            )
        proposal = state.get("proposal")
        return DecisionOutcome(status="proposed" if proposal is not None else "invalidated",
                               proposal=proposal, error=state.get("error"))


class V2Graph:
    """One main call and at most one review OR repair, with no retry edge."""

    def __init__(self, gateway: ModelGateway, review_config: ReviewConfig) -> None:
        self.gateway = gateway
        self.review_config = review_config

    async def decide(self, input: DecisionInput, calls: CallPort) -> DecisionOutcome:
        def valid() -> bool:
            view = calls.status(input.context.state.binding)
            return view.active and view.suggestions_open and view.remaining_s > 0

        def stale():
            return {"error": safe_error("REQUEST_STALE", "Request no longer accepts this suggestion.")}

        async def main(state: ReviewGraphState):
            result = await self.gateway.choose(build_model_input(state["input"]), calls)
            return {"main_result": result, "error": result.error}

        def route(state: ReviewGraphState):
            if not valid():
                return stale()
            result = state["main_result"]
            if result.decision is None:
                error = result.error
                triggers = ((error.code,) if error is not None
                            and error.code in ("MODEL_SCHEMA", "MODEL_ACTION_UNKNOWN") else ())
                return {"trigger_codes": triggers}
            if result.call_id is None:
                return stale()
            triggers = detect_risks(input, result.decision, self.review_config)
            assert input.candidates is not None
            if not calls.register_main(MainReady(
                binding=input.context.state.binding,
                candidate_snapshot_id=input.candidates.candidate_snapshot_id,
                call_id=result.call_id, decision=result.decision, trigger_codes=triggers,
            )):
                return stale()
            return {"trigger_codes": triggers}

        def route_target(state: ReviewGraphState) -> str:
            if state.get("error") is not None and state["error"].code == "REQUEST_STALE":
                return "end"
            if state["main_result"].decision is not None:
                return "review" if state["trigger_codes"] else "finalize"
            return "repair" if state["trigger_codes"] else "fallback"

        async def second(state: ReviewGraphState, kind: str):
            if not valid():
                return stale()
            before = state["main_result"].decision
            model_input = build_model_input(input, role=kind).model_copy(update={
                "main": before, "trigger_codes": state["trigger_codes"],
                "validation_error": state["main_result"].error.code if kind == "repair" else None,
            })
            result = await self.gateway.choose(model_input, calls)
            if not valid():
                return stale()
            if result.calls_started == 0:
                status = "skipped"
            elif result.decision is None:
                status = "failed"
            elif kind == "repair":
                status = "repaired"
            else:
                status = "kept" if result.decision.action_id == before.action_id else "changed"
            error = result.error
            if status in ("skipped", "failed") and error is None:
                error = safe_error("INTERNAL_ERROR", "Second call did not produce a valid authorized result.")
            review = ReviewResult(
                kind=kind, status=status, trigger_codes=state["trigger_codes"], before=before,
                after=result.decision if status in ("kept", "changed", "repaired") else None,
                call_id=result.call_id, error=error,
            )
            return {"review_result": review, "error": error}

        async def review(state: ReviewGraphState):
            return await second(state, "review")

        async def repair(state: ReviewGraphState):
            return await second(state, "repair")

        def second_target(state: ReviewGraphState) -> str:
            if state.get("error") is not None and state["error"].code == "REQUEST_STALE":
                return "end"
            result = state["review_result"]
            if (result.after is not None
                    or result.status == "skipped" and state["main_result"].decision is not None):
                return "finalize"
            return "fallback"

        def finalize(state: ReviewGraphState):
            if not valid():
                return stale()
            review_result = state.get("review_result")
            decision = state["main_result"].decision
            source = "main"
            if review_result is not None and review_result.after is not None:
                decision = review_result.after
                source = review_result.kind
            assert decision is not None and input.candidates is not None
            error = state.get("error")
            return {"proposal": DecisionProposal(
                binding=input.context.state.binding,
                candidate_snapshot_id=input.candidates.candidate_snapshot_id,
                action_id=decision.action_id, source=source, reason=decision.reason,
                cause_code=None if error is None else error.code, review=review_result,
            )}

        def fallback(state: ReviewGraphState):
            if not valid():
                return stale()
            error = state.get("error")
            review_result = state.get("review_result")
            return {"proposal": DecisionProposal(
                binding=input.context.state.binding,
                candidate_snapshot_id=(input.candidates.candidate_snapshot_id
                                       if review_result is not None else None),
                action_id=input.context.fallback_action_id, source="fallback",
                reason="Model decision unavailable; using the prevalidated fallback.",
                cause_code=None if error is None else error.code,
                review=review_result,
            )}

        graph = StateGraph(ReviewGraphState)
        for name, node in (("main", main), ("route", route), ("review", review),
                           ("repair", repair), ("finalize", finalize), ("fallback", fallback)):
            graph.add_node(name, node)
        graph.add_edge(START, "main")
        graph.add_edge("main", "route")
        graph.add_conditional_edges("route", route_target, {
            "review": "review", "repair": "repair", "finalize": "finalize",
            "fallback": "fallback", "end": END,
        })
        for name in ("review", "repair"):
            graph.add_conditional_edges(name, second_target,
                                        {"finalize": "finalize", "fallback": "fallback", "end": END})
        graph.add_edge("finalize", END)
        graph.add_edge("fallback", END)
        with tracing_context(enabled=False):
            state = await graph.compile().ainvoke(
                {"input": input, "proposal": None, "error": None,
                 "trigger_codes": (), "review_result": None},
                config={"callbacks": [], "recursion_limit": 8},
            )
        proposal = state.get("proposal")
        return DecisionOutcome(status="proposed" if proposal is not None else "invalidated",
                               proposal=proposal, error=state.get("error"))
