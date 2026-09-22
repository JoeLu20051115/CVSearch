import unittest

from qavs.evidence_gap.pdf_controller import (
    ActionOutcome,
    PDFTreeController,
    RouteDirective,
    StateAssessment,
)
from qavs.evidence_gap.pdf_types import (
    ActionName,
    EvidenceGapScores,
    PDFSearchConfig,
    SearchStateRecord,
    TerminationType,
)
from tests.test_pdf_types import full_config


def assessment(
    answer,
    *,
    uncertainty,
    gaps=(0.8, 0.2, 0.1, 0.0),
    support=(0.2, 0.1),
    verifier_independent=True,
    aggregation_available=True,
):
    return StateAssessment(
        answer=answer,
        uncertainty=uncertainty,
        gaps=EvidenceGapScores(*gaps),
        support_avg=support[0],
        support_min=support[1],
        verifier_independent=verifier_independent,
        aggregation_available=aggregation_available,
        model_calls=1,
        processed_pixels=100,
    )


def move(state, action, suffix):
    if action is ActionName.SPLIT:
        path = state.path_keys + (suffix,)
    elif action is ActionName.NEXT:
        path = state.path_keys[:-1] + (suffix,)
    else:
        path = state.path_keys
    context = state.context_keys + ((suffix,) if action is ActionName.EXPAND else ())
    return ActionOutcome.changed(
        path_keys=path,
        focus_keys=(path[-1],),
        context_keys=context,
        visited_keys=tuple(dict.fromkeys(state.visited_keys + (suffix,))),
        observation_keys=state.observation_keys + (f"{suffix}@{action.value}",),
    )


class PDFTreeControllerTest(unittest.TestCase):
    def config(self, **controller_updates):
        value = full_config()
        value["controller"].update(controller_updates)
        return PDFSearchConfig.from_mapping(value)

    def test_starts_at_root_descends_by_split_and_uncertainty_certifies_leaf(self):
        seen_paths = []

        def assess(state):
            seen_paths.append(state.path_keys)
            if state.path_keys == ("root",):
                return assessment("root-answer", uncertainty=0.8)
            return assessment(
                "leaf-answer", uncertainty=0.1,
                gaps=(0.1, 0.1, 0.1, 0.1), support=(0.9, 0.8),
            )

        def feasible(state):
            return {action: action is ActionName.SPLIT for action in ActionName if action is not ActionName.BACKTRACK}

        result = PDFTreeController(self.config()).run(
            root_key="root",
            assess=assess,
            feasible=feasible,
            execute=lambda action, state: move(state, action, "leaf"),
            branch_available=lambda state: False,
        )

        self.assertEqual(seen_paths, [("root",), ("root", "leaf")])
        self.assertEqual(result.termination, TerminationType.CERTIFIED_STOP)
        self.assertEqual(result.answer, "leaf-answer")
        self.assertEqual([step.action for step in result.steps], [ActionName.SPLIT])
        self.assertEqual(result.final_state.path_keys, ("root", "leaf"))

    def test_cumulative_acceptance_stops_before_another_assessment(self):
        seen = []

        result = PDFTreeController(self.config()).run(
            root_key="root",
            initial_state=SearchStateRecord(
                state_id=1,
                focus_keys=("proposal",),
                path_keys=("root", "proposal"),
                context_keys=(),
                visited_keys=("root", "proposal"),
                observation_keys=("root@root", "proposal@base"),
                remaining_steps=2,
                remaining_model_calls=3,
                remaining_pixels=300,
            ),
            assess=lambda state: (
                seen.append(state.state_id),
                assessment("accepted", uncertainty=0.9),
            )[1],
            feasible=lambda state: self.fail("no feasibility check after acceptance"),
            execute=lambda action, state: self.fail("no action after acceptance"),
            branch_available=lambda state: False,
            route_after_assessment=lambda state, current: RouteDirective(
                "accept", "paper_acceptance_gate",
            ),
        )

        self.assertEqual(result.termination, TerminationType.ACCEPTED_STOP)
        self.assertEqual(result.selected_history_state_id, 1)
        self.assertEqual(result.assessment_count, 1)
        self.assertEqual(seen, [1])

    def test_continues_from_a_budget_bounded_nonroot_initial_state(self):
        config = full_config()
        config["budget"].update({
            "max_steps": 2,
            "max_model_calls": 3,
            "max_processed_pixels": 300,
        })
        initial = SearchStateRecord(
            state_id=0,
            focus_keys=("recovered",),
            path_keys=("root", "proposal", "recovered"),
            context_keys=(),
            visited_keys=("root", "proposal", "recovered"),
            observation_keys=("root@root", "proposal@base", "recovered@base"),
            remaining_steps=2,
            remaining_model_calls=3,
            remaining_pixels=300,
        )
        seen = []

        result = PDFTreeController(PDFSearchConfig.from_mapping(config)).run(
            root_key="root",
            initial_state=initial,
            assess=lambda state: (
                seen.append(state),
                assessment("recovered", uncertainty=0.8),
            )[1],
            feasible=lambda state: {
                action: False for action in ActionName
                if action is not ActionName.BACKTRACK
            },
            execute=lambda action, state: self.fail("no action should execute"),
            branch_available=lambda state: False,
        )

        self.assertEqual(seen, [initial])
        self.assertEqual(result.final_state.path_keys, initial.path_keys)
        self.assertEqual(result.final_state.visited_keys, initial.visited_keys)
        self.assertEqual(result.final_state.remaining_steps, 2)
        self.assertEqual(result.final_state.remaining_model_calls, 2)
        self.assertEqual(result.final_state.remaining_pixels, 200)

    def test_gap_argmax_executes_zoom_expand_and_next_with_reassessment(self):
        actions = []
        gap_rows = iter(((0.9, 0.2, 0.1, 0.0), (0.1, 0.2, 0.9, 0.0), (0.1, 0.2, 0.3, 0.9)))

        def assess(state):
            try:
                gaps = next(gap_rows)
            except StopIteration:
                gaps = (0.1, 0.1, 0.1, 0.1)
            return assessment("answer", uncertainty=0.8, gaps=gaps)

        def execute(action, state):
            actions.append(action)
            return move(state, action, f"n{len(actions)}")

        config = full_config()
        config["budget"]["max_steps"] = 3
        result = PDFTreeController(PDFSearchConfig.from_mapping(config)).run(
            root_key="root", assess=assess,
            feasible=lambda state: {action: True for action in ActionName if action is not ActionName.BACKTRACK},
            execute=execute, branch_available=lambda state: False,
        )
        self.assertEqual(actions, [ActionName.ZOOM, ActionName.EXPAND, ActionName.NEXT])
        self.assertEqual(result.termination, TerminationType.FORCED_RETURN)
        self.assertEqual(result.assessment_count, 4)

    def test_noop_is_suppressed_and_next_ranked_feasible_action_runs(self):
        calls = []

        def execute(action, state):
            calls.append(action)
            if action is ActionName.ZOOM:
                return ActionOutcome.no_op("already_zoomed")
            return move(state, action, "child")

        config = full_config()
        config["budget"]["max_steps"] = 1
        result = PDFTreeController(PDFSearchConfig.from_mapping(config)).run(
            root_key="root",
            assess=lambda state: assessment("a", uncertainty=0.9, gaps=(0.9, 0.8, 0.2, 0.1)),
            feasible=lambda state: {
                ActionName.ZOOM: True, ActionName.SPLIT: True,
                ActionName.EXPAND: False, ActionName.NEXT: False,
            },
            execute=execute, branch_available=lambda state: False,
        )
        self.assertEqual(calls, [ActionName.ZOOM, ActionName.SPLIT])
        self.assertEqual([step.status for step in result.steps], ["no_op", "changed"])
        self.assertEqual(result.steps[0].reason, "already_zoomed")

    def test_stalled_uncertainty_backtracks_to_ancestor_then_switches_sibling(self):
        visits = []

        def assess(state):
            visits.append(state.path_keys)
            if state.path_keys == ("root", "good"):
                return assessment(
                    "good", uncertainty=0.1,
                    gaps=(0.1, 0.1, 0.1, 0.1), support=(0.9, 0.8),
                )
            return assessment("weak", uncertainty=0.8, gaps=(0.1, 0.9, 0.1, 0.8))

        descended = False

        def feasible(state):
            nonlocal descended
            if state.path_keys == ("root",):
                action = ActionName.NEXT if descended else ActionName.SPLIT
            else:
                action = ActionName.NEXT
            return {item: item is action for item in ActionName if item is not ActionName.BACKTRACK}

        def execute(action, state):
            nonlocal descended
            if action is ActionName.NEXT:
                return ActionOutcome.changed(
                    path_keys=("root", "good"),
                    focus_keys=("good",),
                    context_keys=(),
                    visited_keys=tuple(dict.fromkeys(state.visited_keys + ("good",))),
                    observation_keys=state.observation_keys + ("good@NEXT",),
                )
            descended = True
            return move(state, action, "bad")

        result = PDFTreeController(self.config(stall_patience=1)).run(
            root_key="root", assess=assess, feasible=feasible, execute=execute,
            branch_available=lambda state: state.path_keys == ("root",),
        )
        self.assertIn(ActionName.BACKTRACK, [step.action for step in result.steps])
        self.assertEqual(result.termination, TerminationType.CERTIFIED_STOP)
        self.assertEqual(result.final_state.path_keys, ("root", "good"))
        backtrack = next(step for step in result.steps if step.action is ActionName.BACKTRACK)
        self.assertLess(backtrack.after.remaining_steps, backtrack.before.remaining_steps)

    def test_backtrack_checks_branch_availability_with_cumulative_visited_keys(self):
        next_nodes = iter(("a", "b"))

        def assess(state):
            support = (0.8, 0.7) if state.path_keys[-1] == "a" else (0.2, 0.1)
            return assessment("answer", uncertainty=0.8, support=support)

        def feasible(state):
            available = (
                state.path_keys[-1] in {"root", "a"}
                and "b" not in state.visited_keys
            )
            return {
                action: available and action is ActionName.NEXT
                for action in ActionName if action is not ActionName.BACKTRACK
            }

        result = PDFTreeController(self.config()).run(
            root_key="root",
            assess=assess,
            feasible=feasible,
            execute=lambda action, state: move(state, action, next(next_nodes)),
            branch_available=lambda state: "b" not in state.visited_keys,
        )

        self.assertNotIn(ActionName.BACKTRACK, [step.action for step in result.steps])
        self.assertEqual(result.final_state.path_keys, ("b",))
        self.assertEqual(result.final_state.visited_keys, ("root", "a", "b"))

    def test_rejected_backtrack_restores_observation_and_explores_unseen_sibling(self):
        assessed = []
        routed = []

        def assess(state):
            name = state.path_keys[-1]
            assessed.append(name)
            return assessment(
                name, uncertainty=0.8, gaps=(0.0, 1.0, 0.0, 0.0),
                support=(0.3, 0.3) if name == "a" else (0.2, 0.2),
            )

        def route(state, value):
            name = state.path_keys[-1]
            routed.append(name)
            return RouteDirective(
                "accept" if name == "c" else "backtrack", "test_evidence",
            )

        def execute(action, state):
            child = "c" if "b" in state.visited_keys else "b"
            return move(state, action, child)

        result = PDFTreeController(self.config()).run(
            root_key="root",
            initial_state=SearchStateRecord(
                state_id=1, focus_keys=("a",), path_keys=("root", "a"),
                context_keys=(), visited_keys=("root", "a"),
                observation_keys=("root@root", "a@base"),
                remaining_steps=8, remaining_model_calls=96,
                remaining_pixels=600000000,
            ),
            assess=assess,
            feasible=lambda state: {
                action: action is ActionName.SPLIT
                and state.path_keys[-1] == "a" and "c" not in state.visited_keys
                for action in ActionName if action is not ActionName.BACKTRACK
            },
            execute=execute,
            branch_available=lambda state: "c" not in state.visited_keys,
            route_after_assessment=route,
        )

        self.assertEqual(assessed, ["a", "b", "c"])
        self.assertEqual(routed, ["a", "b", "c"])
        self.assertEqual(result.termination, TerminationType.ACCEPTED_STOP)
        self.assertEqual(result.assessment_count, 3)
        self.assertEqual(result.final_state.remaining_model_calls, 93)
        self.assertEqual(
            [step.action for step in result.steps],
            [ActionName.SPLIT, ActionName.BACKTRACK, ActionName.SPLIT],
        )
        self.assertEqual(
            getattr(result.steps[1], "restored_assessment_state_id", None), 1,
        )

    def test_backtrack_can_revisit_parent_after_exploring_another_child(self):
        assessed = []

        def assess(state):
            name = state.path_keys[-1]
            assessed.append(name)
            return assessment(
                name, uncertainty=0.8, gaps=(0.0, 1.0, 0.0, 0.0),
                support=(0.3, 0.3) if name == "a" else (0.2, 0.2),
            )

        def execute(action, state):
            child = next(key for key in ("b", "c", "d") if key not in state.visited_keys)
            return move(state, action, child)

        result = PDFTreeController(self.config()).run(
            root_key="root",
            initial_state=SearchStateRecord(
                state_id=1, focus_keys=("a",), path_keys=("root", "a"),
                context_keys=(), visited_keys=("root", "a"),
                observation_keys=("root@root", "a@base"),
                remaining_steps=8, remaining_model_calls=96,
                remaining_pixels=600000000,
            ),
            assess=assess,
            feasible=lambda state: {
                action: action is ActionName.SPLIT
                and state.path_keys[-1] == "a" and "d" not in state.visited_keys
                for action in ActionName if action is not ActionName.BACKTRACK
            },
            execute=execute,
            branch_available=lambda state: "d" not in state.visited_keys,
            route_after_assessment=lambda state, value: RouteDirective(
                "accept" if state.path_keys[-1] == "d" else "backtrack", "test_evidence",
            ),
        )

        self.assertEqual(assessed, ["a", "b", "c", "d"])
        self.assertEqual(result.termination, TerminationType.ACCEPTED_STOP)
        self.assertEqual(result.assessment_count, 4)
        self.assertEqual(result.final_state.remaining_model_calls, 92)
        self.assertEqual(
            [step.action for step in result.steps],
            [ActionName.SPLIT, ActionName.BACKTRACK, ActionName.SPLIT,
             ActionName.BACKTRACK, ActionName.SPLIT],
        )

    def test_budget_forced_return_selects_best_historical_state(self):
        def assess(state):
            if state.path_keys == ("root",):
                return assessment("root-best", uncertainty=0.2, support=(0.8, 0.7))
            return assessment("child-worse", uncertainty=0.9, support=(0.1, 0.0))

        config = full_config()
        config["budget"]["max_steps"] = 1
        result = PDFTreeController(PDFSearchConfig.from_mapping(config)).run(
            root_key="root", assess=assess,
            feasible=lambda state: {
                ActionName.ZOOM: False, ActionName.SPLIT: True,
                ActionName.EXPAND: False, ActionName.NEXT: False,
            },
            execute=lambda action, state: move(state, action, "child"),
            branch_available=lambda state: False,
        )
        self.assertEqual(result.termination, TerminationType.FORCED_RETURN)
        self.assertEqual(result.answer, "root-best")
        self.assertEqual(result.selected_history_state_id, 0)

    def test_verifier_fallback_can_never_certify(self):
        config = full_config()
        config["budget"]["max_steps"] = 1
        result = PDFTreeController(PDFSearchConfig.from_mapping(config)).run(
            root_key="root",
            assess=lambda state: assessment(
                "answer", uncertainty=0.0, gaps=(0.0, 0.0, 0.0, 0.0),
                support=(1.0, 1.0), verifier_independent=False,
            ),
            feasible=lambda state: {action: False for action in ActionName if action is not ActionName.BACKTRACK},
            execute=lambda action, state: self.fail("no action should execute"),
            branch_available=lambda state: False,
        )
        self.assertEqual(result.termination, TerminationType.FORCED_RETURN)

    def test_high_multi_prompt_uncertainty_blocks_otherwise_valid_stop(self):
        result = PDFTreeController(self.config()).run(
            root_key="root",
            assess=lambda state: assessment(
                "unstable", uncertainty=0.9,
                gaps=(0.0, 0.0, 0.0, 0.0), support=(1.0, 1.0),
            ),
            feasible=lambda state: {action: False for action in ActionName if action is not ActionName.BACKTRACK},
            execute=lambda action, state: self.fail("no action should execute"),
            branch_available=lambda state: False,
        )
        self.assertEqual(result.termination, TerminationType.FORCED_RETURN)

    def test_assessment_cost_is_preflighted_before_model_callbacks(self):
        class Assessor:
            def __init__(self):
                self.calls = 0

            def estimate_cost(self, state):
                return (1, 1) if self.calls == 0 else (1000, 1)

            def __call__(self, state):
                self.calls += 1
                return assessment("root", uncertainty=0.9)

        assessor = Assessor()
        result = PDFTreeController(self.config()).run(
            root_key="root", assess=assessor,
            feasible=lambda state: {
                ActionName.ZOOM: True, ActionName.SPLIT: False,
                ActionName.EXPAND: False, ActionName.NEXT: False,
            },
            execute=lambda action, state: move(state, action, "zoomed"),
            branch_available=lambda state: False,
        )
        self.assertEqual(assessor.calls, 1)
        self.assertEqual(result.termination, TerminationType.FORCED_RETURN)


if __name__ == "__main__":
    unittest.main()
