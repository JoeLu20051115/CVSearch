import unittest

from cvsearch.evidence_gap.pdf_controller import (
    ActionOutcome,
    PDFTreeController,
    StateAssessment,
)
from cvsearch.evidence_gap.pdf_types import (
    ActionName,
    EvidenceGapScores,
    PDFSearchConfig,
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


if __name__ == "__main__":
    unittest.main()
