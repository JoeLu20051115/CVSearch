from models.modeling_llava import Model, ModelLocal, ModelGlobalLocal
from models.modeling_llava import BOX_COLOR as BOX_COLOR_LLAVA
from models.modeling_internvl import BOX_COLOR as BOX_COLOR_INTERNVL
from models.modeling_qwenvl import BOX_COLOR as BOX_COLOR_QWENVL
from models.tree import ImageTree, Node, NodeState, AdaptiveImageTree, NodeA
from models.utils import include_pronouns, load_json_or_jsonl, extract_visual_objects, normalize_target_text
from models.modeling_sam3 import ConstrainedTreeBuilder
from typing import Union, Callable, List, Tuple
from PIL import Image
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import json
import os
import numpy as np
import torch


@dataclass(frozen=True)
class _FrozenSearchCandidate:
    """Read-only candidate metadata for callbacks; never the live search node."""

    canonical_key: str
    bbox_original: tuple
    depth: int
    render_level: int
    tree_scope: str
    crop_origin: tuple
    source_image_key: str


_ALLOWED_SOURCE_IMAGE_MODES = frozenset((
    "1", "L", "LA", "La", "P", "PA", "I", "I;16", "I;16B", "I;16L", "I;16N",
    "F", "RGB", "RGBA", "RGBa", "RGBX", "CMYK", "YCbCr", "LAB", "HSV",
))

def _make_rank_context(method_trace, outer_question, visual_cue, tree_scope, crop_origin):
    if not isinstance(outer_question, str) or not outer_question.strip():
        raise ValueError("outer question must be a nonempty string")
    query_plan = getattr(method_trace, 'query_plan', None)
    planned_main = getattr(query_plan, 'main_query', None)
    main_query = planned_main if isinstance(planned_main, str) and planned_main.strip() else outer_question
    augmented_queries = getattr(query_plan, 'augmented_queries', None)
    if not isinstance(augmented_queries, (list, tuple)) or not augmented_queries:
        augmented_queries = [visual_cue]
    return {
        'main_query': main_query,
        'augmented_queries': augmented_queries,
        'method_trace': method_trace,
        'tree_scope': tree_scope,
        'crop_origin': crop_origin,
    }


def _json_state_number(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"search state {name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"search state {name} must be numeric") from error
    if not np.isfinite(number):
        raise ValueError(f"search state {name} must be finite")
    return int(number) if number.is_integer() else number


def _source_image_identity(image_pil):
    """A path-free identity for observations rendered from one original image."""
    return {
        "mode": str(image_pil.mode),
        "size": [int(image_pil.width), int(image_pil.height)],
        "pixel_sha256": hashlib.sha256(image_pil.tobytes()).hexdigest(),
    }


def _canonical_source_image_identity(source_image_identity):
    if not isinstance(source_image_identity, Mapping):
        raise ValueError("search state source_image_identity must be a mapping")
    required = {"mode", "size", "pixel_sha256"}
    if set(source_image_identity) != required:
        raise ValueError("search state source_image_identity has invalid keys")
    mode = source_image_identity["mode"]
    if mode not in _ALLOWED_SOURCE_IMAGE_MODES:
        raise ValueError("search state source_image_identity mode must be a known PIL mode")
    size = source_image_identity["size"]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("search state source_image_identity size must contain two integers")
    canonical_size = []
    for dimension in size:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, (int, np.integer)):
            raise ValueError("search state source_image_identity size must contain two integers")
        dimension = int(dimension)
        if dimension <= 0:
            raise ValueError("search state source_image_identity size must be positive")
        canonical_size.append(dimension)
    digest = source_image_identity["pixel_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("search state source_image_identity pixel_sha256 must be lowercase hex")
    return {"mode": mode, "size": canonical_size, "pixel_sha256": digest}


def _make_search_state_context(tree_scope, crop_origin, source_image_identity, search_call_ordinal):
    if tree_scope not in ("main", "cropped"):
        raise ValueError("search state tree_scope must be main or cropped")
    try:
        origin_x, origin_y = crop_origin
    except (TypeError, ValueError) as error:
        raise ValueError("search state crop_origin must have two values") from error
    ordinal = _json_state_number(search_call_ordinal, "search_call_ordinal")
    if ordinal < 1:
        raise ValueError("search state search_call_ordinal must be positive")
    identity = _canonical_source_image_identity(source_image_identity)
    return {
        "tree_scope": tree_scope,
        "crop_origin": [
            _json_state_number(origin_x, "crop_origin"),
            _json_state_number(origin_y, "crop_origin"),
        ],
        "source_image_identity": identity,
        "search_call_ordinal": ordinal,
    }


def _state_context_or_default(search_state_context, image_pil):
    if search_state_context is None:
        return _make_search_state_context(
            "main", (0, 0), _source_image_identity(image_pil), 1,
        )
    if not isinstance(search_state_context, Mapping):
        raise ValueError("search_state_context must be a mapping")
    required = {
        "tree_scope", "crop_origin", "source_image_identity", "search_call_ordinal",
    }
    if set(search_state_context) != required:
        raise ValueError("search_state_context has invalid keys")
    return _make_search_state_context(
        search_state_context["tree_scope"],
        search_state_context["crop_origin"],
        search_state_context["source_image_identity"],
        search_state_context["search_call_ordinal"],
    )


def _state_bbox_original(node, crop_origin):
    try:
        x, y, width, height = node.state.bbox
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("search state node bbox must have four values") from error
    if width is None or height is None:
        raise ValueError("search state node bbox must have four values")
    origin_x, origin_y = crop_origin
    return [
        _json_state_number(x, "bbox") + origin_x,
        _json_state_number(y, "bbox") + origin_y,
        _json_state_number(width, "bbox"),
        _json_state_number(height, "bbox"),
    ]


def _state_node_depth(node):
    return _json_state_number(getattr(node, "depth", 0), "depth")


def _state_node_key(node, crop_origin):
    payload = {
        "bbox": _state_bbox_original(node, crop_origin),
        "depth": _state_node_depth(node),
        "render_level": 0,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _state_optional_number(node, attribute):
    value = getattr(node, attribute, None)
    if value is None:
        return None
    return _json_state_number(value, attribute)


def _state_node_snapshot(node, crop_origin, stage_rank=None):
    parent = getattr(node, "parent", None)
    children = getattr(node, "children", ())
    if children is None:
        children = ()
    try:
        child_keys = [_state_node_key(child, crop_origin) for child in children]
    except TypeError as error:
        raise ValueError("search state node children must be iterable") from error
    source = getattr(node, "search_source", None)
    if not isinstance(source, str):
        source = None
    return {
        "canonical_key": _state_node_key(node, crop_origin),
        "bbox_original": _state_bbox_original(node, crop_origin),
        "parent_key": None if parent is None else _state_node_key(parent, crop_origin),
        "child_keys": child_keys,
        "depth": _state_node_depth(node),
        "render_level": 0,
        "source": source,
        "stage_rank": stage_rank,
        "prior_prob": _state_optional_number(node, "prior_prob"),
        "fast_confidence": _state_optional_number(node, "fast_confidence"),
        "posterior_score": _state_optional_number(node, "posterior_score"),
        "is_evaluated": bool(getattr(node, "is_evaluated", False)),
        "answering_confidence": _state_optional_number(node, "answering_confidence"),
    }


def _frozen_search_candidate(node, state_context):
    snapshot = _state_node_snapshot(node, state_context["crop_origin"])
    return _FrozenSearchCandidate(
        canonical_key=snapshot["canonical_key"],
        bbox_original=tuple(snapshot["bbox_original"]),
        depth=snapshot["depth"],
        render_level=snapshot["render_level"],
        tree_scope=state_context["tree_scope"],
        crop_origin=tuple(state_context["crop_origin"]),
        source_image_key=json.dumps(
            state_context["source_image_identity"], sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ),
    )


def _emit_p0_selected(search_state_sink, image_pil, searched_nodes):
    if search_state_sink is None:
        return
    context = {
        "tree_scope": "main",
        "crop_origin": [0, 0],
        "source_image_identity": _source_image_identity(image_pil),
        # P0 is not a semantic-search invocation.  Zero keeps it distinct
        # from the positive, pre-invocation semantic call ordinals.
        "search_call_ordinal": 0,
    }
    context["source_image_identity"] = _canonical_source_image_identity(
        context["source_image_identity"]
    )
    candidates = [
        _state_node_snapshot(node, context["crop_origin"], stage_rank=index)
        for index, node in enumerate(searched_nodes)
    ]
    keys = [candidate["canonical_key"] for candidate in candidates]
    snapshot = {
        "schema_version": 1,
        "event": "p0_selected",
        **context,
        "visual_cue": None,
        "stage": "P0",
        "depth": 0,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "ordered_keys": keys,
        "popped_keys": [],
        "selected_keys": keys,
        "remaining_keys": [],
    }
    json.dumps(snapshot, allow_nan=False)
    live_refs = MappingProxyType({
        "selected_nodes": tuple(_frozen_search_candidate(node, context) for node in searched_nodes),
    })
    search_state_sink(live_refs, deepcopy(snapshot))

def _observe_answer(answer_observer, annotation, searched_nodes, raw_answer):
    if answer_observer is not None:
        observation_name = 'quick' if annotation.get('search_mode') == 0 else 'search'
        answer_observer(observation_name, list(searched_nodes), deepcopy(raw_answer))
    return raw_answer

def get_cvsearch_response(
        sam_model,
        zoom_model: Model,
        nlp_model,
        annotation,
        ic_examples,
        decomposed_question_template,
        answering_confidence_threshold_upper,
        answering_confidence_threshold_lower,
        fast_threshold,
        pop_limit,
        threshold_descrease,
        image_folder: str = None,
        search_mode=True,
        enable_parent_verification=False,
        node_ranker=None,
        answer_observer=None,
        method_trace=None,
        search_state_sink=None,
):
    # Data loading
    #Default single_target: tree_depth_s = 2, cross_target: tree_depth_c = 3
    tree_prune_threshold = 0.4
    tree_depth_s = 2
    tree_depth_c = 3
    input_image = annotation['input_image']
    if image_folder is not None:
        input_image = os.path.join(image_folder, input_image)
    image_pil = Image.open(input_image).convert('RGB')
    question = annotation['question']
    options = annotation.get('options', None)
    question_free_form = annotation.get("text")
    searched_nodes = []
    source_image_identity = _source_image_identity(image_pil) if search_state_sink is not None else None
    search_call_ordinal = 0

    def next_search_state_context(tree_scope, crop_origin):
        nonlocal search_call_ordinal
        search_call_ordinal += 1
        if search_state_sink is None:
            return None
        return _make_search_state_context(
            tree_scope, crop_origin, source_image_identity, search_call_ordinal,
        )
    ####Quick assessment
    img_w, img_h = image_pil.size
    state = NodeState(image_pil, [0, 0, img_w, img_h])
    root_node = NodeA(state)
    root_node.is_root = True
    root_node.search_source = "global"
    root_ans_conf = zoom_model.get_confidence_value([root_node], image_pil, confidence_type='answering',input_ele=question)
    annotation['root_ans_conf'] = root_ans_conf
    annotation['sam'] = []
    if root_ans_conf>answering_confidence_threshold_lower+fast_threshold:
        #####Quick Answer########
        # print("Quick Answer!")
        searched_nodes.append(root_node)
        annotation['targets'] = None
        annotation['target_sign'] = None
        annotation['num_pop'] = []
        annotation['num_zoom_in'] = []
        annotation['num_zoom_out'] = []
        annotation['search_mode'] = 0
        annotation['sam'].append(None)
    else:
        #####Visual Search########
        # print("Visual Search!")
        # key object extract
        targets = zoom_model.generate_visual_cues_using_ic(ic_examples, question)
        #For the visual cues like "man and his bag", we should remove the pronoun "his bag"
        targets = [t for t in targets if not include_pronouns(nlp_model, t)]
        #For the visual cues like "all dogs", we convert it to "dog"
        processed_results = [normalize_target_text(t) for t in targets]
        targets = [res[0] for res in processed_results]
        # is_type2_triggered = any(res[1] for res in processed_results)
        is_search_second =False
        # sam3
        target_sign = True if len(targets)>0 else False
        ###MLLM extract key objects
        if target_sign:
            text_target = targets
            with torch.inference_mode():
                backbone_out, processed_results, target_id = sam_model.batch_inference(image_pil, text_target)
        else:
            text_target = extract_visual_objects(nlp_model, question)
            with torch.inference_mode():
                backbone_out, processed_results, target_id = sam_model.batch_inference(image_pil, text_target)
        one_target_search = (len(text_target) == 1)

        # print("targets:", targets)
        # print('text_target:', text_target)
        annotation['targets'] = text_target
        annotation['target_sign'] = target_sign
        annotation['num_pop'] = []
        annotation['num_zoom_in'] = []
        annotation['num_zoom_out'] = []

        ####sam3 result -> bbox
        sam_success_flags, sam_bboxes = process_sam_result(processed_results, target_id, is_search_second)

        # Adaptive visual search
        if target_sign:
            # print('MLLM Visual Cue!')
            # MLLM extract target objects
            if sum(sam_success_flags) == len(text_target):
                # sam3 segment all target objects
                fast_node = []
                for search_box in sam_bboxes:
                    x0, y0 = search_box[0], search_box[1]
                    w, h = search_box[2] - search_box[0], search_box[3] - search_box[1]
                    bbox_xywh = [x0, y0, w, h]
                    state = NodeState(image_pil, bbox_xywh)
                    node = NodeA(state)
                    node.search_source = "fast"
                    fast_node.append(node)

                # print("Fast Search Success!")
                searched_nodes.extend(fast_node)
                num_pop = 0
                num_zoom_in = 0
                num_zoom_out = 0
                annotation['num_pop'].append(num_pop)
                annotation['num_zoom_in'].append(num_zoom_in)
                annotation['num_zoom_out'].append(num_zoom_out)
                annotation['search_mode'] = 1
                annotation['sam'].append(True)
            #Fast search fail
            else:
                # print("Fast Search Fail!")
                zoom_node = [] #
                #sam3 segment partial target objects
                image_features_batch = backbone_out['vision_features']
                if isinstance(image_features_batch, torch.Tensor):
                    feat = image_features_batch.detach().cpu().float().numpy()
                else:
                    feat = image_features_batch

                del backbone_out
                del image_features_batch

                feat = feat.squeeze(0)  # batch -> (256, 72, 72), C,H,W
                tree_depth = 3
                builder = ConstrainedTreeBuilder(feat, n_atoms=600, pos_weight=3.5, split_threshold=0.3, keep_threshold=0.15)
                tree_dict = builder.build_tree(max_depth=tree_depth, min_splits=4, max_splits=8)
                feat_shape = feat.shape
                image_tree = AdaptiveImageTree(image_pil, tree_dict, feat_shape)
                num_pop = []
                for flag, search_box, t_target in zip(sam_success_flags, sam_bboxes, text_target):
                    if flag==1:
                        #Successfully Segment
                        x0, y0 = search_box[0], search_box[1]
                        w, h = search_box[2] - search_box[0], search_box[3] - search_box[1]
                        bbox_xywh = [x0, y0, w, h]
                        state = NodeState(image_pil, bbox_xywh)
                        node = NodeA(state)
                        node.search_source = "fast"
                        zoom_node.append(node)
                        num_pop.append(1)
                        annotation['sam'].append(True)
                    else:
                        annotation['sam'].append(False)
                        candidates_search, num_pop_search, is_success = semantic_guide_search_dynamic_depth(
                            zoom_model=zoom_model,
                            pop_limit=pop_limit,
                            num_intervel=2,
                            threshold_descrease=threshold_descrease,
                            depth_limit=tree_depth_s if one_target_search else tree_depth_c,
                            question=question if one_target_search else decomposed_question_template.format(t_target),
                            visual_cue=t_target,
                            answering_confidence_threshold_lower=answering_confidence_threshold_lower,
                            answering_confidence_threshold_upper=answering_confidence_threshold_upper,
                            image_pil=image_pil,
                            image_tree=image_tree,
                            enable_parent_verification=enable_parent_verification,
                            prior_pruning_threshold=tree_prune_threshold,
                            node_ranker=node_ranker,
                            rank_context=_make_rank_context(method_trace, question, t_target, 'main', (0, 0)) if node_ranker is not None else None,
                            search_state_sink=search_state_sink,
                            search_state_context=next_search_state_context('main', (0, 0)),
                        )
                        num_pop.append(num_pop_search)
                        if is_success:
                            #Search successfully
                            for cand in candidates_search:
                                cand.search_source = "fine"

                            zoom_node.extend(candidates_search)
                            # print("Fine Search Success!")
                        else:
                            # Search fail: candidates_search is the sorted node list
                            if candidates_search:
                                best_candidate = candidates_search[0]  # first Node
                                if not search_mode:
                                    best_candidate.search_source = "fine_fallback"
                                    zoom_node.append(best_candidate)
                                    # print("Fine Search Fail!")
                                else:
                                    ###Second search
                                    is_search_second = True
                                    cropped_image, cropped_bbox = crop_image_by_node(image_pil, best_candidate)
                                    if cropped_image:
                                        left, top = cropped_bbox[0], cropped_bbox[1]
                                        with torch.inference_mode():
                                            backbone_out_sub, processed_results_sub, target_id_sub = sam_model.batch_inference(cropped_image, [t_target])
                                        image_features_batch_sub = backbone_out_sub['vision_features']
                                        if isinstance(image_features_batch_sub, torch.Tensor):
                                            feat_sub = image_features_batch_sub.detach().cpu().float().numpy()
                                        else:
                                            feat_sub = image_features_batch_sub
                                        feat_sub = feat_sub.squeeze(0)
                                        del backbone_out_sub
                                        del image_features_batch_sub

                                        sam_success_flags_sub, sam_bboxes_sub = process_sam_result(processed_results_sub, target_id_sub, is_search_second)
                                        if sum(sam_success_flags_sub) == len([t_target]):
                                            # second sam3 inference successfully segmented target objects
                                            fast_node = []
                                            for search_box in sam_bboxes_sub:
                                                x0, y0 = search_box[0], search_box[1]
                                                w, h = search_box[2] - search_box[0], search_box[3] - search_box[1]
                                                bbox_xywh = [x0+left, y0+top, w, h]  ###bbox offset
                                                state = NodeState(image_pil, bbox_xywh)
                                                node = NodeA(state)
                                                node.search_source = "fast"
                                                fast_node.append(node)

                                            # print("Second Fast Search Success!")
                                            zoom_node.extend(fast_node)
                                        else:
                                            # second segmentation still failed
                                            tree_depth_sub = tree_depth_s
                                            depth_limit_sub = tree_depth_s
                                            builder_sub = ConstrainedTreeBuilder(feature_map=feat_sub, n_atoms=600,
                                                                                 pos_weight=3.5, split_threshold=0.3,
                                                                                 keep_threshold=0.15,
                                                                                 use_local_normalization=True,
                                                                                 use_silhouette_score=True)
                                            tree_sub = builder_sub.build_tree(max_depth=tree_depth_sub, min_splits=4, max_splits=8)
                                            feat_shape = feat.shape
                                            image_tree_sub = AdaptiveImageTree(cropped_image, tree_sub, feat_shape)
                                            candidates_search_sub, num_pop_search_sub, is_success_sub = semantic_guide_search_dynamic_depth(
                                                zoom_model=zoom_model,
                                                pop_limit=pop_limit,
                                                num_intervel=2,
                                                threshold_descrease=threshold_descrease,
                                                depth_limit=depth_limit_sub,
                                                question=question if one_target_search else decomposed_question_template.format(t_target),
                                                visual_cue=t_target,
                                                answering_confidence_threshold_lower=answering_confidence_threshold_lower,
                                                answering_confidence_threshold_upper=answering_confidence_threshold_upper,
                                                image_pil=cropped_image,
                                                image_tree=image_tree_sub,
                                                enable_parent_verification=enable_parent_verification,
                                                prior_pruning_threshold=tree_prune_threshold,
                                                node_ranker=node_ranker,
                                                rank_context=_make_rank_context(method_trace, question, t_target, 'cropped', (left, top)) if node_ranker is not None else None,
                                                search_state_sink=search_state_sink,
                                                search_state_context=next_search_state_context('cropped', (left, top)),
                                            )

                                            if is_success_sub:
                                                second_search_node=candidates_search_sub[0]
                                                bbox = second_search_node.state.bbox
                                                x0, y0, w, h = bbox
                                                bbox_shifted = [x0+left, y0+top, w, h]
                                                state = NodeState(image_pil, bbox_shifted)
                                                node = NodeA(state)
                                                node.search_source = "fine"
                                                zoom_node.append(node)
                                                # print("Second Fine Search Success!")
                                            else:
                                                best_candidate.search_source = "fine_fallback"
                                                zoom_node.append(best_candidate)
                                                # print("Fine Search Fail!")

                if len(zoom_node) > 0:
                    searched_nodes.extend(zoom_node)
                    num_zoom_in = 0
                    num_zoom_out = 0
                    annotation['num_pop'].append(num_pop)
                    annotation['num_zoom_in'].append(num_zoom_in)
                    annotation['num_zoom_out'].append(num_zoom_out)
                    annotation['search_mode'] = 2
                else:
                    annotation['search_mode'] = 3

        else:
            # print('Rules Visual Cue!')
            # MLLM extraction failed, rule matches target objects
            if sum(sam_success_flags) == len(text_target):
                fast_node = []
                for search_box in sam_bboxes:
                    x0, y0 = search_box[0], search_box[1]
                    w, h = search_box[2] - search_box[0], search_box[3] - search_box[1]
                    bbox_xywh = [x0, y0, w, h]
                    state = NodeState(image_pil, bbox_xywh)
                    node = NodeA(state)
                    node.search_source = "fast"
                    fast_node.append(node)

                # print("Fast Search Success!")
                searched_nodes.extend(fast_node)
                num_pop = 0
                num_zoom_in = 0
                num_zoom_out = 0
                annotation['num_pop'].append(num_pop)
                annotation['num_zoom_in'].append(num_zoom_in)
                annotation['num_zoom_out'].append(num_zoom_out)
                annotation['search_mode'] = 1
                annotation['sam'].append(True)
            #Fast search fail
            else:
                # print("Fast Search Fail!")
                zoom_node = []
                image_features_batch = backbone_out['vision_features']
                if isinstance(image_features_batch, torch.Tensor):
                    feat = image_features_batch.detach().cpu().float().numpy()
                else:
                    feat = image_features_batch

                del backbone_out
                del image_features_batch

                feat = feat.squeeze(0)
                tree_depth = 3
                builder = ConstrainedTreeBuilder(feat, n_atoms=600, pos_weight=3.5, split_threshold=0.3, keep_threshold=0.25)
                tree_dict = builder.build_tree(max_depth=tree_depth, min_splits=4, max_splits=8)
                feat_shape = feat.shape
                image_tree = AdaptiveImageTree(image_pil, tree_dict, feat_shape)
                num_pop = []
                for flag, search_box, t_target in zip(sam_success_flags, sam_bboxes, text_target):
                    if flag==1:
                        x0, y0 = search_box[0], search_box[1]
                        w, h = search_box[2] - search_box[0], search_box[3] - search_box[1]
                        bbox_xywh = [x0, y0, w, h]
                        state = NodeState(image_pil, bbox_xywh)
                        node = NodeA(state)
                        node.search_source = "fast"
                        zoom_node.append(node)
                        num_pop.append(1)
                        annotation['sam'].append(True)
                    else:
                        annotation['sam'].append(False)
                        candidates_search, num_pop_search, is_success = semantic_guide_search_dynamic_depth(
                            zoom_model=zoom_model,
                            pop_limit=pop_limit,
                            num_intervel=2,
                            threshold_descrease=threshold_descrease,
                            depth_limit=tree_depth_s if one_target_search else tree_depth_c,
                            question=question if one_target_search else decomposed_question_template.format(t_target),
                            visual_cue=t_target,
                            answering_confidence_threshold_lower=answering_confidence_threshold_lower,
                            answering_confidence_threshold_upper=answering_confidence_threshold_upper,
                            image_pil=image_pil,
                            image_tree=image_tree,
                            enable_parent_verification=enable_parent_verification,
                            prior_pruning_threshold=tree_prune_threshold,
                            node_ranker=node_ranker,
                            rank_context=_make_rank_context(method_trace, question, t_target, 'main', (0, 0)) if node_ranker is not None else None,
                            search_state_sink=search_state_sink,
                            search_state_context=next_search_state_context('main', (0, 0)),
                        )
                        num_pop.append(num_pop_search)
                        if is_success:
                            for cand in candidates_search:
                                cand.search_source = "fine"
                            zoom_node.extend(candidates_search)
                        else:
                            if candidates_search:
                                best_candidate = candidates_search[0]
                                if not search_mode:
                                    best_candidate.search_source = "fine_fallback"
                                    zoom_node.append(best_candidate)
                                    # print("Fine Search Fail!")
                                else:
                                    is_search_second = True
                                    cropped_image, cropped_bbox = crop_image_by_node(image_pil, best_candidate)
                                    if cropped_image:
                                        left, top = cropped_bbox[0], cropped_bbox[1]
                                        with torch.inference_mode():
                                            backbone_out_sub, processed_results_sub, target_id_sub = sam_model.batch_inference(
                                                cropped_image, [t_target])
                                        image_features_batch_sub = backbone_out_sub['vision_features']
                                        if isinstance(image_features_batch_sub, torch.Tensor):
                                            feat_sub = image_features_batch_sub.detach().cpu().float().numpy()
                                        else:
                                            feat_sub = image_features_batch_sub
                                        feat_sub = feat_sub.squeeze(0)
                                        del backbone_out_sub
                                        del image_features_batch_sub

                                        sam_success_flags_sub, sam_bboxes_sub = process_sam_result(
                                            processed_results_sub, target_id_sub, is_search_second)
                                        if sum(sam_success_flags_sub) == len([t_target]):
                                            fast_node = []
                                            for search_box in sam_bboxes_sub:
                                                x0, y0 = search_box[0], search_box[1]
                                                w, h = search_box[2] - search_box[0], search_box[3] - search_box[1]
                                                bbox_xywh = [x0 + left, y0 + top, w, h]
                                                state = NodeState(image_pil, bbox_xywh)
                                                node = NodeA(state)
                                                node.search_source = "fast"
                                                fast_node.append(node)

                                            # print("Second Fast Search Success!")
                                            zoom_node.extend(fast_node)
                                        else:
                                            tree_depth_sub = tree_depth_s
                                            depth_limit_sub = tree_depth_s
                                            builder_sub = ConstrainedTreeBuilder(feature_map=feat_sub, n_atoms=600,
                                                                                 pos_weight=3.5,
                                                                                 split_threshold=0.3,
                                                                                 keep_threshold=0.15,
                                                                                 use_local_normalization=True,
                                                                                 use_silhouette_score=True)
                                            tree_sub = builder_sub.build_tree(max_depth=tree_depth_sub,
                                                                              min_splits=4, max_splits=8)
                                            feat_shape = feat.shape
                                            image_tree_sub = AdaptiveImageTree(cropped_image, tree_sub, feat_shape)
                                            candidates_search_sub, num_pop_search_sub, is_success_sub = semantic_guide_search_dynamic_depth(
                                                zoom_model=zoom_model,
                                                pop_limit=pop_limit,
                                                num_intervel=2,
                                                threshold_descrease=threshold_descrease,
                                                depth_limit=depth_limit_sub,
                                                question=question if one_target_search else decomposed_question_template.format(t_target),
                                                visual_cue=t_target,
                                                answering_confidence_threshold_lower=answering_confidence_threshold_lower,
                                                answering_confidence_threshold_upper=answering_confidence_threshold_upper,
                                                image_pil=cropped_image,
                                                image_tree=image_tree_sub,
                                                enable_parent_verification=enable_parent_verification,
                                                prior_pruning_threshold=tree_prune_threshold,
                                                node_ranker=node_ranker,
                                                rank_context=_make_rank_context(method_trace, question, t_target, 'cropped', (left, top)) if node_ranker is not None else None,
                                                search_state_sink=search_state_sink,
                                                search_state_context=next_search_state_context('cropped', (left, top)),
                                            )

                                            if is_success_sub:
                                                second_search_node = candidates_search_sub[0]
                                                bbox = second_search_node.state.bbox
                                                x0, y0, w, h = bbox
                                                bbox_shifted = [x0 + left, y0 + top, w, h]
                                                state = NodeState(image_pil, bbox_shifted)
                                                node = NodeA(state)
                                                node.search_source = "fine"
                                                zoom_node.append(NodeA(state))
                                                # print("Second Fine Search Success!")
                                            else:
                                                best_candidate.search_source = "fine_fallback"
                                                zoom_node.append(best_candidate)
                                                # print("Fine Search Fail!")

                if len(zoom_node) > 0:
                    searched_nodes.extend(zoom_node)
                    num_zoom_in = 0
                    num_zoom_out = 0
                    annotation['num_pop'].append(num_pop)
                    annotation['num_zoom_in'].append(num_zoom_in)
                    annotation['num_zoom_out'].append(num_zoom_out)
                    annotation['search_mode'] = 2
                else:
                    annotation['search_mode'] = 3

    annotation['searched_bbox'] = [node.state.bbox for node in searched_nodes]
    _emit_p0_selected(search_state_sink, image_pil, searched_nodes)
    answer_type = annotation.get('answer_type', 'free_form')
    # For vstar
    if answer_type == "logits_match":
        option_choose = zoom_model.multiple_choices_inference(image_pil, question, options, searched_nodes)
        return _observe_answer(answer_observer, annotation, searched_nodes, option_choose)
    elif answer_type == "free_form":
        if question_free_form:
            response = zoom_model.free_form_using_nodes(image_pil, question_free_form, searched_nodes)
        else:
            response = zoom_model.free_form_using_nodes(image_pil, question, searched_nodes)
        return _observe_answer(answer_observer, annotation, searched_nodes, response)
    # For hr-bench
    elif answer_type == "option_list":
        answers = []
        for option_str in options:
            question_input = format_question(question, option_str)
            answers.append(zoom_model.free_form_using_nodes(image_pil, question_input, searched_nodes))
        return _observe_answer(answer_observer, annotation, searched_nodes, answers)
    # For mme-realworld
    elif answer_type == "Multiple Choice":
        question_input = format_question_multichoice(question, options)
        response = zoom_model.free_form_using_nodes(image_pil, question_input, searched_nodes)
        return _observe_answer(answer_observer, annotation, searched_nodes, response)
    elif answer_type == "option_single":
        question_input = format_question_new(question, options)
        response = zoom_model.free_form_using_nodes(image_pil, question_input, searched_nodes)
        return _observe_answer(answer_observer, annotation, searched_nodes, response)
    else:
        raise NotImplementedError


def process_sam_result(processed_results, target_id, is_second_search):
    """
    process SAM result
    """
    sam_success_flags = []
    sam_bboxes = []
    for t_id in target_id:
        # 1. boxes and scores
        boxes = processed_results[t_id]["boxes"].float().cpu().numpy()
        scores = processed_results[t_id]["scores"].float().cpu().numpy()
        # 2. Check if there are valid bbox
        if boxes.size > 0 and boxes.ndim > 1 and boxes.shape[1] >= 4:
            if not is_second_search:
                min_x1 = np.min(boxes[:, 0])
                min_y1 = np.min(boxes[:, 1])
                max_x2 = np.max(boxes[:, 2])
                max_y2 = np.max(boxes[:, 3])
                merged_bbox = [int(min_x1), int(min_y1), int(max_x2), int(max_y2)]
                sam_bboxes.append(merged_bbox)
                sam_success_flags.append(1)
            else:
                current_target_all_boxes = []
                for box in boxes:
                    bbox_int = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                    current_target_all_boxes.append(bbox_int)

                sam_bboxes.extend(current_target_all_boxes)
                sam_success_flags.append(1)
        else:
            sam_success_flags.append(0)
            sam_bboxes.append([])

    return sam_success_flags, sam_bboxes

def crop_image_by_node(
        image_pil: Image.Image,
        node: dict,
):
    """
    Args:
        image_pil (PIL.Image)
        node (dict): bbox format (x1, x1, w, h)
    Returns:
        PIL.Image: Image Patch
    """
    orig_w, orig_h = image_pil.size
    if isinstance(node, dict):
        bbox = node['bbox']
    elif hasattr(node, 'bbox'):
        bbox = node.bbox
    elif hasattr(node, 'state') and hasattr(node.state, 'bbox'):
        bbox = node.state.bbox
    else:
        raise ValueError("Provided node does not contain valid bbox information.")
    x1, y1, w, h = bbox
    # 4. Feature Map -> Original Image
    oy1 = int(y1)
    ox1 = int(x1)
    oy2 = int(y1+h)
    ox2 = int(x1+w)
    # 5. PIL crop format: left, top, right, bottom
    crop_box = (
        max(0, ox1),  # left
        max(0, oy1),  # top
        min(orig_w, ox2),  # right
        min(orig_h, oy2)  # bottom
    )
    if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
        patch = image_pil.crop(crop_box)
        return patch, crop_box
    else:
        return None, None

def format_question(question, option_str):
    return question + '\n' + option_str + 'Answer the option letter directly.'

def format_question_new(question, option_str):
    return question + " Options:\n" + option_str + "\nSelect the best answer to the above multiple-choice question based on the image. Respond with only the letter of the correct option.\nThe best answer is:"

def format_question_multichoice(question, options):
    ret = question
    for o in options:
        ret += '\n'
        ret += o
    # This prompt is copied from the original paper of MME-RealWorld
    ret += '\nSelect the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.\nThe best answer is:'
    return ret

def semantic_guide_search_dynamic_depth(
        zoom_model,
        pop_limit: Union[int, Callable],
        num_intervel: int,
        threshold_descrease: List[float],
        depth_limit: int,
        question: str,
        visual_cue: str,
        answering_confidence_threshold_lower: float,
        answering_confidence_threshold_upper: float,
        image_pil=None,
        image_tree=None,
        w_current: float = 0.4,
        w_child: float = 0.4,
        w_prior: float = 0.2,
        prior_pruning_threshold: float = 0.4,
        parent_verification_threshold: float = 0.0,
        high_confidence_bypass: float = 0.8,
        enable_parent_verification: bool = True,
        node_ranker=None,
        rank_context=None,
        search_state_sink=None,
        search_state_context=None,
) -> Tuple[List, int, bool]:
    # -------------------------------------------------------------------------
    # 0. Initialization and dynamic depth detection
    # -------------------------------------------------------------------------
    # Determine the maximum depth allowed for this search
    actual_max_depth = min(image_tree.max_depth, depth_limit)
    pop_num_limit = pop_limit(actual_max_depth) if callable(pop_limit) else pop_limit
    state_context = (
        _state_context_or_default(search_state_context, image_pil)
        if search_state_sink is not None else None
    )

    nodes_by_depth = {}
    queue = [image_tree.root]
    while queue:
        node = queue.pop(0)
        if 1 <= node.depth <= actual_max_depth:
            if node.depth not in nodes_by_depth:
                nodes_by_depth[node.depth] = []
            nodes_by_depth[node.depth].append(node)
            node.aggregated_confidence = -1.0
            node.posterior_score = -1.0
            node.fast_confidence = None

        if node.depth < actual_max_depth:
            queue.extend(node.children)

    total_pop = 0

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------
    def calc_existence_and_update_parent(node):
        if node.fast_confidence is None:
            if node.prior_prob > prior_pruning_threshold:
                existence = zoom_model.get_confidence_value([node], image_pil, confidence_type='existence',input_ele=visual_cue)
                node.fast_confidence = existence
                node.is_evaluated = True
            else:
                node.fast_confidence = -1.0
                node.is_evaluated = False

        if node.parent and hasattr(node.parent, 'aggregated_confidence'):
            node.parent.aggregated_confidence = max(node.parent.aggregated_confidence, node.fast_confidence)

    def calc_score_and_sort(nodes, use_child_info=True):
        valid_nodes_for_sorting = []
        for node in nodes:
            calc_existence_and_update_parent(node)
            if not getattr(node, 'is_evaluated', False):
                node.posterior_score = -999.0
                continue

            norm_fast_conf = (node.fast_confidence + 1.0) / 2.0
            norm_agg = (node.aggregated_confidence + 1.0) / 2.0

            if use_child_info:
                score = (w_current * norm_fast_conf) + (w_child * norm_agg) + (w_prior * node.prior_prob)
            else:
                sum_local = w_current + w_prior
                n_wc = w_current / sum_local if sum_local > 0 else 0.5
                n_wp = w_prior / sum_local if sum_local > 0 else 0.5
                score = (n_wc * norm_fast_conf) + (n_wp * node.prior_prob)

            node.posterior_score = score
            valid_nodes_for_sorting.append(node)
        return sorted(valid_nodes_for_sorting, key=lambda x: x.posterior_score, reverse=True)

    def apply_node_ranker(nodes, stage_name):
        context = {} if rank_context is None else rank_context
        main_query = context.get('main_query', question)
        if not isinstance(main_query, str) or not main_query.strip():
            raise ValueError("rank main_query must be a nonempty string")
        augmented_queries = context.get('augmented_queries', [visual_cue])
        result = node_ranker(nodes, image_pil, main_query, augmented_queries)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError("node_ranker must return (ranked_nodes, details)")
        try:
            ranked_nodes = list(result[0])
            details = list(result[1])
        except TypeError as error:
            raise ValueError("node_ranker results must be iterable") from error
        if len(ranked_nodes) != len(nodes) or sorted(map(id, ranked_nodes)) != sorted(map(id, nodes)):
            raise ValueError("node_ranker must preserve the candidate identity multiset")
        if len(details) != len(nodes):
            raise ValueError("node_ranker details must match the candidate count")

        tree_scope = context.get('tree_scope', 'main')
        crop_origin = context.get('crop_origin', (0, 0))
        if tree_scope not in ('main', 'cropped'):
            raise ValueError("node_ranker tree_scope must be main or cropped")
        try:
            origin_x, origin_y = crop_origin
        except (TypeError, ValueError) as error:
            raise ValueError("node_ranker crop_origin must have two values") from error

        def json_number(value, name):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"node_ranker {name} must be numeric")
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"node_ranker {name} must be numeric") from error
            if not np.isfinite(number):
                raise ValueError(f"node_ranker {name} must be finite")
            return int(number) if number.is_integer() else number

        origin_x = json_number(origin_x, 'crop_origin')
        origin_y = json_number(origin_y, 'crop_origin')
        enriched_details = []
        for node, detail in zip(ranked_nodes, details):
            if not isinstance(detail, Mapping):
                raise ValueError("node_ranker details must be mappings")
            enriched = dict(detail)
            node_id = getattr(node, 'id', None)
            if 'node_id' in enriched and enriched['node_id'] != node_id:
                raise ValueError("node_ranker detail node_id must match ranked node")
            try:
                x, y, width, height = node.state.bbox
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError("node_ranker node bbox must have four values") from error
            x = json_number(x, 'bbox')
            y = json_number(y, 'bbox')
            width = json_number(width, 'bbox')
            height = json_number(height, 'bbox')
            enriched.update({
                'node_id': node_id,
                'target': visual_cue,
                'stage': stage_name,
                'tree_scope': tree_scope,
                'crop_origin': [origin_x, origin_y],
                'bbox_original': [x + origin_x, y + origin_y, width, height],
            })
            try:
                json.dumps(enriched, allow_nan=False)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("node_ranker details must be strict JSON-safe") from error
            enriched_details.append(enriched)
        method_trace = context.get('method_trace')
        if method_trace is not None:
            method_trace.candidate_ranks.extend(enriched_details)
        return ranked_nodes

    def emit_search_state(event, stage_name, depth, ordered_nodes, popped_nodes=(),
                          selected_nodes=(), remaining_nodes=()):
        if search_state_sink is None:
            return
        crop_origin = state_context["crop_origin"]
        candidates = [
            _state_node_snapshot(node, crop_origin, stage_rank=index)
            for index, node in enumerate(ordered_nodes)
        ]
        snapshot = {
            "schema_version": 1,
            "event": event,
            **state_context,
            "visual_cue": visual_cue,
            "stage": stage_name,
            "depth": depth,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "ordered_keys": [_state_node_key(node, crop_origin) for node in ordered_nodes],
            "popped_keys": [_state_node_key(node, crop_origin) for node in popped_nodes],
            "selected_keys": [_state_node_key(node, crop_origin) for node in selected_nodes],
            "remaining_keys": [_state_node_key(node, crop_origin) for node in remaining_nodes],
        }
        json.dumps(snapshot, allow_nan=False)
        live_refs = MappingProxyType({
            "ordered_nodes": tuple(_frozen_search_candidate(node, state_context) for node in ordered_nodes),
            "popped_nodes": tuple(_frozen_search_candidate(node, state_context) for node in popped_nodes),
            "selected_nodes": tuple(_frozen_search_candidate(node, state_context) for node in selected_nodes),
            "remaining_nodes": tuple(_frozen_search_candidate(node, state_context) for node in remaining_nodes),
        })
        search_state_sink(live_refs, deepcopy(snapshot))

    def execute_stage_search(Q, stage_name, stage_depth, ordered_nodes,
                             start_pop_count, check_parent=False):
        pop_trace = []
        current_threshold = answering_confidence_threshold_upper
        temp_threshold_descrease = deepcopy(threshold_descrease)
        next_checkpoint = pop_num_limit
        last_step = 0.05
        local_pop = 0
        stage_finished = False

        def finish(selected_nodes=()):
            nonlocal stage_finished
            if stage_finished:
                return
            stage_finished = True
            emit_search_state(
                "stage_finished", stage_name, stage_depth, ordered_nodes,
                popped_nodes=pop_trace,
                selected_nodes=selected_nodes,
                remaining_nodes=Q,
            )

        def validate_node(node, confidence):
            if not check_parent or not node.parent:
                return True, "No Check"

            if node.parent.fast_confidence is None:
                p_exist = zoom_model.get_confidence_value([node.parent], image_pil, confidence_type='existence',input_ele=visual_cue)
                node.parent.fast_confidence = p_exist
            parent_conf = node.parent.fast_confidence
            if parent_conf >= parent_verification_threshold:
                return True, f"Parent Confirmed ({parent_conf:.2f})"
            if confidence >= high_confidence_bypass:
                return True, f"Bypass (Child {confidence:.2f} >> Parent {parent_conf:.2f})"
            return False, f"Rejected (Child {confidence:.2f} & Parent {parent_conf:.2f})"

        while len(Q) > 0:
            cur_node = Q.pop(0)
            local_pop += 1
            ans_conf = zoom_model.get_confidence_value([cur_node], image_pil, confidence_type='answering', input_ele=question)
            cur_node.answering_confidence = ans_conf
            pop_trace.append(cur_node)
            # print(f"[{stage_name}] ID:{cur_node.id} | Ans:{ans_conf:.4f}")

            if ans_conf >= current_threshold:
                is_valid, reason = validate_node(cur_node, ans_conf)
                if is_valid:
                    # print(f"  -> {reason} >>> Hit! Node {cur_node.id}")
                    finish((cur_node,))
                    return True, [cur_node], local_pop
                # else:
                #     print(f"  -> {reason} (Searching next...)")

            if local_pop >= next_checkpoint:
                # print(f"--- {stage_name} Checkpoint reached. Adjusting Threshold... ---")
                step = 0.0
                if len(temp_threshold_descrease) > 0:
                    step = temp_threshold_descrease.pop(0)
                    last_step = step
                else:
                    step = last_step
                if step > 0:
                    current_threshold -= step
                    current_threshold = max(current_threshold, answering_confidence_threshold_lower)
                    # print(f"New Threshold: {current_threshold:.4f}")

                    candidates = [n for n in pop_trace if n.answering_confidence >= current_threshold]
                    if candidates:
                        candidates.sort(key=lambda x: x.answering_confidence, reverse=True)
                        for cand in candidates:
                            is_valid, reason = validate_node(cand, cand.answering_confidence)
                            if is_valid:
                                # print(f">>> {stage_name} Hit via Decay! Node {cand.id} (Reason: {reason})")
                                finish((cand,))
                                return True, [cand], local_pop
                            # else:
                            #     print(f"  [Decay Check] Node {cand.id} skipped: {reason}")
                next_checkpoint += num_intervel
                if current_threshold <= answering_confidence_threshold_lower:
                    # print(f"Threshold hit lower bound. Stopping {stage_name}.")
                    break

        # print(f"--- {stage_name} Search Exhausted. Final Check... ---")
        if pop_trace:
            final_cands = [n for n in pop_trace if n.answering_confidence >= answering_confidence_threshold_lower]
            if final_cands:
                final_cands.sort(key=lambda x: x.answering_confidence, reverse=True)
                for cand in final_cands:
                    is_valid, reason = validate_node(cand, cand.answering_confidence)
                    if is_valid:
                        # print(f">>> {stage_name} Hit via Final Check! Node {cand.id} (Reason: {reason})")
                        finish((cand,))
                        return True, [cand], local_pop
                    # else:
                    #     print(f"  [Final Check] Node {cand.id} skipped: {reason}")
        finish()
        return False, [], local_pop

    # -------------------------------------------------------------------------
    # Main process: dynamic hierarchical search
    # -------------------------------------------------------------------------
    search_depths = sorted([d for d in nodes_by_depth.keys() if d > 1], reverse=True)
    for idx, depth in enumerate(search_depths):
        is_bottom_layer = (idx == 0)
        stage_name = f"Depth {depth}"
        # print(f"\n=== Stage {idx + 1}: Searching {stage_name} (Total {len(nodes_by_depth[depth])} nodes) ===")

        current_use_child_info = not is_bottom_layer
        current_check_parent = is_bottom_layer and enable_parent_verification

        Q = calc_score_and_sort(nodes_by_depth[depth], use_child_info=current_use_child_info)
        if node_ranker is not None and Q:
            Q = apply_node_ranker(Q, stage_name)
        ordered_nodes = tuple(Q)
        emit_search_state("stage_ready", stage_name, depth, ordered_nodes, remaining_nodes=Q)

        success, res, count = execute_stage_search(
            Q, stage_name, depth, ordered_nodes, total_pop, check_parent=current_check_parent
        )

        total_pop += count
        if success:
            return res, total_pop, True

    # -------------------------------------------------------------------------
    # Stage Final: Depth 1
    # -------------------------------------------------------------------------
    if 1 in nodes_by_depth:
        # print(f"\n=== Final Stage: Searching Depth 1 (Total {len(nodes_by_depth[1])} nodes) ===")
        Q = calc_score_and_sort(nodes_by_depth[1], use_child_info=True)
        if node_ranker is not None and Q:
            Q = apply_node_ranker(Q, "Depth 1")
        ordered_nodes = tuple(Q)
        emit_search_state("stage_ready", "Depth 1", 1, ordered_nodes, remaining_nodes=Q)
        if Q:
            target = Q[0]
            total_pop += 1
            ans_conf = zoom_model.get_confidence_value([target], image_pil, confidence_type='answering',input_ele=question)
            target.answering_confidence = ans_conf
            # print(f"[Depth 1] Best Node {target.id} | Ans: {ans_conf:.4f}")
            if ans_conf >= answering_confidence_threshold_lower:
                emit_search_state(
                    "stage_finished", "Depth 1", 1, ordered_nodes,
                    popped_nodes=(target,), selected_nodes=(target,), remaining_nodes=Q[1:],
                )
                return [target], total_pop, True

        all_d1 = sorted(nodes_by_depth[1], key=lambda x: getattr(x, 'posterior_score', -1), reverse=True)
        emit_search_state(
            "stage_finished", "Depth 1", 1, ordered_nodes,
            popped_nodes=(() if not Q else (Q[0],)), remaining_nodes=Q[1:] if Q else (),
        )
        return all_d1, total_pop, False

    emit_search_state("stage_ready", "No Depth", 0, (), remaining_nodes=())
    emit_search_state("stage_finished", "No Depth", 0, (), remaining_nodes=())
    return [], total_pop, False


def get_direct_response(
        zoom_model: Model,
        annotation,
        image_folder
):
    input_image = annotation['input_image']
    if image_folder is not None:
        input_image = os.path.join(image_folder, input_image)
    question = annotation['question']
    options = annotation.get('options', None)

    image_pil = Image.open(input_image).convert('RGB')

    # An empty list will conduct direct answering.
    searched_nodes = []
    answer_type = annotation.get('answer_type', 'free_form')
    # For vstar
    if answer_type == "logits_match":
        option_choose = zoom_model.multiple_choices_inference(image_pil, question, options, searched_nodes)
        return option_choose
    elif answer_type == "free_form":
        return zoom_model.free_form_using_nodes(
            image_pil, annotation.get("text", question), searched_nodes,
        )
    # For hr-bench
    elif answer_type == "option_list":
        answers = []
        for option_str in options:
            question_input = format_question(question, option_str)
            answers.append(zoom_model.free_form_using_nodes(image_pil, question_input, searched_nodes))
        return answers
    elif answer_type == "option_single":
        question_input = format_question_new(question, options)
        response = zoom_model.free_form_using_nodes(image_pil, question_input, searched_nodes)
        return response
    elif answer_type == "Multiple Choice":
        question_input = format_question_multichoice(question, options)
        response = zoom_model.free_form_using_nodes(image_pil, question_input, searched_nodes)
        return response
    else:
        raise NotImplementedError
