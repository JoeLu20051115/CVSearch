import hashlib
import json
import math
import os
import time

import torch
import transformers
import PIL
from torch.nn import CrossEntropyLoss
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from .tree import Node, NodeA
from .utils import *
from cvsearch.evidence_gap.types import EvidenceRequirement, EvidenceSupportResult
BOX_COLOR = "red"

ANSWER_FREE_SUPPORT_PROMPT_VERSION = "qwen_answer_free_evidence_support_v1"
ANSWER_FREE_SUPPORT_PROCESSOR_MODE = "single_rendered_view_chat_left_padding_final_yes_no_logits"
_ANSWER_FREE_SUPPORT_TEMPLATE = (
    "Assess only whether the displayed visual observation provides sufficient visible "
    "evidence for the question and every ordered evidence requirement. Do not answer "
    "the question and do not infer missing evidence.\n"
    "Question: {question}\n"
    "Ordered evidence requirements:\n{requirements}\n"
    "Does this observation provide sufficient visible evidence for the complete "
    "requirement set? Answer Yes or No."
)
_ANSWER_FREE_SUPPORT_TEMPLATE_SHA256 = hashlib.sha256(
    _ANSWER_FREE_SUPPORT_TEMPLATE.encode("utf-8")
).hexdigest()
_FROZEN_YES_TOKEN_ID = 9454
_FROZEN_NO_TOKEN_ID = 2753


def _qualified_class(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _support_view_sha256(image):
    payload = image.mode.encode("utf-8") + b"\x00"
    payload += f"{image.width}x{image.height}".encode("ascii") + b"\x00"
    payload += image.tobytes()
    return hashlib.sha256(payload).hexdigest()

class ModelQwenVL:
    def __init__(self, model_path: str, device: str = "cuda:0", torch_dtype=torch.bfloat16, **kwargs) -> None:
        self.model_checkpoint = os.fspath(model_path)
        self.device = device
        self.dtype = torch_dtype
        load_kwargs = {}
        load_in_8bit = kwargs.get("load_in_8bit", False)
        if load_in_8bit:
            device_map = "auto"
        else:
            device_map = self.device
        self.use_flash_attn = True
        actual_attn_implementation = "flash_attention_2" if self.use_flash_attn else None
        print(f"Flash Attention Enabled: {self.use_flash_attn} (Backend: {actual_attn_implementation})")
        ###Qwen2.5-VL
        if 'Qwen2.5' in model_path:
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.tokenizer = self.processor.tokenizer
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=self.dtype,
                attn_implementation=actual_attn_implementation,
                device_map=device_map,
                **load_kwargs
            )
        ###Qwen3-VL
        else:
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.tokenizer = self.processor.tokenizer
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=self.dtype,
                attn_implementation=actual_attn_implementation,
                device_map=device_map,
                **load_kwargs
            )
        max_pixels = kwargs.get("max_pixels", 12845056) ### 4194304 for Qwen2.5/3-VL-32B on A100 to avoid OOM
        min_pixels = kwargs.get("min_pixels", 3136)
        print("max_pixels:", max_pixels)
        print("min_pixels:", min_pixels)
        self.processor.image_processor.max_pixels = max_pixels
        self.processor.image_processor.min_pixels = min_pixels

        self.bias_value = kwargs.get("bias_value", 0.6)
        print("bias_value:", self.bias_value)

        # We hard code the input size to 448x448 for QwenVL
        self.input_size = (448, 448)
        print("input size:", self.input_size)

        self.view_size = 336 ### crop size
        self.scale_size = 672

        self.background_color = tuple(int(x * 255) for x in self.processor.image_processor.image_mean)
        print("background color:", self.background_color)

        self.patch_scale = kwargs.get("patch_scale", None)
        print("patch scale:", self.patch_scale)

        self.init_prompts()
        self.init_index_yes_no()

    def init_index_yes_no(self):
        print("Yes:", self.tokenizer("Yes").input_ids)
        print("No:", self.tokenizer("No").input_ids)
        if len(self.tokenizer("Yes").input_ids) == 1 and len(self.tokenizer("No").input_ids) == 1:
            self.index_yes = self.tokenizer("Yes").input_ids[0]
            self.index_no = self.tokenizer("No").input_ids[0]
        else:
            assert len(self.tokenizer("Yes").input_ids) == 2 and len(self.tokenizer("No").input_ids) == 2
            self.index_yes = self.tokenizer("Yes").input_ids[1]
            self.index_no = self.tokenizer("No").input_ids[1]
        print("index_yes:", self.index_yes)
        print("index_no:", self.index_no)

    def _support_token_provenance(self):
        yes_tokens = tuple(int(value) for value in self.tokenizer("Yes").input_ids)
        no_tokens = tuple(int(value) for value in self.tokenizer("No").input_ids)
        if len(yes_tokens) not in {1, 2} or len(no_tokens) not in {1, 2}:
            raise ValueError("Yes/No tokenization must contain one selected token")
        yes_id = yes_tokens[-1]
        no_id = no_tokens[-1]
        if (
            yes_id == no_id
            or yes_id != self.index_yes
            or no_id != self.index_no
            or yes_id != _FROZEN_YES_TOKEN_ID
            or no_id != _FROZEN_NO_TOKEN_ID
        ):
            raise ValueError("Yes/No tokenization does not match the frozen checkpoint")
        return yes_tokens, no_tokens, yes_id, no_id

    def _support_processor_fingerprint(self):
        image_processor = self.processor.image_processor
        tokenizer = self.tokenizer
        config = self.model.config
        checkpoint = getattr(config, "_name_or_path", None) or getattr(
            tokenizer, "name_or_path", self.model_checkpoint
        )
        architectures = getattr(config, "architectures", None)
        if architectures is not None:
            architectures = list(architectures)
        payload = {
            "checkpoint_resolved": str(checkpoint),
            "checkpoint_snapshot": os.path.basename(str(checkpoint).rstrip(os.sep)),
            "model_config_class": _qualified_class(config),
            "model_type": getattr(config, "model_type", None),
            "architectures": architectures,
            "wrapper_dtype": str(self.dtype),
            "model_dtype": str(getattr(self.model, "dtype", None)),
            "wrapper_device": str(self.device),
            "model_device": str(getattr(self.model, "device", None)),
            "attention_implementation": getattr(config, "_attn_implementation", None),
            "wrapper_flash_attention": bool(self.use_flash_attn),
            "processor_class": _qualified_class(self.processor),
            "image_processor_class": _qualified_class(image_processor),
            "tokenizer_class": _qualified_class(tokenizer),
            "image_processor_is_fast": getattr(image_processor, "is_fast", None),
            "image_processor_min_pixels": getattr(image_processor, "min_pixels", None),
            "image_processor_max_pixels": getattr(image_processor, "max_pixels", None),
            "image_processor_patch_size": getattr(image_processor, "patch_size", None),
            "image_processor_temporal_patch_size": getattr(
                image_processor, "temporal_patch_size", None
            ),
            "image_processor_merge_size": getattr(image_processor, "merge_size", None),
            "tokenizer_padding_side_default": getattr(tokenizer, "padding_side", None),
            "effective_padding_side": "left",
            "runtime_transformers_version": transformers.__version__,
            "runtime_torch_version": torch.__version__,
            "runtime_pillow_version": PIL.__version__,
            "checkpoint_transformers_version": getattr(config, "transformers_version", None),
        }
        json.dumps(payload, sort_keys=True, allow_nan=False)
        return payload, str(checkpoint)

    def _prepare_evidence_support(
        self, question: str, requirements: tuple[EvidenceRequirement, ...],
    ):
        if not isinstance(requirements, tuple) or not all(
            isinstance(item, EvidenceRequirement) for item in requirements
        ):
            raise TypeError("requirements must be an immutable EvidenceRequirement tuple")
        if not requirements:
            return None
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty q0 string")
        yes_tokens, no_tokens, yes_id, no_id = self._support_token_provenance()
        fingerprint, checkpoint = self._support_processor_fingerprint()
        requirement_lines = "\n".join(
            f"{index + 1}. [{item.requirement_id}] {item.text}"
            for index, item in enumerate(requirements)
        )
        support_text = _ANSWER_FREE_SUPPORT_TEMPLATE.format(
            question=question, requirements=requirement_lines,
        )
        chat_prompt = self.get_prompt_from_qs("<image>\n" + support_text)
        return {
            "yes_tokens": yes_tokens,
            "no_tokens": no_tokens,
            "yes_id": yes_id,
            "no_id": no_id,
            "fingerprint": fingerprint,
            "checkpoint": checkpoint,
            "chat_prompt": chat_prompt,
            "prompt_sha256": hashlib.sha256(chat_prompt.encode("utf-8")).hexdigest(),
            "prompt_template_sha256": _ANSWER_FREE_SUPPORT_TEMPLATE_SHA256,
            "prompt_version": ANSWER_FREE_SUPPORT_PROMPT_VERSION,
            "processor_mode": ANSWER_FREE_SUPPORT_PROCESSOR_MODE,
        }

    @torch.no_grad()
    def evidence_support(
        self, *, question: str, requirements: tuple[EvidenceRequirement, ...],
        rendered_observation: Image.Image, observation_identity: str,
    ) -> EvidenceSupportResult | None:
        """Score one answer-free requirement set with one final-token Yes/No pair."""
        prepared = self._prepare_evidence_support(question, requirements)
        if prepared is None:
            return None
        if not isinstance(rendered_observation, Image.Image):
            raise TypeError("rendered_observation must be a PIL image")
        if not isinstance(observation_identity, str) or not observation_identity:
            raise ValueError("observation_identity must be canonical strict JSON")
        try:
            identity = json.loads(
                observation_identity,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite constant {value}")
                ),
            )
            canonical_identity = json.dumps(
                identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("observation_identity must be canonical strict JSON") from error
        if canonical_identity != observation_identity:
            raise ValueError("observation_identity must be canonical strict JSON")

        started = time.perf_counter()
        model_inputs = self.processor(
            text=[prepared["chat_prompt"]], images=[rendered_observation], return_tensors="pt",
            padding=True, padding_side="left",
        ).to(self.device)
        outputs = self.model(**model_inputs)
        try:
            pair = outputs.logits[0, -1, [prepared["yes_id"], prepared["no_id"]]]
        except (AttributeError, IndexError, TypeError) as error:
            raise ValueError("support logits must contain the final Yes/No pair") from error
        if pair.numel() != 2:
            raise ValueError("support logits must contain exactly two values")
        probabilities = torch.softmax(pair, dim=-1)
        yes_logit, no_logit = (float(value) for value in pair.detach().cpu())
        p_yes, p_no = (float(value) for value in probabilities.detach().cpu())
        if not all(math.isfinite(value) for value in (yes_logit, no_logit, p_yes, p_no)):
            raise ValueError("support logits and probabilities must be finite")
        if abs((p_yes + p_no) - 1.0) > 1e-5:
            raise ValueError("support probabilities must be normalized")
        elapsed = time.perf_counter() - started
        processor_json = json.dumps(
            prepared["fingerprint"], sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        requirement_set_id = EvidenceSupportResult.requirement_set_id_for(requirements)
        return EvidenceSupportResult(
            requirements=requirements,
            requirement_set_id=requirement_set_id,
            observation_identity=observation_identity,
            prompt_version=prepared["prompt_version"],
            prompt_template_sha256=prepared["prompt_template_sha256"],
            prompt_sha256=prepared["prompt_sha256"],
            processor_mode=prepared["processor_mode"],
            processor_fingerprint_json=processor_json,
            checkpoint=prepared["checkpoint"],
            yes_tokenization=prepared["yes_tokens"],
            no_tokenization=prepared["no_tokens"],
            yes_token_id=prepared["yes_id"],
            no_token_id=prepared["no_id"],
            p_yes_transform="softmax([yes_logit,no_logit],dim=-1)[0]",
            yes_logit=yes_logit,
            no_logit=no_logit,
            p_yes=p_yes,
            p_no=p_no,
            support_avg=p_yes,
            support_min=p_yes,
            observation_mode=rendered_observation.mode,
            observation_size=(rendered_observation.width, rendered_observation.height),
            view_sha256=_support_view_sha256(rendered_observation),
            elapsed_seconds=elapsed,
        )

    def get_confidence_weight(self, node: Node, max_depth: int):
        coeff = (1 - self.bias_value) / (max_depth ** 2)
        return coeff * (node.depth ** 2) + self.bias_value

    def init_prompts(self):
        self.prompts = {
            "search": {
                "pre_information_image2": f"<image>\nThis is the main image, and the section enclosed by the {BOX_COLOR} rectangle is the focus region.\n<image>\nThis is the zoomed-in view of the focus region.\n",
                "pre_information_image1": "<image>\n",
                "latent_prompt": "According to your common sense knowledge and the content of image, is it possible to find a {} by further zooming in the image? Answer Yes or No.",
                "existence_prompt": "Is there a '{}' in the image? Answer Yes or No.",
                "answering_prompt": "Question: '{}'\nCould you answer the question based on the available visual information? Answer Yes or No.",
            }
        }

    @torch.no_grad()
    def generate_visual_cues_using_ic(self, ic_examples, question: str, split_tag=r' and |, '):
        ic_question_template = ic_examples["question_template"]
        ic_question_list = ic_examples["question_list"]
        ic_response_list = ic_examples["response_list"]

        message = []
        for q, a in zip(ic_question_list, ic_response_list):
            message.extend([
                {"role": "user", "content": [{"type": "text", "text": ic_question_template.format(q)}]},
                {"role": "assistant", "content": [{"type": "text", "text": a}]}
            ])
        message.extend([
            {"role": "user", "content": [{"type": "text", "text": ic_question_template.format(question)}]},
        ])
        texts = [self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
        model_inputs = self.processor(
            text=texts,
            images=None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False)
        model_inputs = model_inputs.to(self.device)
        generated_ids = self.model.generate(**model_inputs, use_cache=True, max_new_tokens=256, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]

        targets_sentence = extract_targets_SGVS(response)
        if targets_sentence is not None:
            targets = split_targets_sentence(targets_sentence, split_tag)
        else:
            targets = []

        return targets

    def filter_visual_cues(self, image_pil: Image.Image, targets, root_node, decomposed_question_template,answering_confidence_threshold_upper, show_value=False):
        assert root_node.is_root
        ret = []
        if len(targets) > 1:
            for target in targets:
                if target.startswith("all "):
                    ret.append(target)
                    continue
                inputs = {
                    "node": root_node,
                    "image_pil": image_pil,
                    "confidence_type": "answering",
                    "input_ele": decomposed_question_template.format(target)
                }
                inputs['root_anyres'] = False
                conf_root = self.get_confidence_value(**inputs)
                if show_value:
                    print(target, conf_root)
                if conf_root < answering_confidence_threshold_upper:
                    ret.append(target)
        else:
            ret = targets[:]
        return ret

    def get_prompt_from_qs(self, qs, response=None, show_prompt=False):
        message = []
        message.append({"role": "user", "content": []})
        while "<image>\n" in qs:
            index = qs.find("<image>\n")
            if index == 0:
                message[0]["content"].append({"type": "image"})
                qs = qs[len("<image>\n"):]
            else:
                message[0]["content"].append({"type": "text", "text": qs[:index]})
                qs = qs[index:]
        if len(qs) > 0:
            message[0]["content"].append({"type": "text", "text": qs})
        if response is not None:
            message.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        texts = [self.processor.apply_chat_template(message, tokenize=False,add_generation_prompt=True if response is None else False)]

        return texts[0]

    def resize_image(self, image_pil):
        img = deepcopy(image_pil)
        input_size = self.input_size[0]
        if img.width > img.height:
            img = img.resize((input_size, int(img.height * input_size / img.width)))
        else:
            img = img.resize((int(img.width * input_size / img.height), input_size))
        return img

    def resize_image_seg(self, image_pil, max_size):
        img = deepcopy(image_pil)
        input_size = max_size
        if img.width > img.height:
            img = img.resize((input_size, int(img.height * input_size / img.width)))
        else:
            img = img.resize((int(img.width * input_size / img.height), input_size))
        return img

    def get_prompt_tag(self, image_list):
        if len(image_list) == 1:
            prompt_tag = "global"
        elif len(image_list) == 2:
            prompt_tag = "zoom"
        else:
            raise ValueError
        return prompt_tag

    def save_crop(self, image_pil, node, image_path):
        resized_bboxes = self.get_patch(node.state.bbox, image_pil.width, image_pil.height,patch_size=self.input_size[0], patch_scale=self.patch_scale)

        cropped_image = image_pil.crop(resized_bboxes)
        cropped_image.save(image_path)

    def is_root_only(self, nodes: List[Node]):
        return (len(nodes) == 1 and nodes[0].is_root)

    def include_root(self, nodes: List[Node]):
        return any(node.is_root for node in nodes)

    def get_bbox_in_square_image(self, bbox, left, top):
        x1, y1, x2, y2 = bbox
        return [x1 + left, y1 + top, x2 + left, y2 + top]

    def draw_bbox_arrow_in_square_image(self, square_image, resized_bbox, color):
        thickness = square_image.width // 120
        new_bbox = visualize_bbox_and_arrow(square_image, resized_bbox, color, thickness, xyxy=True)
        return new_bbox

    def get_original_image_with_color_bbox(self, square_image, left, top, image_pil):
        image_width, image_height = image_pil.size
        crop_coords = [left, top, left + image_width, top + image_height]
        cropped_image = square_image.crop(crop_coords)
        return cropped_image

    def zoom_in(self, croped_view):
        c_width = croped_view.width
        c_height = croped_view.height
        resize_ratio = min(self.scale_size * 1.0 / c_width, self.scale_size * 1.0 / c_height)
        resize_ratio = max(1.0, resize_ratio)
        zoomed_view = croped_view.resize((int(c_width * resize_ratio), int(c_height * resize_ratio)))

        return zoomed_view

    def get_patch(self, bbox, image_width, image_height, patch_size, patch_scale=None):
        object_width = int(np.ceil(bbox[2]))
        object_height = int(np.ceil(bbox[3]))

        object_center_x = int(bbox[0] + bbox[2] / 2)
        object_center_y = int(bbox[1] + bbox[3] / 2)

        patch_width = max(object_width, patch_size)
        patch_height = max(object_height, patch_size)
        if patch_scale is not None:
            patch_width = int(patch_width * patch_scale)
            patch_height = int(patch_height * patch_scale)

        left = max(0, object_center_x - patch_width // 2)
        right = min(left + patch_width, image_width)

        top = max(0, object_center_y - patch_height // 2)
        bottom = min(top + patch_height, image_height)

        return [left, top, right, bottom]

    def process_nodes_to_image_list(self, nodes, image_pil, root_anyres=True):
        square_image, left, top = expand2square(image_pil, self.background_color)
        if self.is_root_only(nodes):
            return [deepcopy(image_pil)] if root_anyres else [self.resize_image(image_pil)]
        if len(nodes) == 0 or self.include_root(nodes):
            return [deepcopy(image_pil)]

        resized_bboxes = []
        for node in nodes:
            source = getattr(node, 'search_source', 'fine')
            if source == 'fast':
                # Fast: view_size=1/3, none scale
                current_patch_size = self.view_size // 3
                current_patch_scale = None
            else:
                # Fine
                current_patch_size = self.view_size
                current_patch_scale = self.patch_scale

            patch_bbox = self.get_patch(
                bbox=node.state.bbox,
                image_width=image_pil.width,
                image_height=image_pil.height,
                patch_size=current_patch_size,
                patch_scale=current_patch_scale
            )
            resized_bboxes.append(patch_bbox)

        resized_bboxes = merge_bbox_list(resized_bboxes, threshold=0)

        full_color_bboxes = []
        for i in range(len(resized_bboxes)):
            resized_bbox = self.get_bbox_in_square_image(resized_bboxes[i], left, top)
            color_bbox = self.draw_bbox_arrow_in_square_image(square_image, resized_bbox, BOX_COLOR)
            full_color_bboxes.append(color_bbox)

        union_color_bboxes = union_all_bboxes(full_color_bboxes)
        if union_color_bboxes is None:
            return [square_image]
        croped_view = square_image.crop(union_color_bboxes)
        zoomed_view = self.zoom_in(croped_view)
        original_image_with_color_bbox = self.get_original_image_with_color_bbox(square_image, left, top, image_pil)

        return [self.resize_image(original_image_with_color_bbox), croped_view, zoomed_view]

    @torch.no_grad()
    def get_confidence_value(self, node: List[NodeA], image_pil: Image.Image, confidence_type: str, input_ele,root_anyres=True):
        assert confidence_type in ['existence', 'latent', 'answering']
        image_list = self.process_nodes_to_image_list(node, image_pil, root_anyres=root_anyres)
        if len(image_list) > 1:
            raw_image_resize, croped_view, zoom_view = image_list
            image_input = [zoom_view]
        else:
            image_input = image_list
        prompt_tag = "search"
        if len(image_input) > 1:
            pre_information = "pre_information_image2"
        else:
            pre_information = "pre_information_image1"

        qs = self.prompts[prompt_tag][pre_information] + self.prompts[prompt_tag][f"{confidence_type}_prompt"].format(input_ele)
        prompt = self.get_prompt_from_qs(qs)

        model_inputs = self.processor(
            text=[prompt],
            images=image_input,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        model_inputs = model_inputs.to(self.device)
        outputs = self.model(**model_inputs)

        return self._cal_confidence(outputs)

    @torch.no_grad()
    def _cal_confidence(self, outputs):
        logits_yesno = outputs.logits[0, -1, [self.index_yes, self.index_no]]
        confidence = torch.softmax(logits_yesno, dim=-1)[0]
        confidence = 2 * (confidence.item() - 0.5)  # [-1, 1]
        return confidence

    @torch.inference_mode()
    def free_form_using_nodes(self, image_pil, question, searched_nodes: List[NodeA], return_zoomed_view=False):
        image_list = self.process_nodes_to_image_list(searched_nodes, image_pil)
        if len(image_list) > 1:
            raw_image_resize, croped_view, zoom_view = image_list
            image_input = [zoom_view]
        else:
            image_input = image_list

        prompt_tag = "search"
        if len(image_input) > 1:
            pre_information = "pre_information_image2"
        else:
            pre_information = "pre_information_image1"

        qs = self.prompts[prompt_tag][pre_information] + question
        prompt = self.get_prompt_from_qs(qs)
        model_inputs = self.processor(
            text=[prompt],
            images=image_input,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        model_inputs = model_inputs.to(self.device)
        generated_ids = self.model.generate(**model_inputs, use_cache=True, max_new_tokens=256, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]

        if return_zoomed_view:
            response = {'text': outputs, 'output_image': image_list[1] if len(image_list) > 1 else image_list[0]}
            return response

        return outputs

    @torch.inference_mode()
    def multiple_choices_inference(self, image_pil, question, options, searched_nodes: List[NodeA] = None):
        choice, _ = self.multiple_choices_with_losses(image_pil, question, options, searched_nodes)
        return choice

    @torch.inference_mode()
    def multiple_choices_with_losses(self, image_pil, question, options, searched_nodes: List[NodeA] = None):
        if not options:
            raise ValueError("options must be nonempty")
        image_list = self.process_nodes_to_image_list(searched_nodes, image_pil)
        if len(image_list) > 1:
            raw_image_resize, croped_view, zoom_view = image_list
            image_input = [zoom_view]
        else:
            image_input = image_list

        prompt_tag = "search"
        if len(image_input) > 1:
            pre_information = "pre_information_image2"
        else:
            pre_information = "pre_information_image1"

        qs = self.prompts[prompt_tag][pre_information] + question
        prompt = self.get_prompt_from_qs(qs)

        model_inputs = self.processor(
            text=[prompt],
            images=image_input,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        model_inputs = model_inputs.to(self.device)
        question_input_ids = model_inputs.input_ids
        len_question_input_ids = question_input_ids.shape[1]
        output_question = self.model(**model_inputs)
        len_question_logits = output_question.logits.shape[1]

        loss_list = []
        for option in options:
            full_prompt = self.get_prompt_from_qs(qs, option)

            full_model_inputs = self.processor(
                text=[full_prompt],
                images=image_input,
                return_tensors="pt",
                padding=True,
                padding_side="left",
            )
            full_model_inputs = full_model_inputs.to(self.device)
            full_input_ids = full_model_inputs.input_ids

            output_option = self.model(**full_model_inputs)
            logits = output_option.logits[:, len_question_logits - 1:-1]

            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, logits.size(-1))
            labels = full_input_ids[:, len_question_input_ids:].view(-1)

            loss = loss_fct(logits, labels)
            loss_list.append(loss)

        loss_values = [float(loss.detach().cpu()) for loss in loss_list]
        if not all(math.isfinite(loss) for loss in loss_values):
            raise ValueError("option losses must be finite")
        return min(range(len(loss_values)), key=loss_values.__getitem__), loss_values
