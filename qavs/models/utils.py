from PIL import Image, ImageDraw
import numpy as np
import os
import json
import math
import logging
import re
from copy import deepcopy
from typing import List
# For the visual cues like "man and his bag", we should remove the pronoun "his bag"
def include_pronouns(nlp, text):
    doc = nlp(text)
    for token in doc:
        if token.pos_ == 'PRON':
            return True
    return False


def extract_visual_objects(nlp, text):
    doc = nlp(text)
    objects = []

    stop_nouns = set([
        "color", "position", "size", "shape", "texture", "material", "what", "where", "kind", "type",
        "side", "corner", "part", "surface", "area", "region", "level", "spot", "direction",
        "picture", "image", "photo", "scene", "background", "view",
        "left", "right", "top", "bottom", "front", "back", "middle", "center",
        "how", "many", "brand", "name", "object", "thing", "map", "locations", "country"
    ])

    wh_words = set(["which", "what", "whose", "who", "that"])

    def clean_chunk(text_span):
        return re.sub(r'^(the|a|an|this|that|these|those)\s+', '', text_span, flags=re.IGNORECASE).strip()

    def get_smart_expanded_span(chunk):
        root = chunk.root
        min_i = chunk.start
        max_i = chunk.end - 1

        def traverse(node):
            nonlocal max_i
            for child in node.rights:
                if child.dep_ in ['compound', 'amod']:
                    if child.i > max_i:
                        max_i = child.i
                    traverse(child)

                elif child.dep_ in ['prep', 'acl', 'pobj', 'relcl']:
                    is_spatial_bridge = False
                    if child.dep_ == 'acl' and child.lemma_ in ['locate', 'position', 'situate', 'place']:
                        is_spatial_bridge = True

                    if child.dep_ == 'prep':
                        for grandchild in child.rights:
                            if grandchild.dep_ == 'pobj':
                                if grandchild.lemma_.lower() in stop_nouns:
                                    is_spatial_bridge = True
                                break

                    if not is_spatial_bridge:
                        if child.i > max_i:
                            max_i = child.i
                        traverse(child)

        traverse(root)
        final_span = doc[min_i: max_i + 1]
        return final_span.text

    candidates = []
    for chunk in doc.noun_chunks:
        root = chunk.root
        if chunk[0].text.lower() in wh_words:
            continue

        if root.pos_ == 'PRON' or root.lemma_.lower() in stop_nouns:
            continue

        condition = (
                root.dep_ in ["dobj", "nsubj", "nsubjpass", "ROOT", "attr", "pobj", "conj", "appos"]
        )

        if condition:
            expanded_text = get_smart_expanded_span(chunk)
            clean_text = clean_chunk(expanded_text)
            if clean_text.lower() not in stop_nouns and clean_text:
                candidates.append({
                    "text": clean_text,
                    "root_idx": root.i,
                    "length": len(clean_text)
                })

    final_candidates = []
    candidates.sort(key=lambda x: x['length'], reverse=True)

    for cand in candidates:
        current_text = cand['text']
        is_contained = False

        for kept in final_candidates:
            if current_text in kept['text'] and current_text != kept['text']:
                is_contained = True
                break

        if not is_contained:
            final_candidates.append(cand)

    final_candidates.sort(key=lambda x: x['root_idx'])

    objects = [c['text'] for c in final_candidates]
    if not objects:
        return [text.strip()]

    if len(objects) > 5:
        objects = objects[:5]

    return objects


def extract_targets(sentence: str, pattern=r"So I need the information about the following objects: (.+)"):
    match = re.search(pattern, sentence)
    if match:
        return match.group(1)
    return None


def extract_targets_SGVS(sentence: str, pattern=r":\s*(.+)$"):
    match = re.search(pattern, sentence)
    if match:
        return match.group(1)
    return None


def split_targets_sentence(targets_sentence: str, split_tag=r' and |, '):
    if targets_sentence.endswith('.'):
        targets_sentence = targets_sentence[:-1]
    targets = re.split(split_tag, targets_sentence)
    return targets




def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return deepcopy(pil_img), 0, 0
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result, 0, (width - height) // 2
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result, (height - width) // 2, 0

def bbox_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min) * (y_max - y_min)

def intersect_bbox(bboxA, bboxB, distance_buffer=50):
    bbox1 = [v - distance_buffer if i < 2 else v + distance_buffer for i, v in enumerate(bboxA)]
    bbox2 = [v - distance_buffer if i < 2 else v + distance_buffer for i, v in enumerate(bboxB)]
    """ Calculate the union of two bounding boxes. """
    x_min = max(bbox1[0], bbox2[0])
    y_min = max(bbox1[1], bbox2[1])
    x_max = min(bbox1[2], bbox2[2])
    y_max = min(bbox1[3], bbox2[3])

    if x_max > x_min and y_max > y_min:
        return (x_min, y_min, x_max, y_max)

    return None

def merge_bboxes(bbox1, bbox2):
    return (
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3])
    )

def merge_bbox_list(bboxes, threshold=0):
    """merge all cross bboxes in the List bboxes"""
    changed = True
    while changed:
        changed = False
        new_bboxes = []
        used = set()

        for i in range(len(bboxes)):
            if i in used:
                continue
            merged = False

            for j in range(len(bboxes)):
                if j in used or i == j:
                    continue
                intersection = intersect_bbox(bboxes[i], bboxes[j])
                if intersection:
                    if threshold == 0 or (threshold > 0 and (
                            bbox_area(intersection) >= threshold * bbox_area(bboxes[i]) or bbox_area(
                            intersection) >= threshold * bbox_area(bboxes[j]))):
                        new_bbox = merge_bboxes(bboxes[i], bboxes[j])
                        new_bboxes.append(new_bbox)
                        used.update([i, j])
                        changed = True
                        merged = True
                        break
            if not merged and i not in used:
                new_bboxes.append(bboxes[i])

        bboxes = new_bboxes

    return bboxes




def union_all_bboxes(bboxes):
    if len(bboxes) == 0:
        return None
    ret = bboxes[0]
    for bbox in bboxes[1:]:
        ret = merge_bboxes(ret, bbox)
    return ret


def union_blocks_independent(full_image: Image.Image, bbox1, bbox2, resized_long, backgroud_color):
    if bbox1[0] > bbox2[0]:
        bbox1, bbox2 = bbox2, bbox1
    block1 = full_image.crop(bbox1).resize((resized_long, resized_long))
    block2 = full_image.crop(bbox2).resize((resized_long, resized_long))
    background = Image.new('RGB', (2 * resized_long, 2 * resized_long), backgroud_color)
    center_y1 = (bbox1[3] + bbox1[1]) // 2
    center_y2 = (bbox2[3] + bbox2[1]) // 2
    offset_y = center_y2 - center_y1
    offset_y = np.clip(offset_y, -resized_long, resized_long)
    paste_x1 = 0
    paste_y1 = resized_long // 2 - offset_y // 2
    paste_x2 = resized_long
    paste_y2 = resized_long // 2 + offset_y // 2

    background.paste(block1, (paste_x1, paste_y1))
    background.paste(block2, (paste_x2, paste_y2))

    return background


def visualize_bbox_and_arrow(image: Image.Image, bbox, color="red", thickness=2, xyxy=False):
    """Visualizes a single bounding box on the image"""
    if not xyxy:
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
    else:
        x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - thickness)
    y1 = max(0, y1 - thickness)
    x2 = min(image.width, x2 + thickness)
    y2 = min(image.height, y2 + thickness)
    draw = ImageDraw.Draw(image)
    new_bbox = [x1, y1, x2, y2]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=thickness)
    min_distance = thickness * 6
    center_x = image.width // 2
    center_y = image.height // 2
    center_x_bbox = (x1 + x2) // 2
    center_y_bbox = (y1 + y2) // 2
    return new_bbox


def normalize_target_text(t_target):
    is_type2 = False

    if t_target:
        t_target = re.sub(r'^[\W_]+|[\W_]+$', '', t_target)
        if t_target.startswith("all "):
            is_type2 = True
            t_target = t_target[4:]
            if t_target.endswith('s'):
                t_target = t_target[:-1]

    return t_target, is_type2
