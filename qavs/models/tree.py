from PIL import Image
from typing import NamedTuple, List, Optional, Dict
import itertools


class NodeState(NamedTuple):
    original_image_pil: Image.Image
    bbox: List[int]


class Node:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
            self,
            state: Optional[NodeState],
            parent: "Optional[Node]" = None,
            fast_confidence: float = None,
            fast_confidence_details=None,
            is_terminal: bool = False
    ) -> None:

        self.id = next(Node.id_iter)
        if fast_confidence_details is None:
            fast_confidence_details = {}
        self.confidence_details = {}
        self.cum_confidences: list[float] = []
        self.fast_confidence = self.confidence = fast_confidence
        self.fast_confidence_details = fast_confidence_details
        self.answering_confidence = 0

        self.is_terminal = is_terminal
        self.state = state
        self.parent = parent
        self.children: 'Optional[list[Node]]' = []
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_root(self):
        return self.depth == 0

    def add_child(self, child: 'Node'):
        self.children.append(child)

    def save_crop(self, path):
        x, y, w, h = self.state.bbox
        crop_image = self.state.original_image_pil.crop([x, y, x + w, y + h])
        crop_image.save(path)


def is_terminal(node: Node, smallest_size: int) -> bool:
    now_w, now_h = node.state.bbox[2:]
    return max(now_w, now_h) < smallest_size








class NodeA:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.depth = 0
        self.is_leaf = True
        self.is_root = False

        self.complexity = 0.0
        self.split_k = 0

        self.answering_confidence = -1
        self.fast_confidence = None
        self.fast_confidence_details = {}

    def add_child(self, child_node):
        self.children.append(child_node)
        self.is_leaf = False

class AdaptiveImageTree:
    def __init__(self, image_pil, tree_dict_root, feature_map_shape):
        """
        Args:
            image_pil
            tree_dict_root: ConstrainedTreeBuilder
            feature_map_shape: (C, H, W) or (H, W)
        """
        self.image_pil = image_pil
        self.img_w, self.img_h = self.image_pil.size

        if len(feature_map_shape) == 3:
            _, self.feat_h, self.feat_w = feature_map_shape
        else:
            self.feat_h, self.feat_w = feature_map_shape

        # Scale Factor
        self.scale_x = self.img_w / self.feat_w
        self.scale_y = self.img_h / self.feat_h

        print(f"Mapping coords: Feat({self.feat_w}x{self.feat_h}) -> Img({self.img_w}x{self.img_h})")
        print(f"Scale Factors: X={self.scale_x:.4f}, Y={self.scale_y:.4f}")

        self.max_depth = 0
        self.root = self._convert_node_recursive(tree_dict_root, parent_node=None)
        self.root.is_root = True

    def _convert_bbox_feat_to_img_xywh(self, feat_bbox):
        """
        Input: BBox (y1, x1, y2, x2)
        Output: BBox [x, y, w, h]
        """
        y1, x1, y2, x2 = feat_bbox

        orig_x1 = int(x1 * self.scale_x)
        orig_y1 = int(y1 * self.scale_y)
        orig_x2 = int(x2 * self.scale_x)
        orig_y2 = int(y2 * self.scale_y)

        orig_x1 = max(0, orig_x1)
        orig_y1 = max(0, orig_y1)
        orig_x2 = min(self.img_w, orig_x2)
        orig_y2 = min(self.img_h, orig_y2)

        x = orig_x1
        y = orig_y1
        w = max(1, orig_x2 - orig_x1)
        h = max(1, orig_y2 - orig_y1)

        return [x, y, w, h]

    def _convert_node_recursive(self, dict_node, parent_node):
        bbox_xywh = self._convert_bbox_feat_to_img_xywh(dict_node['bbox'])

        state = NodeState(self.image_pil, bbox_xywh)
        current_node = NodeA(state, parent=parent_node)
        current_node.depth = dict_node['depth']
        current_node.complexity = dict_node.get('complexity', 0.0)
        current_node.split_k = dict_node.get('split_k', 0)
        current_node.relative_score = dict_node.get('relative_score', 0.5)
        current_node.id = dict_node.get("node_id", 'none')
        current_node.prior_prob = dict_node.get('prior_prob', 0.5)

        self.max_depth = max(self.max_depth, current_node.depth)

        for child_dict in dict_node['children']:
            child_node = self._convert_node_recursive(child_dict, parent_node=current_node)
            current_node.add_child(child_node)

        return current_node
