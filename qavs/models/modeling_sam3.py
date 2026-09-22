"""SAM3 inference and semantic graph partitioning for QAVS proposals."""

import networkx as nx
import numpy as np
import torch
from skimage import graph
from skimage.measure import regionprops
from skimage.segmentation import slic
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class sam3_inference:
    def __init__(self, model_path, device="cuda:0"):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.train.transforms.basic_for_api import (
            ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI,
        )
        from sam3.eval.postprocessors import PostProcessImage

        self.device = torch.device(device)
        self.model = build_sam3_image_model(
            checkpoint_path=model_path, device=str(self.device),
        ).to(self.device).eval()
        self.processor = Sam3Processor(self.model, device=str(self.device))
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=1008, max_size=1008, square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
        )
        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=0.5,
            to_cpu=False,
            always_interpolate_masks_on_gpu=self.device.type == "cuda",
        )

    @torch.inference_mode()
    def inference(self, image, text_prompt):
        inference_state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)

        return output

    def create_empty_datapoint(self):
        """ A datapoint is a single image on which we can apply several queries at once. """
        from sam3.train.data.sam3_image_dataset import Datapoint

        return Datapoint(find_queries=[], images=[])

    def set_image(self, datapoint, pil_image):
        """ Add the image to be processed to the datapoint """
        from sam3.train.data.sam3_image_dataset import Image as SAMImage

        w, h = pil_image.size
        datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]

        return datapoint

    def add_text_prompt(self, datapoint, text_query, current_id):
        """ Add a text query to the datapoint """
        from sam3.train.data.sam3_image_dataset import FindQueryLoaded, InferenceMetadata

        assert len(datapoint.images) == 1, "please set the image first"

        w, h = datapoint.images[0].size
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[],  # unused for inference
                is_exhaustive=True,  # unused for inference
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=current_id,
                    original_image_id=current_id,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
        )
        return datapoint, current_id

    @torch.inference_mode()
    def batch_inference(self, image, text_prompts):
        from sam3.train.data.collator import collate_fn_api as collate
        from sam3.model.utils.misc import copy_data_to_device

        datapoint = self.create_empty_datapoint()
        datapoint = self.set_image(datapoint, image)
        text_id = []
        for idx, text in enumerate(text_prompts, start=1):
            datapoint, t_id = self.add_text_prompt(datapoint, text, idx)
            text_id.append(t_id)

        datapoint = self.transform(datapoint)
        batch = collate([datapoint], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, self.device, non_blocking=True)

        output = self.model(batch)
        if isinstance(output, tuple):
            output, backbone_out = output
        else:
            # Standard SAM3 keeps the already computed image features in each stage.
            backbone_out = output[0]["prev_encoder_out"]["backbone_out"]
        processed_results = self.postprocessor.process_results(output, batch.find_metadatas)

        return backbone_out, processed_results, text_id


def _calc_complexity_effective_rank(features):
    """
    Effective Rank = exp(Shannon Entropy of Singular Values)
    Args:
        features: (N_subset, C)
    Returns:
        float: 0 ~ 1
    """
    N, C = features.shape
    if N <= 1 or C == 0:
        return 0.0

    centered = features - features.mean(axis=0)

    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        s_sq = s ** 2
        total_energy = np.sum(s_sq) + 1e-10
        probs = s_sq / total_energy
        valid_probs = probs[probs > 1e-10]
        if len(valid_probs) == 0:
            return 0.0
        entropy = -np.sum(valid_probs * np.log(valid_probs))
        effective_rank = np.exp(entropy)
        max_rank = min(N, C)
        if max_rank <= 0: return 0.0

        return effective_rank

    except np.linalg.LinAlgError:
        return 0.0


class ConstrainedTreeBuilder:
    def __init__(self, feature_map, n_atoms=400, pos_weight=2.0, split_threshold=0.3, keep_threshold=0.05, lazy_base=0.4, lazy_bonus=0.6, decay_factor=0.95, use_local_normalization=True,
                 use_silhouette_score=True):
        if isinstance(feature_map, torch.Tensor):
            self.feat = feature_map.detach().cpu().numpy()
        else:
            self.feat = feature_map

        self.C, self.H, self.W = self.feat.shape
        self.n_atoms = n_atoms
        self.pos_weight = pos_weight
        self.split_threshold = split_threshold
        self.keep_threshold = keep_threshold
        self.lazy_base = lazy_base
        self.lazy_bonus = lazy_bonus
        self.decay_factor = decay_factor
        self.use_local_normalization = use_local_normalization
        self.use_silhouette_score = use_silhouette_score
        self.node_registry = {}
        self.atom_labels, self.atom_features, self.atom_bboxes, self.adj_matrix = self._generate_atoms_and_graph()

    def _generate_atoms_and_graph(self):
        feat_tr = self.feat.transpose(1, 2, 0)
        feat_min, feat_max = feat_tr.min(), feat_tr.max()
        feat_norm = (feat_tr - feat_min) / (feat_max - feat_min + 1e-6)
        #SLIC
        atom_map = slic(feat_norm, n_segments=self.n_atoms, compactness=20, start_label=0, channel_axis=2)
        unique_labels = np.unique(atom_map)
        n_actual = len(unique_labels)
        semantic_features = np.zeros((n_actual, self.C), dtype=np.float32)
        spatial_features = np.zeros((n_actual, 2), dtype=np.float32)
        props = regionprops(atom_map + 1)
        atom_bboxes = []

        for i, prop in enumerate(props):
            y_slice, x_slice = prop.slice
            mask_local = prop.image
            feat_crop = self.feat[:, y_slice, x_slice]
            semantic_features[i] = feat_crop[:, mask_local].mean(axis=1)

            cy, cx = prop.centroid
            y_encoded = (cy / self.H) * self.pos_weight
            x_encoded = (cx / self.W) * self.pos_weight
            spatial_features[i] = [y_encoded, x_encoded]

            atom_bboxes.append(prop.bbox)

        final_features = np.concatenate([semantic_features, spatial_features], axis=1)

        # --- Construct adjacency matrix ---
        rag_img = feat_tr[:, :, :3] if self.C >= 3 else feat_tr
        rag = graph.rag_mean_color(rag_img, atom_map, mode='distance')

        # Convert to Sparse Matrix (N_atoms, N_atoms)
        adj_matrix = nx.adjacency_matrix(rag)
        return atom_map, final_features, np.array(atom_bboxes), adj_matrix

    def _calc_overlap_cost(self, child_nodes):
        if len(child_nodes) < 2: return 0.0

        boxes = [n['bbox'] for n in child_nodes]
        total_area = sum([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
        total_overlap = 0.0

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ba, bb = boxes[i], boxes[j]
                iy1, ix1 = max(ba[0], bb[0]), max(ba[1], bb[1])
                iy2, ix2 = min(ba[2], bb[2]), min(ba[3], bb[3])
                inter = max(0, iy2 - iy1) * max(0, ix2 - ix1)
                total_overlap += inter

        if total_area == 0: return 0
        return total_overlap / total_area

    def _calc_region_complexity(self, atom_indices):
        if len(atom_indices) <= 1:
            return 0.0

        # semantic features (N_subset, C)
        features = self.atom_features[atom_indices, :self.C]
        norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-6
        feats_norm = features / norm
        mean_feat = feats_norm.mean(axis=0)
        mean_feat_norm = mean_feat / (np.linalg.norm(mean_feat) + 1e-6)

        # cosine similarity
        # shapes: (N, C) @ (C,) -> (N,)
        cosine_sims = feats_norm @ mean_feat_norm
        avg_sim = cosine_sims.mean()
        # Convert to complexity score
        score = max(0.0, 1.0 - avg_sim)

        avg_magnitude = norm.mean()
        score *= np.log1p(avg_magnitude)

        return score

    def build_tree(self, max_depth=3, min_splits=4, max_splits=8):
        self.node_registry = {}
        all_indices = np.arange(len(self.atom_features))
        # Calculate the complexity of the global image
        global_complexity = self._calc_region_complexity(all_indices)

        root_node = {
            "depth": 0,
            "atom_indices": all_indices,
            "bbox": (0, 0, self.H, self.W),
            "children": [],
            "split_k": 1,
            "complexity": global_complexity,
            "relative_score": 1.0,
            "node_id": "0",
            "prior_prob": 1.0, #root node 1.0
            "parent": None
        }
        self.node_registry["0"] = root_node
        self._recursive_build(root_node, max_depth, min_splits, max_splits)
        return root_node

    def _recursive_build(self, parent_node, max_depth, min_splits, max_splits):
        if parent_node["depth"] >= max_depth:
            return

        indices = parent_node["atom_indices"]
        effective_min_k = min_splits
        if len(indices) < max(min_splits, 2):
            return

        # Semantic Pruning
        if parent_node["depth"] >= 1:
            depth_decay = 0.8 ** (parent_node["depth"] - 1)
            current_threshold = self.split_threshold * depth_decay

            complexity = parent_node.get("complexity", 0)
            if complexity < current_threshold:
                return

        raw_sub_features = self.atom_features[indices]
        semantic_raw = raw_sub_features[:, :self.C]
        if self.use_local_normalization:
            scaler = StandardScaler()
            enhanced_semantic = scaler.fit_transform(semantic_raw)
        else:
            enhanced_semantic = semantic_raw

        spatial_feats = raw_sub_features[:, self.C:]
        spatial_scale = 1.0 / (parent_node["depth"] + 1)
        sub_features_for_clustering = np.concatenate([enhanced_semantic, spatial_feats * spatial_scale], axis=1)

        sub_connectivity = self.adj_matrix[indices, :][:, indices]

        best_score = -float('inf')
        best_children = []
        best_k = effective_min_k
        best_labels = None
        limit_k = min(max_splits, len(indices))
        if limit_k < effective_min_k: return

        for k in range(effective_min_k, limit_k + 1):
            try:
                model = AgglomerativeClustering(
                    n_clusters=k,
                    connectivity=sub_connectivity,
                    linkage='ward'
                )
                labels = model.fit_predict(sub_features_for_clustering)

                if self.use_silhouette_score:
                    if k > 1 and len(indices) > k:
                        sil_score = silhouette_score(enhanced_semantic, labels)
                    else:
                        sil_score = -1.0
                else:
                    sil_score = 0.0

                # Calculate BBox overlap cost
                overlap_cost = self._calc_overlap_cost_for_labels(indices, labels, k)
                # --- Comprehensive scoring formula ---
                combined_score = sil_score - (1.5 * overlap_cost)
                if combined_score > best_score:
                    best_score = combined_score
                    best_k = k
                    best_labels = labels

            except Exception as e:
                continue

        if best_score == -float('inf'): return
        best_children = []
        for lbl in range(best_k):
            child_indices_local = np.where(best_labels == lbl)[0]
            if len(child_indices_local) == 0: continue

            child_atom_indices = indices[child_indices_local]
            c_boxes = self.atom_bboxes[child_atom_indices]
            y1, x1 = np.min(c_boxes[:, 0]), np.min(c_boxes[:, 1])
            y2, x2 = np.max(c_boxes[:, 2]), np.max(c_boxes[:, 3])

            best_children.append({
                "atom_indices": child_atom_indices,
                "bbox": (y1, x1, y2, x2)
            })

        if not best_children: return

        # Calculate complexity and prune
        valid_children_data = []  # (child_data, complexity)
        complexities = []

        for child_data in best_children:
            child_complexity = self._calc_region_complexity(child_data["atom_indices"])
            if child_complexity < self.keep_threshold:
                continue

            valid_children_data.append((child_data, child_complexity))
            complexities.append(child_complexity)

        # All child nodes have been pruned
        if not valid_children_data:
            return

        # relative_score and prior_prob
        max_c = max(complexities)
        min_c = min(complexities)
        range_c = max_c - min_c

        parent_prob = parent_node.get("prior_prob", 1.0)
        for i, (child_data, child_complexity) in enumerate(valid_children_data):

            # --- Intra-Level Normalization)
            if range_c > 1e-6:
                relative_score = (child_complexity - min_c) / range_c
            else:
                relative_score = 1.0
            relative_score = 0.2 + 0.8 * relative_score

            # prior_prob=Parent_Prob * (Base + Bonus * Relative) * Decay
            estimated_transfer = self.lazy_base + self.lazy_bonus * relative_score
            current_prob = parent_prob * estimated_transfer * self.decay_factor
            current_node_id = f"{parent_node['node_id']}-{i}"

            child_node = {
                "depth": parent_node["depth"] + 1,
                "atom_indices": child_data["atom_indices"],
                "bbox": child_data["bbox"],
                "children": [],
                "split_k": best_k,
                "complexity": child_complexity,
                "relative_score": relative_score,
                "prior_prob": current_prob,
                "node_id": current_node_id,
                "parent": parent_node
            }
            parent_node["children"].append(child_node)

            self.node_registry[current_node_id] = child_node
            self._recursive_build(child_node, max_depth, min_splits, max_splits)

    def _calc_overlap_cost_for_labels(self, indices, labels, k):
        temp_children = []
        for lbl in range(k):
            child_indices_local = np.where(labels == lbl)[0]
            if len(child_indices_local) == 0: continue
            child_atom_indices = indices[child_indices_local]
            c_boxes = self.atom_bboxes[child_atom_indices]
            y1, x1 = np.min(c_boxes[:, 0]), np.min(c_boxes[:, 1])
            y2, x2 = np.max(c_boxes[:, 2]), np.max(c_boxes[:, 3])
            temp_children.append({"bbox": (y1, x1, y2, x2)})
        return self._calc_overlap_cost(temp_children)

    def get_flattened_nodes(self, tree_root):
        all_nodes = []

        def _traverse(node):
            node_info = {
                'id': node['node_id'],
                'depth': node['depth'],
                'prob': node['prior_prob'],
                'bbox': node['bbox'],
                'relative_score': node['relative_score']
            }
            all_nodes.append(node_info)
            for child in node.get('children', []):
                _traverse(child)

        _traverse(tree_root)
        return sorted(all_nodes, key=lambda x: x['prob'], reverse=True)

    def get_node_by_id(self, node_id):

        return self.node_registry.get(node_id)
