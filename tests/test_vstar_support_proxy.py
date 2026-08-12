import unittest

from cvsearch.eval.vstar_support_proxy import (
    deterministic_calibration_member,
    vstar_geometry_support_label,
    visible_crops_from_audit,
)


class VstarSupportProxyTest(unittest.TestCase):
    def test_direct_attribute_requires_target_coverage_and_apparent_detail(self):
        row = {
            "test_type": "direct_attributes",
            "bbox": [[490, 490, 5, 5]],
        }
        self.assertEqual(
            vstar_geometry_support_label(row, ((0, 0, 1000, 1000),)), 0,
        )
        self.assertEqual(
            vstar_geometry_support_label(row, ((450, 450, 550, 550),)), 1,
        )

    def test_relation_requires_one_native_crop_to_contain_all_targets(self):
        row = {
            "test_type": "relative_position",
            "bbox": [[10, 10, 5, 5], [90, 90, 5, 5]],
        }
        self.assertEqual(vstar_geometry_support_label(
            row, ((0, 0, 30, 30), (70, 70, 110, 110)),
        ), 0)
        self.assertEqual(vstar_geometry_support_label(
            row, ((0, 0, 110, 110),),
        ), 1)

    def test_extracts_exact_native_crops_for_zoom_and_expand(self):
        zoom = {"coordinate_mapping": [{
            "current_crop_xyxy": [0, 0, 20, 20],
            "candidate_crop_xyxy": [5, 5, 15, 15],
        }]}
        expand = {
            "focus_merge_identity": {"merged_crop_xyxy": [[0, 0, 20, 20]]},
            "context_merge_identity": {"merged_crop_xyxy": [[20, 0, 40, 20]]},
        }
        self.assertEqual(
            visible_crops_from_audit("ZOOM", zoom),
            (((0.0, 0.0, 20.0, 20.0),), ((5.0, 5.0, 15.0, 15.0),)),
        )
        self.assertEqual(
            visible_crops_from_audit("EXPAND", expand),
            (
                ((0.0, 0.0, 20.0, 20.0),),
                ((0.0, 0.0, 20.0, 20.0), (20.0, 0.0, 40.0, 20.0)),
            ),
        )

    def test_calibration_partition_is_source_only_and_deterministic(self):
        first = deterministic_calibration_member("relative_position/image.jpg")
        self.assertEqual(
            first,
            deterministic_calibration_member("relative_position/image.jpg"),
        )
        with self.assertRaises(TypeError):
            deterministic_calibration_member(3)


if __name__ == "__main__":
    unittest.main()
