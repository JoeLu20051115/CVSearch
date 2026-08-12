import unittest

from cvsearch.eval.treebench_support_proxy import treebench_geometry_support_label


class TreeBenchSupportProxyTest(unittest.TestCase):
    def test_perception_requires_each_target_with_minimum_detail(self):
        annotation = {
            "category": "Perception/Attributes",
            "target_instances": "[[10, 10, 20, 20], [40, 40, 50, 50]]",
        }
        self.assertEqual(treebench_geometry_support_label(
            annotation, ((5, 5, 25, 25), (35, 35, 55, 55)),
        ), 1)
        self.assertEqual(treebench_geometry_support_label(
            annotation, ((0, 0, 30, 30),),
        ), 0)
        self.assertEqual(treebench_geometry_support_label(
            annotation, ((0, 0, 2000, 2000),),
        ), 0)

    def test_reasoning_requires_one_crop_covering_all_target_centers(self):
        annotation = {
            "category": "Reasoning/Ordering",
            "target_instances": "[[10, 10, 20, 20], [40, 40, 50, 50]]",
        }
        self.assertEqual(treebench_geometry_support_label(
            annotation, ((0, 0, 60, 60),),
        ), 1)
        self.assertEqual(treebench_geometry_support_label(
            annotation, ((5, 5, 25, 25), (35, 35, 55, 55)),
        ), 0)

    def test_proxy_rejects_malformed_or_unsupported_evaluator_metadata(self):
        for annotation in (
            {"category": "Poison", "target_instances": "[[1,2,3,4]]"},
            {"category": "Perception/OCR", "target_instances": "not-json"},
            {"category": "Perception/OCR", "target_instances": "[[3,2,1,4]]"},
        ):
            with self.subTest(annotation=annotation), self.assertRaises((TypeError, ValueError)):
                treebench_geometry_support_label(annotation, ((0, 0, 10, 10),))


if __name__ == "__main__":
    unittest.main()
