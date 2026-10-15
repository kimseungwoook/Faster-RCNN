import unittest
from src.Faster_RCNN.matcher import AnchorMatcher

class TestAnchorMatcher(unittest.TestCase):
    def setUp(self):
        positive_threshold = 0.7
        negative_threshold = 0.3
        self.matcher = AnchorMatcher(
            positive_threshold, negative_threshold
        )

    def test_basic_matching(self):
        anchors =  [
            [0, 0, 100, 100],
            [200, 200, 300, 300]
        ]
        gt_boxes = [
            [0, 0, 80, 80]
        ]
        valid_flags = [True, True]

        labels, max_ious, matched_gt_indices = self.matcher.match(anchors, gt_boxes, valid_flags)

        self.assertEqual(labels, [1, 0])

        self.assertAlmostEqual(max_ious[0], 0.64)
        self.assertAlmostEqual(max_ious[1], 0.00)

        self.assertEqual(matched_gt_indices, [0, -1])

    def test_same_max_iou(self):
        anchors = [
            [0, 0, 100, 100],
            [0, 0, 100, 100]
        ]

        gt_boxes = [
            [0, 0, 80, 80]
        ]

        valid_flags = [True, True]

        labels, max_ious, matched_gt_indices = self.matcher.match(anchors, gt_boxes, valid_flags)

        self.assertEqual(labels, [1, 1])

        self.assertAlmostEqual(max_ious[0], 0.64)
        self.assertAlmostEqual(max_ious[1], 0.64)

        self.assertEqual(matched_gt_indices, [0, 0])

    def test_invalid_anchor(self):
        anchors = [
            [0, 0, 100, 100],
            [0, 0, 100, 100]
        ]

        gt_boxes = [
            [0, 0, 100, 100]
        ]

        valid_flags = [False, True]

        labels, max_ious, matched_gt_indices = self.matcher.match(anchors, gt_boxes, valid_flags)

        self.assertEqual(labels, [-1, 1])
        self.assertAlmostEqual(max_ious[0], 0.0)
        self.assertAlmostEqual(max_ious[1], 1.0)
        self.assertEqual(matched_gt_indices, [-1, 0])

    def test_multiple_gt(self):
        anchors = [
            [0, 0, 100, 100],  # Anchor 0
            [200, 200, 300, 300],  # Anchor 1
            [400, 400, 500, 500]  # Anchor 2
        ]

        gt_boxes = [
            [0, 0, 100, 100],  # GT 0
            [200, 200, 300, 300]  # GT 1
        ]

        valid_flags = [True, True, True]
        labels, max_ious, matched_gt_indices = self.matcher.match(anchors, gt_boxes, valid_flags)

        self.assertEqual(labels, [1, 1, 0])
        self.assertAlmostEqual(max_ious[0], 1.0)
        self.assertAlmostEqual(max_ious[1], 1.0)
        self.assertAlmostEqual(max_ious[2], 0.0)
        self.assertEqual(matched_gt_indices, [0, 1, -1])


    # 5. Ignore Label 검증
    def test_ignore_label(self):
        anchors = [
            [0, 0, 100, 100],
            [0, 0, 100, 200],
            [200, 200, 300, 300]
        ]
        gt_boxes = [[0, 0, 100, 100]]
        valid_flags = [True, True, True]

        labels, max_ious, matched_gt_indices = self.matcher.match(
            anchors, gt_boxes, valid_flags
        )

        self.assertEqual(labels, [1, -1, 0])
        self.assertAlmostEqual(max_ious[0], 1.0)
        self.assertAlmostEqual(max_ious[1], 0.5)
        self.assertAlmostEqual(max_ious[2], 0.0)
        self.assertEqual(matched_gt_indices, [0, 0, -1])

    # 6. 모든 IoU가 0인 경우
    def test_zero_iou(self):
        anchors = [
            [0, 0, 100, 100],
            [200, 200, 300, 300]
        ]
        gt_boxes = [[400, 400, 500, 500]]
        valid_flags = [True, True]

        labels, max_ious, matched_gt_indices = self.matcher.match(
            anchors, gt_boxes, valid_flags
        )

        self.assertEqual(labels, [0, 0])
        self.assertEqual(max_ious, [0.0, 0.0])
        self.assertEqual(matched_gt_indices, [-1, -1])

    # 7. GT Box가 없는 경우
    def test_empty_gt(self):
        anchors = [
            [0, 0, 100, 100],
            [200, 200, 300, 300]
        ]
        gt_boxes = []
        valid_flags = [True, False]

        labels, max_ious, matched_gt_indices = self.matcher.match(
            anchors, gt_boxes, valid_flags
        )

        self.assertEqual(labels, [0, -1])
        self.assertEqual(max_ious, [0.0, 0.0])
        self.assertEqual(matched_gt_indices, [-1, -1])

    # 8. Anchor가 없는 경우
    def test_empty_anchors(self):
        anchors = []
        gt_boxes = [[0, 0, 100, 100]]
        valid_flags = []

        labels, max_ious, matched_gt_indices = self.matcher.match(
            anchors, gt_boxes, valid_flags
        )

        self.assertEqual(labels, [])
        self.assertEqual(max_ious, [])
        self.assertEqual(matched_gt_indices, [])

    # 9. Threshold 경계값 검증
    def test_threshold_boundary(self):
        anchors = [
            [0, 0, 100, 100],
            [0, 0, 30, 100]
        ]
        gt_boxes = [[0, 0, 100, 100]]
        valid_flags = [True, True]

        labels, max_ious, matched_gt_indices = self.matcher.match(
            anchors, gt_boxes, valid_flags
        )

        self.assertEqual(labels, [1, 0])
        self.assertAlmostEqual(max_ious[0], 1.0)
        self.assertAlmostEqual(max_ious[1], 0.3)
        self.assertEqual(matched_gt_indices, [0, 0])

    # 10. Anchor와 Valid Flag의 길이 불일치
    def test_mismatched_lengths(self):
        anchors = [
            [0, 0, 100, 100],
            [200, 200, 300, 300]
        ]
        gt_boxes = [[0, 0, 100, 100]]
        valid_flags = [True]

        with self.assertRaises(ValueError):
            self.matcher.match(
                anchors, gt_boxes, valid_flags
            )

    def test_shared_best_anchor(self):
        anchors = [
            [0, 0, 100, 100]
        ]

        gt_boxes = [
            [0, 0, 100, 100],  # GT 0
            [50, 0, 150, 100]  # GT 1
        ]

        valid_flags = [True]

        labels, max_ious, matched_gt_indices = self.matcher.match(
            anchors, gt_boxes, valid_flags
        )

        self.assertEqual(labels, [1])
        self.assertAlmostEqual(max_ious[0], 1.0)
        self.assertEqual(matched_gt_indices, [0])

if __name__ == "__main__":
    unittest.main(verbosity=2)