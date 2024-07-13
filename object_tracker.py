# File: object_tracker.py

from collections import defaultdict
from scipy.spatial import distance as dist
import numpy as np

class ObjectTracker:
    def __init__(self, max_disappear=10):
        self.objects = {}        # Dictionary lưu trữ các đối tượng hiện tại (object_id: (label, bbox))
        self.disappeared = {}    # Dictionary lưu trữ số lần không xuất hiện của mỗi đối tượng (object_id: số lần không xuất hiện)
        self.max_disappear = max_disappear  # Số frame tối đa để coi đối tượng đã biến mất
        self.next_object_id = 0  # ID tiếp theo cho đối tượng mới

        self.object_counts = defaultdict(int)  # Dictionary đếm số lượng đối tượng theo nhãn

    def register(self, label, bbox):
        """
        Đăng ký một đối tượng mới.

        Parameters:
        - label: str - nhãn của đối tượng
        - bbox: tuple - tọa độ bounding box (x1, y1, x2, y2)
        """
        self.objects[self.next_object_id] = (label, bbox)
        self.disappeared[self.next_object_id] = 0
        self.object_counts[label] += 1
        self.next_object_id += 1

    def deregister(self, object_id):
        """
        Hủy đăng ký một đối tượng.

        Parameters:
        - object_id: int - ID của đối tượng cần hủy đăng ký
        """
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, labels, bboxes):
        """
        Cập nhật vị trí của các đối tượng dựa trên các nhãn và bounding boxes nhận diện được.

        Parameters:
        - labels: list - danh sách nhãn tương ứng với các đối tượng
        - bboxes: list - danh sách bounding boxes tương ứng với các đối tượng

        Returns:
        - objects: dict - dictionary chứa thông tin cập nhật của các đối tượng (object_id: (label, bbox))
        """
        if len(bboxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappear:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for label, bbox in zip(labels, bboxes):
                self.register(label, bbox)
        else:
            object_ids = list(self.objects.keys())
            object_bboxes = [self.objects[object_id][1] for object_id in object_ids]

            D = dist.cdist(np.array(object_bboxes), np.array(bboxes))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = (labels[col], bboxes[col])
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappear:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(labels[col], bboxes[col])

        return self.objects

    def get_object_counts(self):
        """
        Trả về số lượng các đối tượng đã được nhận diện.

        Returns:
        - object_counts: dict - dictionary chứa số lượng đối tượng theo nhãn
        """
        return self.object_counts

    def get_tracked_objects(self):
        """
        Trả về các đối tượng hiện đang được theo dõi.

        Returns:
        - objects: dict - dictionary chứa thông tin các đối tượng đang theo dõi (object_id: (label, bbox))
        """
        return self.objects
