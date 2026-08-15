"""
Module 1: Pose Extractor & Online Tracker
Trích xuất 17 khớp chuẩn COCO từ khung hình và duy trì bộ đệm chuỗi thời gian.
"""
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment
from rtmlib import RTMO


class PoseTracker:
    def __init__(self, onnx_model_path: str, window_size: int = 45, max_tracks: int = 2, device: str = 'cuda'):
        self.window_size = window_size
        self.max_tracks = max_tracks
        self.pose_buffer = deque(maxlen=window_size)
        
        # Khởi tạo mô hình RTMO ONNX
        self.model = RTMO(
            onnx_model=onnx_model_path,
            model_input_size=(640, 640),
            nms_thr=0.45,
            score_thr=0.4,
            to_openpose=False,      # Chuẩn 17 keypoints COCO
            backend='onnxruntime',
            device=device
        )

    @staticmethod
    def _dist_ske(ske1: np.ndarray, ske2: np.ndarray) -> float:
        """Tính khoảng cách hình học giữa 2 bộ khung xương."""
        dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
        diff = np.abs(ske1[:, 2] - ske2[:, 2])
        return np.sum(np.maximum(dist, diff))

    def _track_poses(self, thre: int = 30):
        """Gán ID và chọn ra tối đa max_tracks khung xương chính theo chuẩn PySKL."""
        tracks, num_tracks = [], 0
        num_joints = 17
        
        for idx, poses in enumerate(self.pose_buffer):
            if len(poses) == 0:
                continue
            track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
            n, m = len(track_proposals), len(poses)
            scores = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    scores[i][j] = self._dist_ske(track_proposals[i]['data'][-1][1], poses[j])
            row, col = linear_sum_assignment(scores)
            for r, c in zip(row, col):
                track_proposals[r]['data'].append((idx, poses[c]))
            if m > n:
                for j in range(m):
                    if j not in col:
                        num_tracks += 1
                        new_track = {'track_id': num_tracks, 'data': [(idx, poses[j])]}
                        tracks.append(new_track)

        if len(tracks) == 0:
            return None, None

        # Sắp xếp lấy 2 đối tượng xuất hiện nhiều nhất trong cửa sổ trượt
        tracks.sort(key=lambda x: -len(x['data']))
        result = np.zeros((self.max_tracks, len(self.pose_buffer), num_joints, 3), dtype=np.float32)
        
        for i, track in enumerate(tracks[:self.max_tracks]):
            for item in track['data']:
                idx, pose = item
                result[i, idx] = pose

        return result[..., :2], result[..., 2]

    def process_frame(self, frame: np.ndarray):
        """
        Xử lý từng khung hình:
        - Trả về keypoints & scores để vẽ trực quan.
        - Trả về dữ liệu tensor đã tracking nếu bộ đệm đã đủ số frame.
        """
        keypoints, scores = self.model(frame)
        
        current_frame_poses = []
        if len(scores) > 0:
            for kpt, sc in zip(keypoints, scores):
                if np.mean(sc) > 0.2:  # Lọc khung xương rác
                    pose_3d = np.concatenate([kpt, sc[:, None]], axis=-1)
                    current_frame_poses.append(pose_3d)

        self.pose_buffer.append(current_frame_poses)

        # Kiểm tra nếu bộ đệm đã tích lũy đủ khung hình
        is_ready = (len(self.pose_buffer) == self.window_size)
        tracked_kpts, tracked_scores = None, None
        
        if is_ready:
            tracked_kpts, tracked_scores = self._track_poses()

        return keypoints, scores, is_ready, tracked_kpts, tracked_scores
