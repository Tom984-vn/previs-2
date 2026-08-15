"""
Module 2: Action Classifier (PySKL / ST-GCN++)
Nhận tensor chuỗi thời gian của khung xương và phân loại hành vi.
"""
from typing import Tuple, List
import numpy as np
from pyskl.apis import init_recognizer, inference_recognizer


class ActionClassifier:
    def __init__(self, config_path: str, checkpoint_path: str, label_map_path: str, device: str = 'cuda:0'):
        print(f"[*] Đang tải ST-GCN++ trên thiết bị: {device}...")
        self.device = device
        self.model = init_recognizer(config_path, checkpoint_path, device=device)
        self.label_names = [x.strip() for x in open(label_map_path).readlines()]

    def predict(self, tracked_kpts: np.ndarray, tracked_scores: np.ndarray, 
                img_shape: Tuple[int, int], total_frames: int) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Thực hiện suy luận hành động từ tensor khung xương đã theo vết.
        """
        if tracked_kpts is None or tracked_scores is None:
            return "No Person", 0.0, []

        fake_anno = dict(
            frame_dir='',
            label=-1,
            img_shape=img_shape,
            original_shape=img_shape,
            start_index=0,
            modality='Pose',
            total_frames=total_frames,
            keypoint=tracked_kpts,
            keypoint_score=tracked_scores
        )

        # Suy luận nhãn hành vi
        results = inference_recognizer(self.model, fake_anno)
        
        top1_idx, top1_score = results[0]
        top1_action = self.label_names[top1_idx]
        
        # Danh sách Top-5
        top5_predictions = [(self.label_names[idx], score) for idx, score in results[:5]]
        
        return top1_action, float(top1_score), top5_predictions
