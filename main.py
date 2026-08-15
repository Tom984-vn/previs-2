import os
import sys

# ==============================================================================
# 1. BIẾN MÔI TRƯỜNG BẮT BUỘC (ĐẶT TRÊN CÙNG TRƯỚC TẤT CẢ IMPORT)
# ==============================================================================
# Bỏ qua xung đột OpenMP / MKL giữa PyTorch và ONNX Runtime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Fix lỗi sập giao diện Qt / Wayland trên Fedora
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Nạp động thư viện CUDA của NVIDIA
conda_prefix = sys.prefix
site_nvidia = os.path.join(conda_prefix, f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia')
env_paths = [os.path.join(conda_prefix, 'lib')]
if os.path.exists(site_nvidia):
    for root, dirs, files in os.walk(site_nvidia):
        if os.path.basename(root) == 'lib':
            env_paths.append(root)
os.environ['LD_LIBRARY_PATH'] = ':'.join(env_paths) + ':' + os.environ.get('LD_LIBRARY_PATH', '')

# ==============================================================================
# 2. IMPORT THƯ VIỆN
# ==============================================================================
import time
import cv2
import torch
import numpy as np
from rtmlib import draw_skeleton

from rtmo_realtime import PoseTracker
from action_classifier import ActionClassifier

# ==============================================================================
# 3. CẤU HÌNH ĐƯỜNG DẪN
# ==============================================================================
RTMO_ONNX_PATH = '/home/muffin/Project/Previs/mmpose/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211/end2end.onnx'
GCN_CONFIG = '/home/muffin/Project/Previs/pyskl/configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py'
GCN_CHECKPOINT = 'http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth'
LABEL_MAP_PATH = '/home/muffin/Project/Previs/pyskl/tools/data/label_map/nturgbd_120.txt'

VIDEO_SOURCE = '/home/muffin/Project/Previs/tn.mp4'
# Chuyển cả ST-GCN++ lên GPU nếu có CUDA để tránh lệch ngữ cảnh
GCN_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

WINDOW_SIZE = 45

INFER_INTERVAL = 5


def main():
    print("[*] Khởi tạo Tracker & Classifier...")
    tracker = PoseTracker(
        onnx_model_path=RTMO_ONNX_PATH, 
        window_size=WINDOW_SIZE, 
        max_tracks=2, 
        device='cuda'
    )
    
    classifier = ActionClassifier(
        config_path=GCN_CONFIG, 
        checkpoint_path=GCN_CHECKPOINT, 
        label_map_path=LABEL_MAP_PATH, 
        device='cpu'
    )

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở nguồn video: {VIDEO_SOURCE}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_shape = (h, w)

    frame_idx = 0
    prev_time = time.time()
    current_action = "Analyzing..."
    current_confidence = 0.0

    print("[*] Pipeline sẵn sàng. Bắt đầu xử lý luồng...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            print("[*] Hết video hoặc mất luồng stream.")
            break

        frame_idx += 1

        # A. Trích xuất Pose & Duy trì Tracking
        keypoints, scores, is_ready, tracked_kpts, tracked_scores = tracker.process_frame(frame)

        # B. Suy luận Hành vi định kỳ
        if is_ready and (frame_idx % INFER_INTERVAL == 0):
            action, conf, _ = classifier.predict(
                tracked_kpts=tracked_kpts,
                tracked_scores=tracked_scores,
                img_shape=img_shape,
                total_frames=WINDOW_SIZE
            )
            current_action = action
            current_confidence = conf

        # C. Hiển thị Trực quan
        img_show = frame.copy()
        if len(scores) > 0:
            # Đảm bảo mảng numpy liên tục trong bộ nhớ trước khi vẽ
            img_show = draw_skeleton(np.ascontiguousarray(img_show), keypoints, scores, kpt_thr=0.3)

        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0
        prev_time = cur_time

        # Bảng HUD
        cv2.rectangle(img_show, (10, 10), (520, 100), (0, 0, 0), -1)
        cv2.putText(img_show, f"FPS: {fps:.1f} | Frame: {frame_idx} | Buffer: {len(tracker.pose_buffer)}/{WINDOW_SIZE}", 
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        alert_color = (0, 0, 255) if current_confidence > 0.4 else (0, 255, 0)
        cv2.putText(img_show, f"Action: {current_action}", 
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
        cv2.putText(img_show, f"Confidence: {current_confidence * 100:.1f}%", 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 1)

        cv2.imshow("PreVis Modular Pipeline", img_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
