import os
import sys

# Biến môi trường chống crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import torch
import numpy as np
from pyskl.apis import init_recognizer, inference_recognizer

# 1. Cấu hình
GCN_CONFIG = '/home/muffin/Project/Previs/pyskl/configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py'
GCN_CHECKPOINT = 'http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth'
LABEL_MAP_PATH = '/home/muffin/Project/Previs/pyskl/tools/data/label_map/nturgbd_120.txt'
LABEL_MAP = 'tools/data/label_map/nturgbd_120.txt'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f"[1] Khởi tạo model ST-GCN++ trên: {DEVICE}...")
model = init_recognizer(GCN_CONFIG, GCN_CHECKPOINT, device=DEVICE)
print("[✓] Model init thành công!")

# 2. Tạo dữ liệu giả lập: 2 người, 45 frames, 17 khớp (Chuẩn COCO)
WINDOW_SIZE = 45
M, T, V = 2, WINDOW_SIZE, 17

dummy_kpt = np.random.randn(M, T, V, 2).astype(np.float32) * 100 + 200
dummy_score = np.random.rand(M, T, V).astype(np.float32)

fake_anno = dict(
    frame_dir='',
    label=-1,
    img_shape=(720, 1280),
    original_shape=(720, 1280),
    start_index=0,
    modality='Pose',
    total_frames=WINDOW_SIZE,
    keypoint=dummy_kpt,
    keypoint_score=dummy_score
)

print("[2] Chạy suy luận thử qua PySKL...")
with torch.no_grad():
    results = inference_recognizer(model, fake_anno)

print("[✓] Suy luận ST-GCN++ thành công!")
print(f"Top 1 Action Index: {results[0][0]}, Score: {results[0][1]:.4f}")

if os.path.exists(LABEL_MAP):
    label_names = [x.strip() for x in open(LABEL_MAP).readlines()]
    print(f"Top 1 Action Name: {label_names[results[0][0]]}")
