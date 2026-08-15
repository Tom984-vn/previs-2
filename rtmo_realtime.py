import os
import time
import cv2
import torch
import numpy as np
from rtmlib import RTMO, draw_skeleton

# 1. Khởi tạo mô hình RTMO ONNX
ONNX_MODEL_PATH = '/home/muffin/Project/Previs/mmpose/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211/end2end.onnx'

model = RTMO(
    onnx_model=ONNX_MODEL_PATH,
    model_input_size=(640, 640),
    nms_thr=0.45,
    score_thr=0.4,          # Ngưỡng bbox score
    to_openpose=False,      # 17 khớp COCO
    backend='onnxruntime',  # 'onnxruntime' hoặc 'tensorrt'
    device='cuda'           # 'cuda' hoặc 'cpu'
)

print("--------- MODEL INITIALIZED ------------")
for key, item in vars(model).items():
    print(f"{key}: {item}")

# 2. Mở luồng video
source_video_path = "/home/muffin/Project/Previs/tn.mp4"
video_capture = cv2.VideoCapture(source_video_path)

window_name = "PreVis - RTMO Keypoint Stream"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

if not video_capture.isOpened():
    raise RuntimeError(f"Failed to open video source: {source_video_path}")

prev_time = time.time()

while True:
    success, frame_bgr = video_capture.read()
    
    # 1. Kiểm tra frame trước khi đưa vào model (Tránh crash khi hết video)
    if not success or frame_bgr is None:
        print("[*] Video ended or failed to read frame.")
        break

    # 2. Chạy suy luận Pose
    start_infer = time.time()
    keypoints, scores = model(frame_bgr)
    infer_ms = (time.time() - start_infer) * 1000

    # 3. Lọc bỏ các skeleton rỗng (khi trong frame không có người)
    valid_count = 0
    if len(scores) > 0:
        valid_count = sum(1 for s in scores if np.mean(s) > 0.1)

    # 4. Vẽ khung xương trực tiếp trên ảnh BGR
    img_show = frame_bgr.copy()
    if valid_count > 0:
        img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.3)

    # 5. Tính toán và hiển thị FPS thực tế
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    info_text = f"FPS: {fps:.1f} | Infer: {infer_ms:.1f}ms | Persons: {valid_count}"
    cv2.putText(img_show, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow(window_name, img_show)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
