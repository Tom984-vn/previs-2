import onnxruntime as ort

print("1. Danh sách Provider tìm thấy:", ort.get_available_providers())

# Ép duy nhất CUDA, nếu thiếu thư viện sẽ ném lỗi chi tiết
try:
    sess = ort.InferenceSession(
        '/home/muffin/Project/Previs/mmpose/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211/end2end.onnx',
        providers=['CUDAExecutionProvider']
    )
    print("2. Kết nối thành công GPU:", sess.get_providers())
except Exception as e:
    print("\n[!] LỖI GÂY TỤT VỀ CPU:", e)
