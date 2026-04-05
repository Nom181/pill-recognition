# 💊 Pill Recognition - Nhận Dạng Thuốc

Ứng dụng nhận dạng thuốc bằng hình ảnh sử dụng kỹ thuật Few-shot Learning.

## 🔗 Demo trực tiếp
👉 [Bấm vào đây để dùng thử](https://huggingface.co/spaces/mtriuu/pill-recognition)

## 📌 Giới thiệu
Mỗi năm có hàng triệu ca nhầm thuốc xảy ra do nhiều viên thuốc có hình dạng tương tự nhau. Dự án này xây dựng ứng dụng web giúp nhận dạng thuốc chỉ bằng cách chụp ảnh.

## 🧠 Công nghệ sử dụng
- **Few-shot Learning** - Nhận dạng chỉ với 3-5 ảnh mẫu
- **Transfer Learning** - Dùng ResNet50 pretrained
- **Cosine Similarity** - So sánh độ tương đồng giữa ảnh
- **Flask** - Backend API
- **Gradio** - Giao diện web

## 💊 Các loại thuốc nhận dạng được
| Tên thuốc | Công dụng |
|---|---|
| Paracetamol | Hạ sốt, giảm đau |
| Amoxicillin | Kháng sinh |
| Vitamin C | Tăng đề kháng |

## ⚙️ Cách hoạt động
## 🚀 Chạy trên máy local
```bash
git clone https://github.com/Nom181/pill-recognition.git
cd pill-recognition
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## 👨‍💻 Nhóm thực hiện
Đồ án môn học - 2026
