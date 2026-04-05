from flask import Flask, request, jsonify, render_template
import os
from model import load_model, load_support_set, predict

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tải model và support set 1 lần khi khởi động
print("Đang tải model...")
model = load_model()
print("Đang tải ảnh mẫu...")
support_embeddings = load_support_set(model)
print("Sẵn sàng!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_pill():
    if 'image' not in request.files:
        return jsonify({'error': 'Không có ảnh'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn ảnh'}), 400
    
    # Lưu ảnh upload
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Dự đoán
    label, score = predict(model, support_embeddings, filepath)
    
    # Tên đẹp hơn để hiển thị
    label_display = {
        'paracetamol': 'Paracetamol (Hạ sốt, giảm đau)',
        'amoxicillin': 'Amoxicillin (Kháng sinh)',
        'vitamin_c': 'Vitamin C (Tăng đề kháng)'
    }
    
    return jsonify({
        'label': label_display.get(label, label),
        'confidence': round(score * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)