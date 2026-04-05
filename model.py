import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cài đặt model ResNet50 pretrained
def load_model():
    model = models.resnet50(pretrained=True)
    # Bỏ lớp cuối để lấy embedding
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

# Chuẩn hóa ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Trích xuất embedding từ 1 ảnh
def get_embedding(model, image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img)
    return embedding.squeeze().numpy()

# Tải toàn bộ ảnh mẫu trong support_set
def load_support_set(model, support_dir='support_set'):
    support_embeddings = {}
    for label in os.listdir(support_dir):
        label_dir = os.path.join(support_dir, label)
        if os.path.isdir(label_dir):
            embeddings = []
            for img_file in os.listdir(label_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(label_dir, img_file)
                    emb = get_embedding(model, img_path)
                    embeddings.append(emb)
            if embeddings:
                # Lấy trung bình embedding của tất cả ảnh mẫu
                support_embeddings[label] = np.mean(embeddings, axis=0)
    return support_embeddings

# Dự đoán thuốc từ ảnh query
def predict(model, support_embeddings, query_image_path):
    query_emb = get_embedding(model, query_image_path)
    
    best_label = None
    best_score = -1
    
    for label, support_emb in support_embeddings.items():
        score = cosine_similarity(
            query_emb.reshape(1, -1),
            support_emb.reshape(1, -1)
        )[0][0]
        if score > best_score:
            best_score = score
            best_label = label
    
    return best_label, float(best_score)