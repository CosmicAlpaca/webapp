# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Module, Sequential, ReLU, Dropout
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  # Sử dụng loader mới
from torch_geometric.nn import GCNConv, global_mean_pool

import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from flask import Flask, request, render_template, redirect, url_for

# ==================================================================
#                       PHẦN 1: KHỞI TẠO VÀ TẢI MODEL
# ==================================================================

# --- Thiết lập các hằng số và đường dẫn ---
SAVED_MODEL_DIR = "saved_model"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# <<< THAY ĐỔI: Cập nhật tên file theo yêu cầu của bạn >>>
MODEL_FILE_PATH = os.path.join(SAVED_MODEL_DIR, "occupational_disease_pyg.pth")
COLUMNS_FILE_PATH = os.path.join(SAVED_MODEL_DIR, "LEGACY_COLUMNS.pkl")
MEAN_VALUES_FILE_PATH = os.path.join(SAVED_MODEL_DIR, "LEGACY_MEAN_VALUES.pkl")
MAX_VALUES_FILE_PATH = os.path.join(SAVED_MODEL_DIR, "LEGACY_MAX_VALUES.pkl")

DEVICE = torch.device('cpu')
IMAGE_VECTOR_DIM = 1280


# --- Sao chép lại kiến trúc model từ script huấn luyện ---
class MultiInputGCN(Module):
    def __init__(self, num_node_features, graph_embedding_dim=64, image_feature_dim=IMAGE_VECTOR_DIM):
        super(MultiInputGCN, self).__init__()
        self.gcn1 = GCNConv(num_node_features, 128)
        self.gcn2 = GCNConv(128, graph_embedding_dim)
        self.image_mlp = Sequential(Linear(image_feature_dim, 256), ReLU(), Dropout(0.5),
                                    Linear(256, graph_embedding_dim))
        self.classifier = Sequential(Linear(graph_embedding_dim * 2, 128), ReLU(), BatchNorm1d(128), Dropout(0.5),
                                     Linear(128, 1))

    def forward(self, graph_data, image_features):
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        # Xử lý trường hợp graph không có node nào
        if x.shape[0] == 0:
            graph_embedding = torch.zeros(batch.max().item() + 1, self.gcn2.out_channels, device=DEVICE)
        else:
            x = F.relu(self.gcn1(x, edge_index))
            x = self.gcn2(x, edge_index)
            graph_embedding = global_mean_pool(x, batch)

        image_embedding = self.image_mlp(image_features)
        combined_embedding = torch.cat([graph_embedding, image_embedding], dim=1)
        return self.classifier(combined_embedding)


# --- Hàm tải tất cả các "artefact" cần thiết ---
def load_artifacts():
    print(">>> Đang tải model và các tệp tiền xử lý (legacy)...")
    # Tải các đối tượng tiền xử lý
    with open(COLUMNS_FILE_PATH, 'rb') as f:
        columns = pickle.load(f)
    with open(MEAN_VALUES_FILE_PATH, 'rb') as f:
        mean_values = pickle.load(f)
    with open(MAX_VALUES_FILE_PATH, 'rb') as f:
        max_values = pickle.load(f)

    # Tải model đã huấn luyện
    model = MultiInputGCN(num_node_features=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location=DEVICE))
    model.eval()

    # Tải mô hình trích xuất đặc trưng ảnh
    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    image_feature_extractor = mobilenet_v2(weights=weights)
    image_feature_extractor.classifier = torch.nn.Identity()
    image_feature_extractor.to(DEVICE)
    image_feature_extractor.eval()
    image_transforms = weights.transforms()

    print(">>> Tải thành công!")
    return model, columns, mean_values, max_values, image_feature_extractor, image_transforms


# --- Tải các đối tượng một lần duy nhất khi ứng dụng khởi động ---
model, columns, mean_values, max_values, image_feature_extractor, image_transforms = load_artifacts()

# --- Khởi tạo ứng dụng Flask ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ==================================================================
#          PHẦN 2: CÁC HÀM XỬ LÝ THEO LOGIC CŨ (LEGACY)
# ==================================================================

def preprocess_data_legacy_style(X):
    """Hàm này chứa toàn bộ logic mã hóa từ hàm `encoded()` trong code cũ."""
    X.replace(['khong', 'co', 'ko', 'khong hut thuoc', 'co hut thuoc'], [0, 1, 0, 0, 1], inplace=True)
    X.replace(['', ' ', '.', 'khong biet'], np.nan, inplace=True)

    try:
        X['A6'].replace(['Da ket hon va dang chung song', 'Chua ket hon', 'Li di', 'Goa', 'Li than'], [2, 0, 1, 0, 1],
                        inplace=True)
        X.replace(['binh thuong', 'tot', 'rat tot', 'khong tot'], [0, 1, 2, -1], inplace=True)
        # Thêm các quy tắc replace khác từ code cũ của bạn nếu cần
    except KeyError as e:
        print(f"Cảnh báo: Cột {e} không tồn tại để thay thế giá trị.")

    # Chuyển đổi tất cả các cột sang dạng số, nếu không được thì thành NaN
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    return X


def normalize_data_legacy_style(df, mean_vals, max_vals):
    """Hàm chuẩn hóa dữ liệu theo logic cũ."""
    df_normalized = df.copy()
    for column in df_normalized.columns:
        if column in mean_vals and column in max_vals:
            denominator = max_vals[column] - mean_vals[column]
            if not pd.isna(denominator) and denominator != 0:
                df_normalized[column] = (df_normalized[column] - mean_vals[column]) / denominator
            else:
                df_normalized[column] = 0  # Hoặc một giá trị mặc định khác
    return df_normalized


def process_image(image_path):
    """Trích xuất vector đặc trưng từ một file ảnh."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = image_transforms(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = image_feature_extractor(img_tensor)
        return features
    except Exception as e:
        print(f"Lỗi xử lý ảnh: {e}")
        return torch.zeros(1, IMAGE_VECTOR_DIM).to(DEVICE)


def create_graph_from_df(df_row):
    """Tạo một đối tượng Data của PyG từ một hàng dữ liệu đã xử lý."""
    # Chỉ tạo node cho các đặc trưng có giá trị (không phải NaN)
    valid_features = df_row.dropna()
    node_features = torch.tensor(valid_features.values, dtype=torch.float).view(-1, 1)

    if node_features.shape[0] == 0:
        return Data(x=torch.empty(0, 1), edge_index=torch.empty(2, 0, dtype=torch.long))

    # Tạo cấu trúc đồ thị kết nối đầy đủ
    num_nodes = len(valid_features)
    source_nodes, target_nodes = [], []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            source_nodes.extend([i, j])
            target_nodes.extend([j, i])

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    return Data(x=node_features, edge_index=edge_index)


# ==================================================================
#                       PHẦN 3: CÁC ROUTE CỦA WEB APP
# ==================================================================

@app.route('/')
def index():
    """Hiển thị trang chủ với form nhập liệu."""
    # Lấy danh sách cột không bao gồm 'file_name' để hiển thị trên form
    form_columns = [col for col in columns if col != 'file_name']
    return render_template('index.html', columns=form_columns)


@app.route('/predict', methods=['POST'])
def predict():
    """Nhận dữ liệu, xử lý và trả về kết quả dự đoán."""
    if request.method == 'POST':
        # --- 1. Xử lý ảnh tải lên ---
        image_file = request.files.get('xray_image')
        if not image_file or image_file.filename == '':
            return "Lỗi: Vui lòng tải lên một ảnh X-quang.", 400

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)
        image_features_tensor = process_image(image_path)

        # --- 2. Xử lý dữ liệu từ form theo logic cũ ---
        form_data = request.form.to_dict()
        # Tạo DataFrame với tất cả các cột cần thiết, điền giá trị rỗng cho cột thiếu
        df = pd.DataFrame([form_data], columns=columns)

        # Áp dụng các hàm tiền xử lý legacy
        df_encoded = preprocess_data_legacy_style(df)
        df_normalized = normalize_data_legacy_style(df_encoded, mean_values, max_values)
        df_normalized.fillna(0, inplace=True)  # Điền 0 cho các giá trị còn thiếu

        # --- 3. Tạo đồ thị từ dữ liệu đã xử lý ---
        graph_data = create_graph_from_df(df_normalized.iloc[0])
        graph_batch = next(iter(DataLoader([graph_data], batch_size=1)))

        # --- 4. Thực hiện dự đoán ---
        with torch.no_grad():
            output = model(graph_batch.to(DEVICE), image_features_tensor.to(DEVICE))
            probability = torch.sigmoid(output).item()

        prediction_threshold = 0.5
        result_text = "Bệnh" if probability > prediction_threshold else "Không Bệnh"

        # --- 5. Trả về trang kết quả ---
        return render_template('result.html',
                               prediction=result_text,
                               probability=f"{probability * 100:.2f}%")


if __name__ == '__main__':
    app.run(debug=True)

