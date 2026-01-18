# Violence Detection in Surveillance Videos

Đồ án môn học CS231 - Nhận dạng thị giác nâng cao

## 📋 Giới thiệu

Đây là repository chứa mã nguồn triển khai hai mô hình cho bài toán phát hiện bạo lực từ video giám sát, sử dụng bộ dữ liệu **RWF-2000**:

1. **CUE-Net** (CLIP-based UniFormerV2 Enhanced Network) - Mô hình chính với độ chính xác cao
2. **FlowGate Network** - Mô hình baseline sử dụng Optical Flow attention

---

## 🏗️ Kiến trúc CUE-Net

CUE-Net được xây dựng dựa trên **UniFormerV2** với backbone **CLIP ViT-L/14-336**, kết hợp:
- **Local UniBlocks**: Trích xuất đặc trưng không gian-thời gian cục bộ
- **Global UniBlocks (MEAA)**: Multi-Head Efficient Additive Attention cho ngữ cảnh toàn cục
- **CLIP Pre-training**: Tận dụng tri thức từ mô hình vision-language quy mô lớn

### Thông số CUE-Net
| Thông số | Giá trị |
|----------|---------|
| Backbone | CLIP ViT-L/14-336 |
| Input size | 336 × 336 × 64 frames |
| Num classes | 2 (Fight/NonFight) |
| Total parameters | ~354M |
| Global UniBlocks | 4 layers |
| Hidden dim | 1024 |
| Attention heads | 16 |

---

## 🏗️ Kiến trúc FlowGate Network

FlowGate Network sử dụng kiến trúc two-stream với cơ chế attention từ Optical Flow:
- **RGB Branch**: Trích xuất đặc trưng không gian từ video gốc
- **Optical Flow Branch**: Trích xuất đặc trưng chuyển động với Sigmoid attention
- **Fusion**: Element-wise multiplication để kết hợp hai nhánh

### Thông số FlowGate Network
| Thông số | Giá trị |
|----------|---------|
| Input size | 224 × 224 × 64 frames |
| Input channels | 5 (3 RGB + 2 Optical Flow) |
| Num classes | 2 (Fight/NonFight) |
| Total parameters | ~580K |
| Conv3D Blocks | 4 blocks mỗi nhánh + 3 blocks merging |
| Regularization | L2 (0.0005) |

## 📁 Cấu trúc thư mục

```
cs231_cuenet/
├── UniFormerV2/                    # CUE-Net model code
│   ├── slowfast/
│   │   ├── config/                 # Configuration files
│   │   ├── models/                 # Model architecture
│   │   │   ├── uniformerv2.py      # Wrapper class
│   │   │   ├── uniformerv2_model.py # Core model implementation
│   │   │   └── build.py            # Model builder
│   │   ├── datasets/               # Data loading
│   │   └── utils/                  # Utilities
│   ├── exp/
│   │   └── RWF_exp/
│   │       └── config.yaml         # Training configuration
│   └── tools/
│       ├── train_net.py            # Training script
│       └── test_net.py             # Testing script
│
├── model_flowgatenetwork/          # FlowGate Network
│   ├── flowgate-train_v1.ipynb     # Training notebook v1
│   ├── flowgate-train-v2.ipynb     # Training notebook v2
│   ├── video2npy.ipynb             # Video preprocessing
│   ├── compare_v1_v2.ipynb         # Compare versions
│   ├── demo_flowgate_2.py          # Streamlit demo app
│   ├── best_model_v1.h5            # Trained weights v1
│   └── best_model_v2.h5            # Trained weights v2
│
├── data_paths/                     # Dataset split files
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── api/                            # Inference API
│   └── fight_detection_api.py
│
├── inference_single_video.py       # Single video inference (CUE-Net)
├── evaluate_validation.py          # Evaluation script
├── visualize_meaningful_v2.py      # Feature visualization (Eigen-CAM)
├── create_csv.py                   # Create dataset CSV files
└── README.md
```

## ⚙️ Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- PyTorch 2.0+ với CUDA support (cho CUE-Net)
- TensorFlow 2.x (cho FlowGate Network)
- GPU với ≥4GB VRAM (inference) hoặc ≥48GB VRAM (training CUE-Net)

### Cài đặt CUE-Net

```bash
# 1. Clone repository
git clone https://github.com/manhdungcr7/cs231_cuenet.git
cd cs231_cuenet

# 2. Cài đặt dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install fvcore iopath simplejson psutil opencv-python tensorboard
pip install timm einops decord pytorchvideo

# 3. Cài đặt slowfast
cd UniFormerV2
pip install -e .
cd ..

# 4. Tải CLIP weights (ViT-L/14-336)
# File: vit_l14_336.pth → đặt vào UniFormerV2/model_chkpts/
```

### Cài đặt FlowGate Network

```bash
# Cài đặt TensorFlow và dependencies
pip install tensorflow opencv-python streamlit

# Chạy demo Streamlit
cd model_flowgatenetwork
streamlit run demo_flowgate_2.py
```

## 🚀 Sử dụng

### Inference trên video đơn

```python
python inference_single_video.py --video path/to/video.avi
```

### Đánh giá trên tập validation

```python
python evaluate_validation.py
```

### Visualization (Eigen-CAM + Temporal Importance)

```python
python visualize_meaningful_v2.py --video path/to/video.avi
```

## 🔧 Training

Để huấn luyện mô hình từ đầu (yêu cầu GPU 48GB+):

```bash
cd UniFormerV2

# Training
python tools/train_net.py \
  --cfg exp/RWF_exp/config.yaml \
  DATA.PATH_TO_DATA_DIR /path/to/rwf2000 \
  NUM_GPUS 1 \
  TRAIN.BATCH_SIZE 2
```

### Cấu hình huấn luyện chính
- **Optimizer**: AdamW (weight decay = 0.05)
- **Learning rate**: 4e-4 với Cosine scheduler
- **Epochs**: 51
- **Batch size**: 2-4 (tùy VRAM)
- **Dropout**: 0.5

## 📚 Tài liệu tham khảo

1. [UniFormerV2: Spatiotemporal Learning by Arming Image ViTs with Video UniFormer](https://arxiv.org/abs/2211.09552)
2. [RWF-2000: An Open Large Scale Video Database for Violence Detection](https://arxiv.org/abs/1911.05913)

## 👨‍💻 Tác giả 1

- **Họ tên**: Đào Mạnh Dũng
- **MSSV**: 23520325
- **Email**: 23520325@gm.uit.edu.vn

## 👨‍💻 Tác giả 2

- **Họ tên**: Mai Xuân Tuấn
- **MSSV**: 23521714
- **Email**: 23521714@gm.uit.edu.vn

## 📄 License

MIT License
