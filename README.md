# CUE-Net: Violence Detection in Surveillance Videos

Đồ án môn học CS231 - Nhận dạng thị giác nâng cao

## 📋 Giới thiệu

Đây là repository chứa mã nguồn triển khai mô hình **CUE-Net** (CLIP-based UniFormerV2 Enhanced Network) cho bài toán phát hiện bạo lực từ video giám sát, sử dụng bộ dữ liệu **RWF-2000**.

## 🏗️ Kiến trúc mô hình

CUE-Net được xây dựng dựa trên **UniFormerV2** với backbone **CLIP ViT-L/14-336**, kết hợp:
- **Local UniBlocks**: Trích xuất đặc trưng không gian-thời gian cục bộ
- **Global UniBlocks (MEAA)**: Multi-Head Efficient Additive Attention cho ngữ cảnh toàn cục
- **CLIP Pre-training**: Tận dụng tri thức từ mô hình vision-language quy mô lớn

### Thông số mô hình
| Thông số | Giá trị |
|----------|---------|
| Backbone | CLIP ViT-L/14-336 |
| Input size | 336 × 336 × 64 frames |
| Num classes | 2 (Fight/NonFight) |
| Total parameters | ~354M |
| Global UniBlocks | 4 layers |
| Hidden dim | 1024 |
| Attention heads | 16 |

## 📁 Cấu trúc thư mục

```
cs231_cuenet/
├── UniFormerV2/                    # Core model code
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
├── data_paths/                     # Dataset split files
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── models/                         # Trained checkpoints
│   └── cuenet_rwf2000_epoch51.pyth
│
├── api/                            # Inference API
│   └── fight_detection_api.py
│
├── visualizations/                 # Output visualizations
│
├── inference_single_video.py       # Single video inference
├── evaluate_validation.py          # Evaluation script
├── visualize_meaningful_v2.py      # Feature visualization (Eigen-CAM)
├── create_csv.py                   # Create dataset CSV files
└── README.md
```

## ⚙️ Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- PyTorch 2.0+ với CUDA support
- GPU với ≥4GB VRAM (inference) hoặc ≥48GB VRAM (training)

### Các bước cài đặt

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

## 👨‍💻 Tác giả

- **Họ tên**: Đào Mạnh Dũng
- **MSSV**: 23520325
- **Email**: 23520325@gm.uit.edu.vn

## 📄 License

MIT License
