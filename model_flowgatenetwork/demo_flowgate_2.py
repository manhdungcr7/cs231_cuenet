import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# ===================== GPU CONFIGURATION (Hidden) =====================
# TensorFlow: Force CPU
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

# ===================== PATHS CONFIG =====================
try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except:
    MOVIEPY_AVAILABLE = False

# Try import PyTorch for CUENet
try:
    import torch
    TORCH_AVAILABLE = True
    
    # Quietly configure PyTorch to use GPU if available (no print to console)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.set_default_device('cuda')
except:
    TORCH_AVAILABLE = False

# ===================== PATHS CONFIG =====================
BASE_DIR = Path(__file__).parent.parent  # E:\fight_detection_cuenet
UNIFORMER_PATH = BASE_DIR / "UniFormerV2"
CUENET_PATH = BASE_DIR / "CUENet" / "Enhanced_Uniformer_V2"
CUENET_CONFIG_PATH = UNIFORMER_PATH / "exp" / "RWF_exp" / "config.yaml"
CUENET_CHECKPOINT_PATH = BASE_DIR / "models" / "cuenet_rwf2000_epoch51.pyth"

# FlowGate model paths (trong cùng thư mục với file này)
FLOWGATE_DIR = Path(__file__).parent

# Add UniFormerV2 to path if exists
if UNIFORMER_PATH.exists():
    sys.path.insert(0, str(UNIFORMER_PATH))
if CUENET_PATH.exists():
    sys.path.insert(0, str(CUENET_PATH))

# ===================== CẤU HÌNH TRANG =====================
st.set_page_config(
    page_title="Violence Detection - FlowGate & CUENet",
    page_icon="🚨",
    layout="wide"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .fight-result {
        background-color: #DC3545;
        border: 3px solid #A02834;
        color: white;
    }
    .fight-result h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .fight-result h2 {
        color: #FFE5E5;
    }
    .nonfight-result {
        background-color: #28A745;
        border: 3px solid #1E7E34;
        color: white;
    }
    .nonfight-result h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .nonfight-result h2 {
        color: #E5F5E5;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        border-radius: 5px;
        background-color: #F0F2F6;
    }
</style>
""", unsafe_allow_html=True)

# ===================== HELPER FUNCTIONS =====================

# extract the rgb images 
def get_rgb(input_x):
    rgb = input_x[...,:3]
    return rgb

# extract the optical flows
def get_opt(input_x):
    opt= input_x[...,3:5]
    return opt

@st.cache_resource
def load_violence_model(model_path):
    """Load FlowGate model với caching để tránh load lại nhiều lần"""
    my_custom_objects = {
        'get_rgb': get_rgb,
        'get_opt': get_opt,
        'tf': tf
    }
    model = load_model(model_path, custom_objects=my_custom_objects)
    return model


# ===================== CUENET MODEL CLASS =====================
class CUENetModel:
    """CUENet Model wrapper (PyTorch) with GPU support"""
    
    def __init__(self):
        self.model = None
        self.cfg = None
        self.device = None
        self.is_loaded = False
    
    def load(self, checkpoint_path, config_path):
        """Load CUENet model từ checkpoint với GPU support"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch không được cài đặt!")
        
        try:
            from slowfast.config.defaults import get_cfg
            from slowfast.models import build_model
            
            # Load config
            self.cfg = get_cfg()
            self.cfg.merge_from_file(str(config_path))
            
            # Force GPU usage if available (quietly)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0
            self.cfg.TEST.CHECKPOINT_FILE_PATH = str(checkpoint_path)
            self.cfg.TRAIN.ENABLE = False
            self.cfg.TEST.ENABLE = True
            
            # Build model
            self.model = build_model(self.cfg)
            self.model.eval()
            
            # Load checkpoint
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
            
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'], strict=False)
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            # Move to GPU
            self.model = self.model.to(self.device)
            
            # Don't use FP16 - causes issues with LayerNorm
            
            self.is_loaded = True
            # print(f"✅ CUENet model loaded on {self.device}")  # Hidden output
            return True
            
        except Exception as e:
            self.is_loaded = False
            raise e
    
    def short_side_scale(self, frames, size):
        """Scale frames giữ aspect ratio"""
        height, width = frames[0].shape[:2]
        if height < width:
            new_height = size
            new_width = int(width * (size / height))
        else:
            new_width = size
            new_height = int(height * (size / width))
        
        scaled_frames = []
        for frame in frames:
            scaled = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            scaled_frames.append(scaled)
        
        return scaled_frames
    
    def center_crop(self, frames, crop_size):
        """Center crop frames"""
        height, width = frames[0].shape[:2]
        
        y_offset = (height - crop_size) // 2
        x_offset = (width - crop_size) // 2
        
        cropped_frames = []
        for frame in frames:
            cropped = frame[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]
            cropped_frames.append(cropped)
        
        return cropped_frames
    
    def preprocess_video(self, video_path):
        """Preprocess video cho CUENet model"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None, 0
        
        # Uniform temporal sampling
        num_frames = self.cfg.DATA.NUM_FRAMES if self.cfg else 64
        img_size = self.cfg.DATA.TEST_CROP_SIZE if self.cfg else 336
        
        if total_frames < num_frames:
            indices = list(range(total_frames)) + [total_frames-1] * (num_frames - total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        
        # Load frames
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return None, 0
        
        # Spatial preprocessing
        frames = self.short_side_scale(frames, img_size)
        frames = self.center_crop(frames, img_size)
        
        # Convert to tensor
        frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
        video_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2)  # (C, T, H, W)
        
        # Move to device first before normalization (quietly use GPU if available)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        video_tensor = video_tensor.to(device)
        
        # Normalize (mean and std on same device as video_tensor)
        mean = torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32, device=device).view(3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32, device=device).view(3, 1, 1, 1)
        video_tensor = (video_tensor - mean) / std
        
        return video_tensor.unsqueeze(0), total_frames  # (1, C, T, H, W)
    
    def predict(self, video_path):
        """Dự đoán video có đánh nhau hay không với GPU acceleration"""
        if not self.is_loaded:
            raise RuntimeError("Model chưa được load!")
        
        # Preprocess
        video_tensor, num_frames = self.preprocess_video(video_path)
        if video_tensor is None:
            raise ValueError("Không thể load hoặc xử lý video")
        
        # Move to device (no FP16 conversion)
        video_tensor = video_tensor.to(self.device)
        
        # Predict with GPU
        with torch.inference_mode():
            preds = self.model([video_tensor])
            probs = torch.nn.functional.softmax(preds, dim=1)
        
        probs = probs.squeeze().cpu().float().numpy()
        pred_label = int(np.argmax(probs))
        
        class_names = ['NonFight', 'Fight']
        
        return {
            "prediction": class_names[pred_label],
            "confidence": float(probs[pred_label] * 100),
            "prob_nonfight": float(probs[0]),
            "prob_fight": float(probs[1]),
            "num_frames": num_frames
        }


@st.cache_resource
def load_cuenet_model():
    """Load CUENet model với caching và GPU support"""
    model = CUENetModel()
    model.load(str(CUENET_CHECKPOINT_PATH), str(CUENET_CONFIG_PATH))
    return model

def getOpticalFlow(video):
    """Calculate dense optical flow of input video"""
    gray_video = []
    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
        gray_video.append(np.reshape(img, (224, 224, 1)))

    flows = []
    for i in range(0, len(video)-1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_video[i], gray_video[i+1], None, 
            0.5, 3, 15, 3, 5, 1.2, 
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        flows.append(flow)
    
    flows.append(np.zeros((224, 224, 2)))
    return np.array(flows, dtype=np.float32)

def uniform_sampling(video, target_frames=64):
    """Sampling đều về 64 frames"""
    len_frames = len(video)
    interval = int(np.ceil(len_frames / target_frames))
    
    sampled_video = []
    for i in range(0, len_frames, interval):
        sampled_video.append(video[i])
    
    num_pad = target_frames - len(sampled_video)
    if num_pad > 0:
        padding = [video[-1]] * num_pad
        sampled_video += padding
    
    return np.array(sampled_video[:target_frames], dtype=np.float32)

def normalize(data):
    """Normalize theo mean và std"""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def preprocess_video(video_path, target_frames=64):
    """Xử lý video thành format (64, 224, 224, 5)"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    frames = np.array(frames, dtype=np.uint8)
    
    if len(frames) == 0:
        raise ValueError("Không đọc được frame nào từ video!")
    
    optical_flows = getOpticalFlow(frames)
    frames = frames.astype(np.float32)
    data = np.concatenate([frames, optical_flows], axis=-1)
    data = uniform_sampling(data, target_frames=target_frames)
    
    data[..., :3] = normalize(data[..., :3])
    data[..., 3:] = normalize(data[..., 3:])
    
    return data, len(frames)

def predict_video(video_path, model):
    """Dự đoán video có violence hay không"""
    data, num_frames = preprocess_video(video_path)
    data = np.expand_dims(data, axis=0)
    
    predictions = model.predict(data, verbose=0)
    
    prob_fight = predictions[0][0]
    prob_nonfight = predictions[0][1]
    
    class_names = ['Fight', 'NonFight']
    predicted_class = np.argmax(predictions[0])
    class_name = class_names[predicted_class]
    confidence = predictions[0][predicted_class] * 100
    
    return class_name, confidence, prob_fight, prob_nonfight, num_frames

# ===================== MAIN APP =====================

def main():
    # Header
    st.markdown('<h1 class="main-title">🚨 Violence Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Hệ thống phát hiện bạo lực trong video - Hỗ trợ FlowGate (TensorFlow) & CUENet (PyTorch)</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Cấu hình")
        
        # Model Type selection
        st.subheader("Chọn Model")
        model_type = st.radio(
            "Loại model:",
            ["FlowGate (TensorFlow)", "CUENet (PyTorch)"],
            help="FlowGate: CNN + Optical Flow | CUENet: UniFormerV2 + CLIP"
        )
        
        # Sub-model selection based on type
        if model_type == "FlowGate (TensorFlow)":
            st.info("FlowGate sử dụng RGB + Optical Flow")
            model_options = {
                'FlowGate V1': str(FLOWGATE_DIR / 'best_model_v1.h5'),
                'FlowGate V2': str(FLOWGATE_DIR / 'best_model_v2.h5'),
            }
            selected_model_name = st.selectbox("Chọn phiên bản:", list(model_options.keys()))
            model_path = model_options[selected_model_name]
        else:
            st.info("CUENet: Accuracy 90.75% | ROC-AUC 0.969")
            selected_model_name = "CUENet (UniFormerV2)"
            
            # Check if CUENet files exist
            cuenet_available = CUENET_CHECKPOINT_PATH.exists() and CUENET_CONFIG_PATH.exists()
            if not cuenet_available:
                st.error("❌ CUENet model files không tìm thấy!")
                st.write(f"Checkpoint: {CUENET_CHECKPOINT_PATH}")
                st.write(f"Config: {CUENET_CONFIG_PATH}")
            
            if not TORCH_AVAILABLE:
                st.error("❌ PyTorch chưa được cài đặt!")
        
        st.divider()
        
        # Model Info
        st.header("📊 Model Info")
        if model_type == "FlowGate (TensorFlow)":
            st.markdown("""
            **FlowGate Network**
            - Framework: TensorFlow/Keras
            - Input: RGB + Optical Flow (5 channels)
            - Resolution: 224×224
            - Frames: 64
            - Architecture: Dual-stream Conv3D + Gating
            """)
        else:
            st.markdown("""
            **CUENet**
            - Framework: PyTorch
            - Backbone: UniFormerV2 + CLIP
            - Resolution: 336×336
            - Frames: 64
            - Accuracy: **90.75%**
            - ROC-AUC: **0.969**
            """)
        
        st.divider()
        
        st.header("📝 Hướng dẫn")
        st.write("""
        1. Chọn model (FlowGate hoặc CUENet)
        2. Upload video (.mp4, .avi, .mov)
        3. Nhấn "Phân tích video"
        4. Xem kết quả dự đoán
        """)
    
    # Main content
    st.header("Upload Video")
    
    uploaded_file = st.file_uploader(
        "Chọn video cần phân tích",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload video để phát hiện bạo lực"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily first
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_video_path = tmp_file.name
        
        # Display video
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎬 Video đã upload")
            
            # Kiểm tra nếu là file AVI - convert sang MP4 bằng moviepy
            if uploaded_file.name.lower().endswith('.avi'):
                tmp_mp4_path = tmp_video_path.replace('.avi', '_preview.mp4')
                video_converted = False
                
                if MOVIEPY_AVAILABLE:
                    st.info("🔄 Đang chuyển đổi AVI sang MP4 bằng moviepy...")
                    try:
                        # Dùng moviepy convert (API cho moviepy 2.x)
                        clip = VideoFileClip(tmp_video_path)
                        clip.write_videofile(
                            tmp_mp4_path,
                            codec='libx264',
                            audio_codec='aac'
                        )
                        clip.close()
                        
                        # Hiển thị video đã convert
                        if os.path.exists(tmp_mp4_path) and os.path.getsize(tmp_mp4_path) > 0:
                            with open(tmp_mp4_path, 'rb') as video_file:
                                st.video(video_file.read())
                            os.unlink(tmp_mp4_path)
                            video_converted = True
                            st.success("✅ Chuyển đổi thành công!")
                        
                    except Exception as e:
                        st.error(f"⚠️ Lỗi moviepy: {str(e)}")
                else:
                    st.warning("⚠️ Moviepy không khả dụng. Cài đặt: pip install moviepy")
                
                # Nếu không convert được, hiển thị frame đầu tiên
                if not video_converted:
                    st.info("📸 Hiển thị frame đầu tiên (video vẫn được phân tích):")
                    cap = cv2.VideoCapture(tmp_video_path)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, use_container_width=True)
                        
                        # Thêm thông tin video
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        st.caption(f"📊 {frame_count} frames | {fps:.1f} FPS | {duration:.1f}s")
                    cap.release()
            else:
                # MP4, MOV - hiển thị trực tiếp
                st.video(uploaded_file)
        
        with col2:
            st.subheader("📋 Thông tin file")
            st.write(f"**Tên file:** {uploaded_file.name}")
            st.write(f"**Kích thước:** {uploaded_file.size / (1024*1024):.2f} MB")
            st.write(f"**Định dạng:** {uploaded_file.type}")
            st.write(f"**Model:** {selected_model_name}")
        
        # Predict button
        if st.button("Phân tích video", type="primary", use_container_width=True):
            try:
                if model_type == "FlowGate (TensorFlow)":
                    # ===================== FLOWGATE PREDICTION =====================
                    with st.spinner(f"⏳ Đang load model {selected_model_name}..."):
                        model = load_violence_model(model_path)
                    
                    with st.spinner("⏳ Đang xử lý video và tính toán optical flow..."):
                        progress_bar = st.progress(0)
                        progress_bar.progress(30)
                        
                        class_name, confidence, prob_fight, prob_nonfight, num_frames = predict_video(tmp_video_path, model)
                        
                        progress_bar.progress(100)
                
                else:
                    # ===================== CUENET PREDICTION =====================
                    if not TORCH_AVAILABLE:
                        st.error("❌ PyTorch chưa được cài đặt! Chạy: pip install torch torchvision")
                        return
                    
                    if not CUENET_CHECKPOINT_PATH.exists():
                        st.error(f"❌ Checkpoint không tìm thấy: {CUENET_CHECKPOINT_PATH}")
                        return
                    
                    with st.spinner(f"⏳ Đang load CUENet model (có thể mất 30-60s)..."):
                        cuenet_model = load_cuenet_model()
                    
                    with st.spinner("⏳ Đang xử lý video với CUENet..."):
                        progress_bar = st.progress(0)
                        progress_bar.progress(30)
                        
                        result = cuenet_model.predict(tmp_video_path)
                        
                        class_name = result["prediction"]
                        confidence = result["confidence"]
                        prob_fight = result["prob_fight"]
                        prob_nonfight = result["prob_nonfight"]
                        num_frames = result["num_frames"]
                        
                        progress_bar.progress(100)
                
                st.success("✅ Phân tích hoàn tất!")
                
                # Display results
                st.header("📊 Kết quả phân tích")
                
                # Model badge
                if model_type == "FlowGate (TensorFlow)":
                    st.info(f"Model: **{selected_model_name}** (FlowGate - TensorFlow)")
                else:
                    st.info(f"Model: **CUENet** (UniFormerV2 + CLIP - PyTorch)")
                
                # Result box
                result_class = "fight-result" if class_name == "Fight" else "nonfight-result"
                result_emoji = "⚠️" if class_name == "Fight" else "✅"
                result_text = "BẠO LỰC PHÁT HIỆN" if class_name == "Fight" else "KHÔNG CÓ BẠO LỰC"
                
                st.markdown(f"""
                <div class="result-box {result_class}">
                    <h1>{result_emoji} {result_text}</h1>
                    <h2>Độ tin cậy: {confidence:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("Xác suất Fight", f"{prob_fight*100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("Xác suất NonFight", f"{prob_nonfight*100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("Số frames của video", f"{num_frames}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Probability chart
                st.subheader("📈 Biểu đồ xác suất")
                
                fig, ax = plt.subplots(figsize=(8, 5))
                classes = ['Fight', 'NonFight']
                probabilities = [prob_fight*100, prob_nonfight*100]
                colors = ['#DC3545', '#28A745']  # Đỏ cho Fight, Xanh lá cho NonFight
                
                bars = ax.bar(classes, probabilities, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
                
                # Thêm giá trị lên đỉnh cột
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{prob:.2f}%',
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                ax.set_ylabel('Xác suất (%)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Phân loại', fontsize=12, fontweight='bold')
                ax.set_title(f'Phân bố xác suất dự đoán ({selected_model_name})', fontsize=14, fontweight='bold', pad=20)
                ax.set_ylim(0, 105)  # Cố định trục Y từ 0-105%
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Tăng kích thước font cho trục
                ax.tick_params(axis='both', labelsize=11)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"❌ Lỗi trong quá trình xử lý: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            
            finally:
                # Clean up temp file
                if os.path.exists(tmp_video_path):
                    os.unlink(tmp_video_path)
    
    else:
        # Empty state
        st.info("👆 Hãy upload video để bắt đầu phân tích")
        
        # Model comparison section
        with st.expander("📊 So sánh FlowGate vs CUENet"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                ### FlowGate Network
                **Ưu điểm:**
                - Nhẹ, chạy được trên CPU
                - Tốc độ nhanh
                - Sử dụng optical flow để capture motion
                """)
            
            with col2:
                st.markdown(f"""
                ### CUENet               
                **Ưu điểm:**
                - Accuracy cao nhất
                - State-of-the-art architecture
                - Pretrained trên dữ liệu lớn
                """)
        
        # Example section
        with st.expander("📚 Xem ví dụ"):
            st.write("""
            **Ví dụ về các loại video:**
            
            🥊 **Fight (Bạo lực):**
            - Đánh nhau, ẩu đả
            - Bạo lực thể chất
            - Xô xát tập thể
            
            ✅ **NonFight (Không bạo lực):**
            - Hoạt động thể thao bình thường
            - Múa, biểu diễn
            - Hoạt động hàng ngày
            """)

if __name__ == "__main__":
    main()
