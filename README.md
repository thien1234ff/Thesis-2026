# Thesis-2026
# Face Recognition — Training & Evaluation (FP32 vs Quantized)

Repo này chứa pipeline **training** và **eval** cho mô hình nhận diện khuôn mặt (iResNet18/iResNet50/MobileFaceNet) theo hướng **so sánh FP32 và mô hình quantized** (ví dụ Q6 = weight_bit=6, act_bit=6).

> Lưu ý: Scripts trong thư mục `training/` và `eval/` hiện đang mang tính “research/experiment” (nhiều đường dẫn dataset/weights đang là path cụ thể trên môi trường Kaggle). Phần README này mô tả **cấu trúc, mục tiêu, và cách chạy theo logic của code**, để bạn dễ chuyển sang local/hệ của bạn.

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Các mô hình trong repo](#các-mô-hình-trong-repo)
- [Training](#training)
  - [PTQ (Post-Training Quantization)](#ptq-post-training-quantization)
  - [Warming-up / Cool-down (QAT-style, quantization-aware fine-tuning)](#warming-up--cool-down-qat-style-quantization-aware-fine-tuning)
- [Evaluation](#evaluation)
  - [Verification chuẩn (LFW/CFP_FP/AGEDB_30/CALFW/CPLFW)](#verification-chuẩn-lfwcfp_fpagedb_30calfwcplfw)
  - [IJB-C evaluation](#ijb-c-evaluation)
- [Face recognition realtime (demo)](#face-recognition-realtime-demo)
- [Cấu hình & threshold](#cấu-hình--threshold)
- [Yêu cầu môi trường](#yêu-cầu-môi-trường)

---

## Tổng quan

Pipeline được chia thành 3 khối chính:

1. **Training / fine-tuning / PTQ** để tạo model FP32 và model quantized (ví dụ Q6).
2. **Evaluation** trên các benchmark verification (LFW/CFP_FP/…) và IJB-C.
3. **Realtime demo** (script `main.py`) để chạy webcam và nhận diện dựa trên embedding + database đã lưu.

---

## Các mô hình trong repo

Repo có nhiều script cho các backbone, trong đó nổi bật:

- **iResNet18 / iResNet50**
- **MobileFaceNet**

Trong `training/` và `eval/` bạn thấy các file pattern tương tự:

- `training/ptq/*.py`
- `training/w0-3p/*.py`, `training/w-3p/*.py`
- `eval/eval_iresnet18.py`, `eval/eval_iresnet50.py`, `eval/eval_mobilefacenet.py`
- `eval/eval_IJB-C.py`

---

## Training

### Cấu trúc thư mục training

- `training/ptq/`
  - Script **PTQ**: quantize post-training, chạy **calibration** để đóng băng observer (tùy cách implement trong `quantize_model`).
- `training/w0-3p/`, `training/w-3p/`
  - Script kiểu **QAT-style fine-tuning** với các pha:
    - **Warm-up** (trước khi freeze observer)
    - **Main/Cool-down** (sau khi freeze observer)

Mục tiêu chung của training là tạo checkpoint dạng `*.pth` (ví dụ `iresnet18_q6_epoch{epoch}.pth`) để đưa vào `eval/`.

---

### PTQ (Post-Training Quantization)

**File ví dụ:** `training/ptq/iresnet18.py`

Quy trình trong code:

1. **Load backbone FP32** (`iResNet18(num_features=512)`), nạp weight từ `FP32_PATH`.
2. Gọi `quantize_model(model_fp32, weight_bit=6, act_bit=6)` để tạo **student quantized**.
3. Chuẩn bị **calibration dataset** bằng `datasets.ImageFolder(DATA_ROOT, transform=...)`.
4. Chạy qua một số batch calibration (`CALIB_BATCHES`) ở chế độ `no_grad()` để **tích lũy thống kê** cho observer.
5. Gọi `freeze_observer(model_q)` để **disable observer** (để lượng tử hóa “ổn định”).
6. `torch.save(model_q.state_dict(), SAVE_PATH)`.

**Kết quả:** file quantized checkpoint (ví dụ `iresnet18_q6_ptq.pth`) sẵn để benchmark verification.

---

### Warming-up / Cool-down (QAT-style, quantization-aware fine-tuning)

**File ví dụ:** `training/w0-3p/iresnet18.py` và `training/w-3p/iresnet18.py`

Quy trình phổ biến:

1. Load **teacher FP32** (backbone FP32), `requires_grad=False`.
2. Load **student quantized** từ `quantize_model(student_fp32, weight_bit=6, act_bit=6)`.
3. Huấn luyện student bằng **distillation kiểu embedding regression**:
   - Lấy embedding của teacher và student, normalize về đơn vị.
   - Dùng `MSELoss` giữa embedding student và teacher.
4. Dùng optimizer (thường SGD + momentum, weight decay) và scheduler theo nhiều pha.
5. Một điểm quan trọng là **freeze observer** theo phase:
   - Ở giai đoạn warm-up: cho phép observer cập nhật.
   - Sau khi qua một mốc epoch: gọi `freeze_observer(student)` và chuyển sang cool-down.
6. Log metrics (loss/acc/tar theo cách code định nghĩa) và lưu checkpoint mỗi epoch:
   - `iresnet18_q{WEIGHT_BIT}_epoch{epoch}.pth`.

**Ghi chú về metrics trong code:**
- Trong nhiều script, `compute_metrics()` dùng cosine similarity threshold để ước lượng accuracy/TAR (cách định nghĩa mang tính surrogate cho training; benchmark cuối cùng vẫn nên dùng `eval/`).

---

## Evaluation

### Cấu trúc evaluation

- Script evaluation thường:
  1. Load model FP32 hoặc quantized.
  2. Load dataset theo format benchmark (LFW/CFP_FP/… dạng `*.bin`, IJB-C có meta + loose_crop).
  3. Extract embedding (thường có **flip test**).
  4. Tính verification metrics:
     - **10-fold accuracy**
     - **TAR@FAR** (FAR=1e-4, …)

---

### Verification chuẩn (LFW/CFP_FP/AGEDB_30/CALFW/CPLFW)

**File ví dụ:** `eval/eval_iresnet18.py`

Pipeline:

1. Đọc benchmark dạng InsightFace bin (`load_bin()`):
   - decode ảnh từ bytes
   - resize về (112,112)
   - normalize theo mean/std 0.5
2. Extract embedding với flip test (`extract_embeddings()`):
   - chạy model(batch)
   - nếu `use_flip=True` thì lấy embedding của ảnh lật ngang, cộng vào embedding gốc
   - normalize embedding (L2 normalize)
3. **10-fold verification** (`evaluate_10fold()`):
   - chia folds
   - trên train split tìm threshold cho best accuracy
   - đánh giá trên test split
4. Tính **TAR@FAR** (`tar_at_far()`):
   - dùng ROC curve từ `roc_curve(labels, scores)`
   - nội suy tại FAR mục tiêu.
5. In kết quả cho từng dataset.

Trong `eval/eval_iresnet18.py`, code có sẵn ví dụ so sánh:
- `FP32_PATH` vs `Q16_PATH` và gọi `eval_model(Q16_PATH, "Q6", bit=6)`.

---

### IJB-C evaluation

**File ví dụ:** `eval/eval_IJB-C.py`

Pipeline theo từng bước:

1. Load quantized model (ví dụ iResNet18 Q6) tương tự `eval_iresnet18.py`.
2. Dùng **RetinaFace** thông qua `insightface.app.FaceAnalysis` để detect landmark.
3. **Align** template:
   - đọc ảnh `loose_crop`
   - lấy landmark `face.kps`
   - ước lượng similarity transform về mốc 5 điểm
   - warp về 112x112 và lưu vào `aligned/`
4. Extract features:
   - đọc aligned images
   - dùng **flip** (ảnh lật ngang) và cộng feature của 2 hướng
   - normalize embedding
5. Build template pooling:
   - gom theo `template id`
   - mỗi media được lấy trung bình embedding
   - template feature = sum(media_feats)
   - normalize template feature
6. Score cho mỗi pair `(t1, t2)` bằng dot product cosine similarity.
7. Với từng FAR mục tiêu (1e-4, 1e-5, 1e-6), tính TAR bằng ROC curve.

---

## Face recognition realtime (demo)

**File ví dụ:** `main.py`

Demo chạy webcam để:

1. Load quantized model qua `utils/model_loader.py`.
2. Load database embedding từ `face_database.pkl` thông qua `utils/database.py`.
3. Detect face (`utils/face_detector.py`).
4. Lấy embedding (`utils/embedding.py`).
5. Match embedding với database và hiển thị `name (similarity)`.
6. Cho phép:
   - Nhấn **Q** để thoát
   - Nhấn **S** để lưu ảnh debug.

---

## Cấu hình & threshold

**File:** `config.py`

Thông tin quan trọng:

- `MODEL_PATH`: đường dẫn checkpoint quantized (ví dụ `models/iresnet18_q6.pth`).
- `SIMILARITY_THRESHOLD = 0.25`
  - Đây là threshold để decide “Unknown” trong realtime demo.
- `CAMERA_ID`, `FRAME_WIDTH`, `FRAME_HEIGHT`.

---

## Yêu cầu môi trường

**File:** `requirements.txt`

Các package chính:
- `torch`, `torchvision`
- `opencv-python`
- `numpy`
- `Pillow`
- `tqdm`

---

## Gợi ý workflow end-to-end

1. **Chọn backbone** (iResNet18/iResNet50/MobileFaceNet)
2. **Training**
   - Nếu muốn nhanh: chạy `training/ptq/*` (PTQ)
   - Nếu muốn accuracy cao hơn: chạy `training/w*-3p/*` hoặc `training/w0-3p/*` (fine-tuning quantized)
3. **Evaluation**
   - Verification: `eval/eval_iresnet18.py`, `eval/eval_iresnet50.py`, `eval/eval_mobilefacenet.py`
   - IJB-C: `eval/eval_IJB-C.py`
4. **Realtime demo**
   - đảm bảo `config.py` trỏ đúng `MODEL_PATH`
   - đảm bảo database `face_database.pkl` đã được tạo

---


