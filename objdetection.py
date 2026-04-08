import time
from collections import Counter

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="M3U8 Vehicle Detection", layout="wide")

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

st.title("Deteksi Kendaraan dari Stream .m3u8")
st.caption("Input link stream .m3u8 secara dinamis, lalu jalankan deteksi kendaraan realtime.")

with st.sidebar:
    st.header("Pengaturan")
    stream_url = st.text_input(
        "URL Stream .m3u8",
        value="https://atcsdishub.medan.go.id/stream/L1RADENSALEHBALAIKOTA/stream.m3u8",
        help="Masukkan link HLS / .m3u8 yang ingin dideteksi.",
    )
    model_name = st.selectbox("Model YOLO", ["yolov8n.pt", "yolov8s.pt"], index=0)
    conf_thres = st.slider("Confidence Threshold", 0.10, 0.95, 0.30, 0.05)
    iou_thres = st.slider("IoU Threshold", 0.10, 0.95, 0.45, 0.05)
    imgsz = st.select_slider("Ukuran Inferensi", options=[320, 416, 480, 640, 960], value=640)
    skip_frames = st.slider("Skip Frame", 1, 10, 2, 1)
    max_width = st.slider("Lebar Tampilan Frame", 480, 1280, 960, 40)
    retry_open = st.slider("Percobaan Buka Stream", 1, 10, 3, 1)

st.markdown(
    """
**Catatan**
- Aplikasi ini tidak memakai `av`, jadi lebih aman untuk deploy di Streamlit Cloud.
- Jika stream gagal dibuka, biasanya penyebabnya adalah stream diblokir, butuh header khusus, atau server stream sedang tidak aktif.
- Untuk menghentikan proses deteksi, klik **Stop detection** lalu jalankan ulang jika perlu.
"""
)

if "run_detection" not in st.session_state:
    st.session_state.run_detection = False

col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    if st.button("Start detection", use_container_width=True):
        st.session_state.run_detection = True
with col_btn2:
    if st.button("Stop detection", use_container_width=True):
        st.session_state.run_detection = False

status_box = st.empty()
frame_box = st.empty()
metric_row = st.columns(6)
log_box = st.empty()

@st.cache_resource(show_spinner=True)
def load_model(name: str):
    return YOLO(name)

def resize_keep_ratio(frame, target_width: int):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    ratio = target_width / float(w)
    new_h = int(h * ratio)
    return cv2.resize(frame, (target_width, new_h))

def open_stream(url: str, retries: int = 3, wait_seconds: int = 2):
    cap = None
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
        time.sleep(wait_seconds)
    return None

def annotate_and_count(frame, results):
    counts = Counter({"car": 0, "motorcycle": 0, "bus": 0, "truck": 0})
    annotated = frame.copy()

    if not results or len(results) == 0:
        return annotated, counts, 0

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return annotated, counts, 0

    for box in boxes:
        cls_id = int(box.cls[0].item())
        if cls_id not in VEHICLE_CLASSES:
            continue

        label = VEHICLE_CLASSES[cls_id]
        conf = float(box.conf[0].item())
        counts[label] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    total = sum(counts.values())
    return annotated, counts, total

def update_metrics(fps=0.0, total=0, counts=None):
    if counts is None:
        counts = Counter({"car": 0, "motorcycle": 0, "bus": 0, "truck": 0})
    labels = [
        ("FPS", f"{fps:.1f}"),
        ("Total", int(total)),
        ("Mobil", int(counts["car"])),
        ("Motor", int(counts["motorcycle"])),
        ("Bus", int(counts["bus"])),
        ("Truk", int(counts["truck"])),
    ]
    for col, (name, value) in zip(metric_row, labels):
        col.metric(name, value)

update_metrics()

if st.session_state.run_detection:
    if not stream_url.strip():
        status_box.error("URL stream kosong. Masukkan link .m3u8 terlebih dahulu.")
        st.stop()

    status_box.info("Memuat model...")
    model = load_model(model_name)

    status_box.info("Membuka stream...")
    cap = open_stream(stream_url, retries=retry_open)

    if cap is None:
        status_box.error("Gagal membuka stream. Cek kembali URL .m3u8 atau status stream.")
        st.session_state.run_detection = False
        st.stop()

    status_box.success("Stream berhasil dibuka. Deteksi sedang berjalan...")

    frame_index = 0
    processed_frames = 0
    start_time = time.time()
    last_counts = Counter({"car": 0, "motorcycle": 0, "bus": 0, "truck": 0})
    last_total = 0

    try:
        while st.session_state.run_detection:
            ok, frame = cap.read()
            if not ok or frame is None:
                status_box.warning("Frame tidak terbaca atau stream terputus.")
                break

            frame_index += 1

            if frame_index % skip_frames != 0:
                continue

            t0 = time.time()
            results = model.predict(
                frame,
                conf=conf_thres,
                iou=iou_thres,
                imgsz=imgsz,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False,
                device="cpu",
            )
            annotated, counts, total = annotate_and_count(frame, results)
            annotated = resize_keep_ratio(annotated, max_width)

            processed_frames += 1
            elapsed = max(time.time() - start_time, 1e-6)
            fps = processed_frames / elapsed

            last_counts = counts
            last_total = total

            frame_box.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
            update_metrics(fps=fps, total=total, counts=counts)

            infer_ms = int((time.time() - t0) * 1000)
            log_box.info(
                f"Frame diproses: {processed_frames} | "
                f"Mobil: {counts['car']} | Motor: {counts['motorcycle']} | "
                f"Bus: {counts['bus']} | Truk: {counts['truck']} | "
                f"Total: {total} | Inferensi: {infer_ms} ms"
            )

    except Exception as e:
        status_box.error(f"Terjadi error saat proses deteksi: {e}")
    finally:
        cap.release()
        st.session_state.run_detection = False
        update_metrics(fps=0.0, total=last_total, counts=last_counts)
        status_box.info("Deteksi berhenti. Anda bisa menekan Start detection untuk menjalankan lagi.")
