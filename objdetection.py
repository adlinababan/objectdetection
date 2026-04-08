import time
from collections import Counter

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Deteksi Kendaraan dari Stream M3U8", layout="wide")

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

DISPLAY_NAMES = {
    "car": "Mobil",
    "motorcycle": "Motor",
    "bus": "Bus",
    "truck": "Truk",
}

DEFAULT_URL = "https://atcsdishub.medan.go.id/stream/L1RADENSALEHBALAIKOTA/stream.m3u8"


@st.cache_resource
def load_model(model_name: str):
    return YOLO(model_name)


def open_stream(url: str, transport: str = "tcp"):
    options = {
        "rtsp_transport": transport,
        "fflags": "nobuffer",
        "flags": "low_delay",
        "strict": "experimental",
    }
    return av.open(url, options=options)


def run_inference(model, frame, conf: float, imgsz: int):
    results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
    result = results[0]

    counts = Counter()
    annotated = frame.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return annotated, counts

    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    for box, cls_id, score in zip(boxes, classes, confs):
        if cls_id not in VEHICLE_CLASSES:
            continue

        label_key = VEHICLE_CLASSES[cls_id]
        counts[label_key] += 1

        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label_key} {score:.2f}"
        cv2.putText(
            annotated,
            text,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated, counts


def counts_to_df(counts: Counter):
    rows = []
    total = 0
    for key in ["car", "motorcycle", "bus", "truck"]:
        value = int(counts.get(key, 0))
        total += value
        rows.append({"Jenis Kendaraan": DISPLAY_NAMES[key], "Jumlah": value})
    rows.append({"Jenis Kendaraan": "Total", "Jumlah": total})
    return pd.DataFrame(rows)


st.title("Deteksi Kendaraan dari Stream .m3u8")
st.caption("Masukkan link stream .m3u8 secara dinamis, lalu jalankan deteksi kendaraan secara realtime.")

with st.sidebar:
    st.header("Pengaturan")
    stream_url = st.text_input("Link stream .m3u8", value=DEFAULT_URL, placeholder="https://.../stream.m3u8")
    model_name = st.selectbox("Model YOLO", ["yolov8n.pt", "yolov8s.pt"], index=0)
    confidence = st.slider("Confidence", 0.1, 1.0, 0.3, 0.05)
    imgsz = st.select_slider("Ukuran inferensi", options=[320, 416, 480, 640, 800], value=640)
    skip_frames = st.slider("Skip frame", 1, 10, 2)
    max_width = st.slider("Lebar tampilan frame", 480, 1280, 960, 80)
    reconnect_delay = st.slider("Delay reconnect (detik)", 1, 15, 3)
    start_button = st.button("Mulai Deteksi", type="primary")
    stop_button = st.button("Stop")

if "run_detection" not in st.session_state:
    st.session_state.run_detection = False

if start_button:
    st.session_state.run_detection = True
if stop_button:
    st.session_state.run_detection = False

col1, col2 = st.columns([3, 1])
frame_placeholder = col1.empty()
info_placeholder = col1.empty()
stats_placeholder = col2.empty()
table_placeholder = col2.empty()
status_placeholder = st.empty()

if st.session_state.run_detection:
    if not stream_url.strip():
        st.session_state.run_detection = False
        st.error("Link stream .m3u8 wajib diisi.")
    else:
        try:
            model = load_model(model_name)
            status_placeholder.info("Menghubungkan ke stream...")
            container = open_stream(stream_url.strip())

            frame_index = 0
            latest_counts = Counter()
            fps_time = time.time()
            frames_processed = 0

            for frame in container.decode(video=0):
                if not st.session_state.run_detection:
                    status_placeholder.warning("Deteksi dihentikan.")
                    break

                frame_index += 1
                if frame_index % skip_frames != 0:
                    continue

                image = frame.to_ndarray(format="bgr24")
                annotated, counts = run_inference(model, image, confidence, imgsz)
                latest_counts = counts

                h, w = annotated.shape[:2]
                new_h = int((max_width / w) * h)
                resized = cv2.resize(annotated, (max_width, new_h))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

                frames_processed += 1
                elapsed = max(time.time() - fps_time, 1e-6)
                fps = frames_processed / elapsed
                total_detected = sum(latest_counts.values())

                info_placeholder.markdown(
                    f"**FPS proses:** {fps:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; **Total terdeteksi pada frame:** {total_detected}"
                )

                table_placeholder.dataframe(counts_to_df(latest_counts), use_container_width=True, hide_index=True)
                stats_placeholder.metric("Total Frame Diproses", frames_processed)

            try:
                container.close()
            except Exception:
                pass

        except av.error.FFmpegError as e:
            st.session_state.run_detection = False
            status_placeholder.error(f"Gagal membuka stream: {e}")
            st.info(f"Silakan cek kembali URL .m3u8 atau coba lagi dalam {reconnect_delay} detik.")
        except Exception as e:
            st.session_state.run_detection = False
            status_placeholder.error(f"Terjadi error: {e}")
else:
    st.info("Masukkan link stream .m3u8 di sidebar, lalu klik **Mulai Deteksi**.")
    table_placeholder.dataframe(counts_to_df(Counter()), use_container_width=True, hide_index=True)

