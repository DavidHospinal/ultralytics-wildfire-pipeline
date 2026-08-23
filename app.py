"""
H'spinal Systems — Wildfire Early Warning System
MVP: Streamlit interface for an Ultralytics Platform inference endpoint
Author: Hospinal Systems
"""

import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DEFAULT_URL = os.environ.get("ULTRALYTICS_ENDPOINT_URL", "")
DEFAULT_API_KEY = os.environ.get("ULTRALYTICS_API_KEY", "")

CLASS_CONFIG = {
    0: {"name": "smoke", "color": (180, 180, 180)},   # gray  (BGR)
    1: {"name": "fire",  "color": (0, 80, 255)},      # red-orange (BGR)
}

# ──────────────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="H'spinal Systems · Wildfire Early Warning",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS — Dark professional theme
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d0f14;
        color: #e2e8f0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #111520;
        border-right: 1px solid #1e2535;
    }
    /* ── Header ── */
    .app-header {
        padding: 1.4rem 0 0.6rem 0;
        border-bottom: 1px solid #1e2535;
        margin-bottom: 1.6rem;
    }
    .app-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #f97316;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin: 0.2rem 0 0 0;
    }
    /* ── Metric cards ── */
    .metric-card {
        background: #111520;
        border: 1px solid #1e2535;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f97316;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* ── Detection badge ── */
    .badge-fire {
        display: inline-block;
        background: rgba(249,115,22,0.15);
        color: #f97316;
        border: 1px solid #f97316;
        border-radius: 5px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-smoke {
        display: inline-block;
        background: rgba(148,163,184,0.15);
        color: #94a3b8;
        border: 1px solid #94a3b8;
        border-radius: 5px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-clear {
        display: inline-block;
        background: rgba(34,197,94,0.15);
        color: #22c55e;
        border: 1px solid #22c55e;
        border-radius: 5px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    /* ── Buttons ── */
    .stButton > button {
        background: #f97316;
        color: #0d0f14;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.6rem;
        width: 100%;
    }
    .stButton > button:hover {
        background: #ea6d0c;
        color: #0d0f14;
    }
    /* ── Slider / inputs ── */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #f97316 !important;
    }
    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #111520;
        border: 1.5px dashed #1e2535;
        border-radius: 10px;
        padding: 0.5rem;
    }
    /* ── Divider ── */
    hr { border-color: #1e2535; }
    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d0f14; }
    ::-webkit-scrollbar-thumb { background: #1e2535; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <p class="app-title">H'spinal Systems · Wildfire Early Warning</p>
        <p class="app-subtitle">
            Detección temprana de incendios forestales · Powered by Ultralytics Platform
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# SIDEBAR — Configuration
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Endpoint Configuration")
    endpoint_url = st.text_input(
        "Inference URL",
        value=DEFAULT_URL,
        placeholder="https://.../predict",
        help="Ultralytics Platform deployment endpoint",
    )
    api_key = st.text_input(
        "API Key",
        value=DEFAULT_API_KEY,
        type="password",
        placeholder="Paste your API key",
        help="Bearer token. Never commit this value to Git.",
    )

    st.divider()
    st.markdown("### Inference Parameters")
    conf_threshold = st.slider("Confidence threshold", 0.10, 0.95, 0.25, 0.05)
    iou_threshold  = st.slider("IoU threshold (NMS)", 0.10, 0.95, 0.70, 0.05)
    imgsz          = st.select_slider(
        "Input image size", options=[320, 416, 512, 640, 800], value=640
    )

    st.divider()
    st.markdown("### Video Parameters")
    frame_skip = st.slider(
        "Process every N frames",
        1, 15, 5,
        help="Higher = faster but lower temporal resolution",
    )
    show_frame_counter = st.checkbox("Show frame counter on video", value=True)

    st.divider()
    st.markdown(
        "<p style='font-size:0.72rem;color:#334155;'>"
        "Hospinal Systems · 2026 · Ultralytics Platform</p>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def validate_config(url: str, key: str) -> tuple[bool, str]:
    if not url.startswith("https://"):
        return False, "La URL debe comenzar con https://"
    if len(key) < 20:
        return False, "La API Key parece demasiado corta. Verifiquela."
    return True, ""


def call_inference_api(
    image_bytes: bytes,
    filename: str,
    url: str,
    key: str,
    conf: float,
    iou: float,
    imgsz: int,
) -> dict:
    """POST a single image to the Ultralytics inference endpoint."""
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        data={"conf": conf, "iou": iou, "imgsz": imgsz},
        files={"file": (filename, image_bytes, "image/jpeg")},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def parse_detections(api_response: dict) -> list[dict]:
    """
    Normalize Ultralytics HUB response to a flat list of detections.
    Expected format:
        {"images": [{"results": [{"class": int, "name": str,
                                  "confidence": float,
                                  "box": {"x1","y1","x2","y2"}}]}]}
    """
    detections = []
    images = api_response.get("images", [])
    if not images:
        return detections
    for result in images[0].get("results", []):
        box = result.get("box", {})
        detections.append(
            {
                "class_id":   int(result.get("class", -1)),
                "name":       result.get("name", "unknown"),
                "confidence": float(result.get("confidence", 0.0)),
                "x1": int(box.get("x1", 0)),
                "y1": int(box.get("y1", 0)),
                "x2": int(box.get("x2", 0)),
                "y2": int(box.get("y2", 0)),
            }
        )
    return detections


def draw_detections(frame_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on a BGR frame."""
    out = frame_bgr.copy()
    for det in detections:
        cfg   = CLASS_CONFIG.get(det["class_id"], {"name": det["name"], "color": (200, 200, 200)})
        color = cfg["color"]
        label = f"{cfg['name']} {det['confidence']:.2f}"
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            out, label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 1, cv2.LINE_AA,
        )
    return out


def encode_frame_to_jpeg(frame_bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


def status_badge(detections: list[dict]) -> str:
    names = {d["name"].lower() for d in detections}
    if "fire" in names:
        return '<span class="badge-fire">FIRE DETECTED</span>'
    if "smoke" in names:
        return '<span class="badge-smoke">SMOKE DETECTED</span>'
    return '<span class="badge-clear">CLEAR</span>'


# ──────────────────────────────────────────────
# INPUT — File uploader (image + video)
# ──────────────────────────────────────────────
ok, err = validate_config(endpoint_url, api_key)
if not ok:
    st.error(f"Configuracion invalida: {err}")
    st.stop()

uploaded = st.file_uploader(
    "Cargar imagen o video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"],
    help="Formatos soportados: JPG, PNG, BMP, MP4, AVI, MOV",
)

# ──────────────────────────────────────────────
# PROCESS — IMAGE
# ──────────────────────────────────────────────
if uploaded is not None:
    file_ext = Path(uploaded.name).suffix.lower()
    is_video = file_ext in {".mp4", ".avi", ".mov"}

    if not is_video:
        # ── IMAGE PROCESSING ──────────────────
        col_in, col_out = st.columns(2, gap="large")

        with col_in:
            st.markdown("**Input**")
            image_pil = Image.open(uploaded).convert("RGB")
            st.image(image_pil, use_container_width=True)

        with col_out:
            st.markdown("**Detection Result**")
            run_btn = st.button("Analyze Image", key="btn_image")

        if run_btn:
            with st.spinner("Sending to Ultralytics endpoint..."):
                try:
                    img_bytes = uploaded.getvalue()
                    t0 = time.perf_counter()
                    response_json = call_inference_api(
                        img_bytes, uploaded.name,
                        endpoint_url, api_key,
                        conf_threshold, iou_threshold, imgsz,
                    )
                    latency = time.perf_counter() - t0

                    detections = parse_detections(response_json)

                    # Draw boxes
                    np_img = np.array(image_pil)
                    frame_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                    annotated  = draw_detections(frame_bgr, detections)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                    with col_out:
                        st.image(annotated_rgb, use_container_width=True)
                        st.markdown(status_badge(detections), unsafe_allow_html=True)

                    # Metrics row
                    st.divider()
                    m1, m2, m3, m4 = st.columns(4)
                    fire_dets  = [d for d in detections if d["name"].lower() == "fire"]
                    smoke_dets = [d for d in detections if d["name"].lower() == "smoke"]
                    top_conf   = max((d["confidence"] for d in detections), default=0.0)

                    m1.markdown(
                        f'<div class="metric-card"><div class="metric-value">{len(detections)}</div>'
                        f'<div class="metric-label">Total Detections</div></div>',
                        unsafe_allow_html=True,
                    )
                    m2.markdown(
                        f'<div class="metric-card"><div class="metric-value">{len(fire_dets)}</div>'
                        f'<div class="metric-label">Fire</div></div>',
                        unsafe_allow_html=True,
                    )
                    m3.markdown(
                        f'<div class="metric-card"><div class="metric-value">{len(smoke_dets)}</div>'
                        f'<div class="metric-label">Smoke</div></div>',
                        unsafe_allow_html=True,
                    )
                    m4.markdown(
                        f'<div class="metric-card"><div class="metric-value">{top_conf:.0%}</div>'
                        f'<div class="metric-label">Top Confidence</div></div>',
                        unsafe_allow_html=True,
                    )

                    # Detail table
                    if detections:
                        st.divider()
                        st.markdown("**Detection Details**")
                        rows = "".join(
                            f"<tr><td>{i+1}</td><td>{d['name']}</td>"
                            f"<td>{d['confidence']:.3f}</td>"
                            f"<td>({d['x1']},{d['y1']}) → ({d['x2']},{d['y2']})</td></tr>"
                            for i, d in enumerate(detections)
                        )
                        st.markdown(
                            f"<table style='width:100%;font-size:0.82rem;color:#cbd5e1;'>"
                            f"<thead><tr style='color:#64748b;'><th>#</th><th>Class</th>"
                            f"<th>Conf</th><th>BBox</th></tr></thead>"
                            f"<tbody>{rows}</tbody></table>",
                            unsafe_allow_html=True,
                        )

                    st.caption(f"Latency: {latency*1000:.0f} ms · "
                               f"Model imgsz={imgsz} · conf={conf_threshold} · iou={iou_threshold}")

                except requests.exceptions.HTTPError as e:
                    st.error(f"HTTP {e.response.status_code}: {e.response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("No se pudo conectar al endpoint. Verifique la URL y su conexion.")
                except requests.exceptions.Timeout:
                    st.error("La peticion supero el tiempo limite (30s). Intente con imgsz menor.")
                except Exception as e:
                    st.error(f"Error inesperado: {e}")

    else:
        # ── VIDEO PROCESSING ──────────────────
        st.markdown("---")
        st.markdown("**Video Analysis — Frame-by-Frame Inference**")

        col_ctrl, col_info = st.columns([1, 2])
        with col_ctrl:
            run_video = st.button("Process Video", key="btn_video")
        with col_info:
            st.caption(
                f"Se procesara 1 de cada {frame_skip} frames. "
                f"Los frames intermedios se copian del ultimo resultado."
            )

        if run_video:
            # Write uploaded video to a temp file (OpenCV needs a path)
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = cap.get(cv2.CAP_PROP_FPS) or 25
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            st.caption(
                f"Video: {width}x{height} · {total_frames} frames · {fps:.1f} fps · "
                f"Frames a procesar: {total_frames // frame_skip}"
            )

            # Output video writer
            out_path = tmp_path.replace(file_ext, "_annotated.mp4")
            fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
            writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            progress_bar  = st.progress(0, text="Procesando frames...")
            preview_slot  = st.empty()
            stats_slot    = st.empty()

            frame_idx       = 0
            processed_count = 0
            fire_frames     = 0
            smoke_frames    = 0
            last_detections: list[dict] = []
            last_annotated: np.ndarray | None = None
            api_errors      = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    try:
                        jpeg_bytes = encode_frame_to_jpeg(frame)
                        resp_json  = call_inference_api(
                            jpeg_bytes, f"frame_{frame_idx:06d}.jpg",
                            endpoint_url, api_key,
                            conf_threshold, iou_threshold, imgsz,
                        )
                        last_detections = parse_detections(resp_json)
                        processed_count += 1

                        if any(d["name"].lower() == "fire"  for d in last_detections):
                            fire_frames  += 1
                        if any(d["name"].lower() == "smoke" for d in last_detections):
                            smoke_frames += 1

                    except Exception:
                        api_errors += 1
                        last_detections = []

                # Draw on every frame (use last known detections)
                annotated = draw_detections(frame, last_detections)

                if show_frame_counter:
                    cv2.putText(
                        annotated,
                        f"Frame {frame_idx}/{total_frames}",
                        (12, height - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (100, 100, 100), 1, cv2.LINE_AA,
                    )

                writer.write(annotated)
                last_annotated = annotated

                # Update UI every 10 written frames
                if frame_idx % (frame_skip * 3) == 0:
                    pct = frame_idx / max(total_frames, 1)
                    progress_bar.progress(
                        min(pct, 1.0),
                        text=f"Procesando frame {frame_idx}/{total_frames}…",
                    )
                    if last_annotated is not None:
                        preview_rgb = cv2.cvtColor(last_annotated, cv2.COLOR_BGR2RGB)
                        preview_slot.image(preview_rgb, use_container_width=True)
                    stats_slot.markdown(
                        f"Frames procesados via API: **{processed_count}** · "
                        f"Fire frames: **{fire_frames}** · "
                        f"Smoke frames: **{smoke_frames}** · "
                        f"Errores API: **{api_errors}**"
                    )

                frame_idx += 1

            cap.release()
            writer.release()

            progress_bar.progress(1.0, text="Procesamiento completado.")

            # Final stats
            st.divider()
            s1, s2, s3, s4 = st.columns(4)
            s1.markdown(
                f'<div class="metric-card"><div class="metric-value">{processed_count}</div>'
                f'<div class="metric-label">Frames Analizados</div></div>',
                unsafe_allow_html=True,
            )
            s2.markdown(
                f'<div class="metric-card"><div class="metric-value">{fire_frames}</div>'
                f'<div class="metric-label">Frames con Fire</div></div>',
                unsafe_allow_html=True,
            )
            s3.markdown(
                f'<div class="metric-card"><div class="metric-value">{smoke_frames}</div>'
                f'<div class="metric-label">Frames con Smoke</div></div>',
                unsafe_allow_html=True,
            )
            risk_pct = fire_frames / max(processed_count, 1) * 100
            s4.markdown(
                f'<div class="metric-card"><div class="metric-value">{risk_pct:.0f}%</div>'
                f'<div class="metric-label">Risk Rate</div></div>',
                unsafe_allow_html=True,
            )

            # Download button for annotated video
            st.divider()
            with open(out_path, "rb") as f:
                st.download_button(
                    label="Download Annotated Video",
                    data=f,
                    file_name=f"wildfire_annotated_{uploaded.name}",
                    mime="video/mp4",
                )

            if api_errors > 0:
                st.warning(
                    f"{api_errors} frames fallaron al contactar la API. "
                    f"Se usaron las detecciones del frame anterior para esos frames."
                )

else:
    # ── EMPTY STATE ───────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 4rem 0; color: #334155;">
            <p style="font-size:3rem;">🌲</p>
            <p style="font-size:1.05rem; color:#475569;">
                Cargue una imagen o video para comenzar el analisis de deteccion de incendios.
            </p>
            <p style="font-size:0.8rem; color:#334155;">
                Clases detectadas: <strong style="color:#f97316;">fire</strong>
                &nbsp;|&nbsp;
                <strong style="color:#94a3b8;">smoke</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
