import io
import wave
import streamlit as st
import wfdb
import neurokit2 as nk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="ECG Arrhythmia Classification", layout="wide")
st.title("🫀 ECG Arrhythmia Detection & Classification Dashboard")

BEAT_CLASSES = {
    'N': 'Normal Beat (N)',
    'V': 'Premature Ventricular Contraction (PVC)',
    'A': 'Atrial Premature Beat (APB)',
    'L': 'Left Bundle Branch Block (LBBB)',
    'R': 'Right Bundle Branch Block (RBBB)'
}

COLOR_MAP = {
    'Normal Beat (N)': '#2ca02c',
    'Premature Ventricular Contraction (PVC)': '#d62728',
    'Atrial Premature Beat (APB)': '#ff7f0e',
    'Left Bundle Branch Block (LBBB)': '#9467bd',
    'Right Bundle Branch Block (RBBB)': '#8c564b'
}

def generate_ecg_audio(r_peaks, y_pred, total_samples, ecg_fs=360, audio_fs=22050):
    duration_sec = total_samples / ecg_fs
    n_audio_samples = int(duration_sec * audio_fs)
    audio = np.zeros(n_audio_samples, dtype=np.float32)

    for pk, label in zip(r_peaks, y_pred):
        t_sec = pk / ecg_fs
        start_idx = int(t_sec * audio_fs)

        if label == 'Normal Beat (N)':
            beep_len = int(0.07 * audio_fs)
            t = np.linspace(0, 0.07, beep_len, endpoint=False)
            envelope = np.sin(np.pi * t / 0.07)
            tone = 0.45 * envelope * np.sin(2 * np.pi * 1000 * t)
        else:
            beep_len = int(0.18 * audio_fs)
            t = np.linspace(0, 0.18, beep_len, endpoint=False)
            pulse = np.where((t < 0.07) | (t > 0.11), 1.0, 0.0)
            tone = 0.75 * pulse * np.sin(2 * np.pi * 600 * t)

        end_idx = min(start_idx + len(tone), n_audio_samples)
        if start_idx < n_audio_samples:
            audio[start_idx:end_idx] += tone[:end_idx - start_idx]

    audio = np.clip(audio, -1.0, 1.0)
    pcm_data = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(audio_fs)
        wf.writeframes(pcm_data.tobytes())
    return buf.getvalue()

def extract_mitbih_features(record_id, max_samples=None):
    fs = 360
    rec = wfdb.rdrecord(record_id, pn_dir='mitdb', sampto=max_samples)
    ann = wfdb.rdann(record_id, 'atr', pn_dir='mitdb', sampto=max_samples)

    raw_ecg = rec.p_signal[:, 0]
    cleaned_ecg = nk.ecg_clean(raw_ecg, sampling_rate=fs)

    valid_idx = [i for i, sym in enumerate(ann.symbol) if sym in BEAT_CLASSES]
    r_peaks = ann.sample[valid_idx]
    symbols = [ann.symbol[i] for i in valid_idx]

    features, labels, valid_peaks = [], [], []
    window = int(0.1 * fs)

    for i in range(1, len(r_peaks) - 1):
        pk = r_peaks[i]
        if pk - window < 0 or pk + window >= len(cleaned_ecg):
            continue

        pre_rr = (r_peaks[i] - r_peaks[i - 1]) / fs
        post_rr = (r_peaks[i + 1] - r_peaks[i]) / fs
        beat_seg = cleaned_ecg[pk - window : pk + window]

        features.append([pre_rr, post_rr, cleaned_ecg[pk], np.ptp(beat_seg), np.std(beat_seg)])
        labels.append(BEAT_CLASSES[symbols[i]])
        valid_peaks.append(pk)

    return cleaned_ecg, fs, np.array(valid_peaks), np.array(features), np.array(labels)

def analyze_custom_signal(raw_signal, fs=360):
    cleaned_ecg = nk.ecg_clean(raw_signal, sampling_rate=fs)
    _, info = nk.ecg_peaks(cleaned_ecg, sampling_rate=fs)
    r_peaks = info["ECG_R_Peaks"]

    features, valid_peaks = [], []
    window = int(0.1 * fs)

    for i in range(1, len(r_peaks) - 1):
        pk = r_peaks[i]
        if pk - window < 0 or pk + window >= len(cleaned_ecg):
            continue

        pre_rr = (r_peaks[i] - r_peaks[i - 1]) / fs
        post_rr = (r_peaks[i + 1] - r_peaks[i]) / fs
        beat_seg = cleaned_ecg[pk - window : pk + window]

        features.append([pre_rr, post_rr, cleaned_ecg[pk], np.ptp(beat_seg), np.std(beat_seg)])
        valid_peaks.append(pk)

    return cleaned_ecg, np.array(valid_peaks), np.array(features)

def digitize_ecg_image(uploaded_img, duration_sec=10, target_fs=360):
    img = Image.open(uploaded_img).convert("L")
    img_arr = np.array(img, dtype=float)
    height, width = img_arr.shape

    inverted = 255.0 - img_arr
    trace_y = []
    for col in range(width):
        col_vals = inverted[:, col]
        max_val = np.max(col_vals)
        mask = col_vals >= (0.75 * max_val)
        y_indices = np.where(mask)[0]
        trace_y.append(np.mean(y_indices) if len(y_indices) > 0 else height / 2.0)

    raw_wave = (height / 2.0) - np.array(trace_y)
    raw_wave = (raw_wave - np.mean(raw_wave)) / (np.std(raw_wave) + 1e-6)

    target_len = int(duration_sec * target_fs)
    x_old = np.linspace(0, 1, len(raw_wave))
    x_new = np.linspace(0, 1, target_len)
    resampled_signal = np.interp(x_new, x_old, raw_wave)
    return resampled_signal

@st.cache_resource
def train_arrhythmia_model():
    train_records = ["100", "106", "118", "119", "200", "208"]
    X_all, y_all = [], []
    for rec_id in train_records:
        _, _, _, X, y = extract_mitbih_features(rec_id, max_samples=360 * 180)
        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_test_pred)
    return clf, acc, y_test, y_test_pred

with st.spinner("Loading Arrhythmia Classification Model..."):
    model, model_acc, y_test, y_test_pred = train_arrhythmia_model()

st.sidebar.header("1. Choose ECG Input Source")
input_mode = st.sidebar.radio(
    "Select Data Source:",
    ["🌐 MIT-BIH Online Dataset", "📄 Offline CSV File Upload", "🖼️ ECG Graph Image Upload"]
)

st.sidebar.success(f"Model Validation Accuracy: **{model_acc * 100:.1f}%**")

@st.cache_data
def generate_demo_csv():
    rec = wfdb.rdrecord("106", pn_dir='mitdb', sampto=360 * 20)
    df = pd.DataFrame({"ECG_mV": rec.p_signal[:, 0]})
    return df.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    label="⬇️ Download Sample Offline CSV (Record 106)",
    data=generate_demo_csv(),
    file_name="sample_ecg_record_106.csv",
    mime="text/csv",
    help="Save this CSV before your demo so you have an offline test file ready!"
)

tab1, tab2 = st.tabs(["🩺 Live ECG Monitor", "📊 Model Evaluation (Report & Viva)"])

with tab1:
    fs = 360
    cleaned_ecg, r_peaks, X_patient, y_true = None, None, None, None
    data_ready = False

    if input_mode == "🌐 MIT-BIH Online Dataset":
        st.sidebar.subheader("MIT-BIH Settings")
        selected_record = st.sidebar.selectbox(
            "Select Patient Record:", ["106", "119", "200", "208", "100", "118"]
        )
        duration_sec = st.sidebar.slider("Signal Duration (seconds):", 10, 60, 25)
        cleaned_ecg, fs, r_peaks, X_patient, y_true = extract_mitbih_features(
            selected_record, max_samples=duration_sec * fs
        )
        data_ready = True

    elif input_mode == "📄 Offline CSV File Upload":
        st.sidebar.subheader("CSV Upload Settings")
        uploaded_csv = st.sidebar.file_uploader("Upload ECG CSV File:", type=["csv"])
        fs = st.sidebar.number_input("Sampling Rate (Hz):", min_value=100, max_value=1000, value=360)

        if uploaded_csv is not None:
            df_csv = pd.read_csv(uploaded_csv)
            numeric_cols = df_csv.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                ecg_col = st.sidebar.selectbox("Select ECG Signal Column:", numeric_cols)
                raw_sig = df_csv[ecg_col].dropna().values
                cleaned_ecg, r_peaks, X_patient = analyze_custom_signal(raw_sig, fs=fs)
                data_ready = True
            else:
                st.error("No numeric ECG column found in the uploaded CSV.")
        else:
            st.info("👈 Upload a CSV file in the sidebar (or click **Download Sample Offline CSV** to test immediately).")

    elif input_mode == "🖼️ ECG Graph Image Upload":
        st.sidebar.subheader("Image Digitizer Settings")
        uploaded_img = st.sidebar.file_uploader("Upload ECG Strip Image (PNG/JPG):", type=["png", "jpg", "jpeg"])
        img_duration = st.sidebar.slider("Estimated Strip Duration (seconds):", 5, 30, 10)

        if uploaded_img is not None:
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(uploaded_img, caption="Uploaded ECG Strip Image", width='stretch')
            with col_img2:
                st.info("Image Digitization Pipeline: Grayscale Conversion → Grid Suppression → Trace Extraction → 360 Hz Resampling → Arrhythmia Classification.")

            raw_sig = digitize_ecg_image(uploaded_img, duration_sec=img_duration, target_fs=fs)
            cleaned_ecg, r_peaks, X_patient = analyze_custom_signal(raw_sig, fs=fs)
            data_ready = True
        else:
            st.info("👈 Upload a clear photo or screenshot of an ECG waveform strip (dark trace on a light background) in the sidebar.")

    if data_ready and X_patient is not None and len(X_patient) > 0:
        y_pred = model.predict(X_patient)

        rr_intervals = np.diff(r_peaks) / fs
        avg_hr = int(np.mean(60.0 / rr_intervals)) if len(rr_intervals) > 0 else 0
        total_beats = len(r_peaks)
        anomalies_mask = y_pred != 'Normal Beat (N)'
        anomaly_count = int(np.sum(anomalies_mask))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average Heart Rate", f"{avg_hr} BPM")
