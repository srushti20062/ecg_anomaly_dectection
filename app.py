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
st.title("ðŸ«€ ECG Arrhythmia Detection & Classification Dashboard")

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

FEATURE_NAMES = [
    "Pre-RR Interval (sec)",
    "Post-RR Interval (sec)",
    "R-Peak Amplitude (mV)",
    "QRS Peak-to-Peak Height (mV)",
    "Beat Waveform Energy (Std Dev)"
]

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
            tone = 0.45 * np.sin(np.pi * t / 0.07) * np.sin(2 * np.pi * 1000 * t)
        else:
            beep_len = int(0.18 * audio_fs)
            t = np.linspace(0, 0.18, beep_len, endpoint=False)
            pulse = np.where((t < 0.07) | (t > 0.11), 1.0, 0.0)
            tone = 0.75 * pulse * np.sin(2 * np.pi * 600 * t)

        end_idx = min(start_idx + len(tone), n_audio_samples)
        if start_idx < n_audio_samples:
            audio[start_idx:end_idx] += tone[:end_idx - start_idx]

    pcm_data = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
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
    cleaned_ecg = nk.ecg_clean(rec.p_signal[:, 0], sampling_rate=fs)

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
        mask = col_vals >= (0.75 * np.max(col_vals))
        y_indices = np.where(mask)
        trace_y.append(np.mean(y_indices) if len(y_indices) > 0 else height / 2.0)

    raw_wave = (height / 2.0) - np.array(trace_y)
    raw_wave = (raw_wave - np.mean(raw_wave)) / (np.std(raw_wave) + 1e-6)
    target_len = int(duration_sec * target_fs)
    return np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(raw_wave)), raw_wave)

def get_precautions_and_guidance(avg_hr, y_pred):
    unique_abnormal = [b for b in np.unique(y_pred) if b != 'Normal Beat (N)']
    if avg_hr < 60:
        hr_status = "Bradycardia (Low Resting Rate < 60 BPM)"
    elif avg_hr > 100:
        hr_status = "Tachycardia (Elevated Rate > 100 BPM)"
    else:
        hr_status = "Normal Resting Heart Rate (60â€“100 BPM)"

    precautions = []
    if len(unique_abnormal) == 0 and 60 <= avg_hr <= 100:
        level = "normal"
        summary = "No arrhythmic beats detected. Rhythm is consistent with Normal Sinus Rhythm."
        precautions = [
            "Maintain regular cardiovascular exercise (e.g., 30 minutes of brisk walking most days).",
            "Stay well-hydrated and follow a balanced, heart-healthy diet low in excess sodium and trans fats.",
            "Continue routine annual health checkups and monitor blood pressure periodically."
        ]
    else:
        level = "warning"
        detected_str = ", ".join(unique_abnormal) if unique_abnormal else "Heart Rate outside 60â€“100 BPM range"
        summary = f"Irregularities detected: **{detected_str}** | Rate Status: **{hr_status}**."

        if any("PVC" in b or "APB" in b for b in unique_abnormal):
            precautions.append("Limit stimulants such as caffeine, nicotine, energy drinks, and alcohol, which commonly trigger premature beats.")
            precautions.append("Prioritize 7â€“8 hours of quality sleep and practice stress-reduction techniques.")
        if any("Bundle Branch Block" in b for b in unique_abnormal):
            precautions.append("Schedule a 12-lead ECG and echocardiogram with a cardiologist to evaluate underlying conduction pathways.")
        if avg_hr > 100:
            precautions.append("Rest in a calm seated position, rehydrate slowly, and avoid strenuous exertion until heart rate stabilizes.")
        if avg_hr < 60:
            precautions.append("Monitor for dizziness, lightheadedness, or unusual fatigue, and review any rate-lowering medications with a doctor.")

        precautions.append("Consult a qualified cardiologist or physician for formal clinical evaluationâ€”seek urgent medical care if accompanied by chest pain, shortness of breath, or fainting.")

    return level, hr_status, summary, precautions

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
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    return clf, accuracy_score(y_test, y_test_pred), y_test, y_test_pred

model, model_acc, y_test, y_test_pred = train_arrhythmia_model()

st.sidebar.header("1. Choose ECG Input Source")
input_mode = st.sidebar.radio("Select Data Source:", ["MIT-BIH Online Dataset", "Offline CSV File Upload", "ECG Graph Image Upload"])
st.sidebar.success(f"Model Validation Accuracy: **{model_acc * 100:.1f}%**")

@st.cache_data
def generate_demo_csv():
    rec = wfdb.rdrecord("106", pn_dir='mitdb', sampto=360 * 20)
    return pd.DataFrame({"ECG_mV": rec.p_signal[:, 0]}).to_csv(index=False).encode("utf-8")

st.sidebar.download_button("Download Sample Offline CSV (Record 106)", data=generate_demo_csv(), file_name="sample_ecg_record_106.csv", mime="text/csv")

tab1, tab2 = st.tabs(["Live ECG Monitor", "Model Evaluation (Report & Viva)"])

with tab1:
    fs = 360
    cleaned_ecg, r_peaks, X_patient, y_true = None, None, None, None
    data_ready = False

    if input_mode == "MIT-BIH Online Dataset":
        selected_record = st.sidebar.selectbox("Select Patient Record:", ["106", "119", "200", "208", "100", "118"])
        duration_sec = st.sidebar.slider("Signal Duration (seconds):", 10, 60, 25)
        cleaned_ecg, fs, r_peaks, X_patient, y_true = extract_mitbih_features(selected_record, max_samples=duration_sec * fs)
        data_ready = True

    elif input_mode == "Offline CSV File Upload":
        uploaded_csv = st.sidebar.file_uploader("Upload ECG CSV File:", type=["csv"])
        fs = st.sidebar.number_input("Sampling Rate (Hz):", min_value=100, max_value=1000, value=360)
        if uploaded_csv is not None:
            df_csv = pd.read_csv(uploaded_csv)
            numeric_cols = df_csv.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                ecg_col = st.sidebar.selectbox("Select ECG Signal Column:", numeric_cols)
                cleaned_ecg, r_peaks, X_patient = analyze_custom_signal(df_csv[ecg_col].dropna().values, fs=fs)

    if data_ready and X_patient is not None and len(X_patient) > 0:
        y_pred = model.predict(X_patient)
        rr_intervals = np.diff(r_peaks) / fs
        avg_hr = int(np.mean(60.0 / rr_intervals)) if len(rr_intervals) > 0 else 0
        normal_count = int(np.sum(y_pred == 'Normal Beat (N)'))
        anomalies_mask = y_pred != 'Normal Beat (N)'
        anomaly_count = int(np.sum(anomalies_mask))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average Heart Rate", f"{avg_hr} BPM")
        c2.metric("Normal Beats (N)", f"{normal_count} beats")
        c3.metric("Abnormal Beats", f"{anomaly_count} beats")
        c4.metric("Classifier Accuracy", f"{model_acc * 100:.1f}%")

        st.subheader("Cardiac Monitor Sound (Normal Beep vs Arrhythmia Alarm)")
        st.audio(generate_ecg_audio(r_peaks, y_pred, len(cleaned_ecg), ecg_fs=fs), format="audio/wav")

        time_axis = np.arange(len(cleaned_ecg)) / fs
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_axis, y=cleaned_ecg, mode='lines', name='Filtered ECG', line=dict(color='#1f77b4', width=1.5)))

        for beat_type in np.unique(y_pred):
            idx = np.where(y_pred == beat_type)
            is_normal = (beat_type == 'Normal Beat (N)')
            fig.add_trace(go.Scatter(
                x=r_peaks[idx] / fs, y=cleaned_ecg[r_peaks[idx]], mode='markers', name=beat_type,
                marker=dict(color=COLOR_MAP.get(beat_type, 'red'), size=8 if is_normal else 14, symbol='circle' if is_normal else 'diamond')
            ))

        fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Amplitude (mV)", height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Precautions & Assessment Based on Result")
        level, hr_status, summary, precautions = get_precautions_and_guidance(avg_hr, y_pred)
        if level == "normal":
            st.success(f"**Status:** {hr_status} â€” {summary}")
        else:
            st.warning(f"**Status:** {summary}")
        for item in precautions:
            st.markdown(f"- {item}")

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Arrhythmia Breakdown")
            counts_df = pd.Series(y_pred).value_counts().reset_index()
            counts_df.columns = ["Beat Classification", "Count"]
            st.dataframe(counts_df, use_container_width=True, hide_index=True)

        with col_right:
            st.subheader("Detected Abnormal Beats Log")
            if anomaly_count > 0:
                log_data = {"Time (sec)": np.round(r_peaks[anomalies_mask] / fs, 2), "Predicted": y_pred[anomalies_mask], "Pre-RR (sec)": np.round(X_patient[anomalies_mask, 0], 3)}
                if y_true is not None:
                    log_data["Ground Truth"] = y_true[anomalies_mask]
                st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
            else:
                st.info("No arrhythmias detected.")

with tab2:
    st.subheader("1. Basis of Classification: ECG Feature Importance")
    fi_df = pd.DataFrame({"ECG Feature": FEATURE_NAMES, "Importance (%)": np.round(model.feature_importances_ * 100, 2)}).sort_values("Importance (%)", ascending=True)
    fig_fi = px.bar(fi_df, x="Importance (%)", y="ECG Feature", orientation="h", text="Importance (%)", color="Importance (%)", color_continuous_scale="Teal")
    fig_fi.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.subheader("2. Confusion Matrix (MIT-BIH Test Split)")
    labels_sorted = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, y_test_pred, labels=labels_sorted)
    st.plotly_chart(px.imshow(cm, text_auto=True, x=labels_sorted, y=labels_sorted, color_continuous_scale="Blues"), use_container_width=True)

    st.subheader("3. Classification Report (Precision, Recall, F1-Score)")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).transpose().round(3), use_container_width=True)
