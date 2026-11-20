import streamlit as st
import wfdb
import neurokit2 as nk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

def extract_beats_and_features(record_id, max_samples=None):
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

        r_amp = cleaned_ecg[pk]
        qrs_ptp = np.ptp(beat_seg)
        qrs_std = np.std(beat_seg)

        features.append([pre_rr, post_rr, r_amp, qrs_ptp, qrs_std])
        labels.append(BEAT_CLASSES[symbols[i]])
        valid_peaks.append(pk)

    return cleaned_ecg, fs, np.array(valid_peaks), np.array(features), np.array(labels)

@st.cache_resource
def train_arrhythmia_model():
    train_records = ["100", "106", "118", "119", "200", "208"]
    X_all, y_all = [], []
    for rec_id in train_records:
        _, _, _, X, y = extract_beats_and_features(rec_id, max_samples=360 * 180)
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

with st.spinner("Training arrhythmia classifier on MIT-BIH ground-truth labels..."):
    model, model_acc, y_test, y_test_pred = train_arrhythmia_model()

# Sidebar Controls
st.sidebar.header("1. Patient Record Settings")
selected_record = st.sidebar.selectbox(
    "Select Test Patient Record:",
    ["106", "119", "200", "208", "100", "118"]
)
duration_sec = st.sidebar.slider("Signal Duration (seconds):", 10, 60, 25)
st.sidebar.success(f"Validation Accuracy: **{model_acc * 100:.1f}%**")

# Create Two Tabs
tab1, tab2 = st.tabs(["🩺 Live ECG Monitor", "📊 Model Evaluation (Report & Viva)"])

with tab1:
    cleaned_ecg, fs, r_peaks, X_patient, y_true = extract_beats_and_features(
        selected_record, max_samples=duration_sec * fs
    )
    y_pred = model.predict(X_patient)

    rr_intervals = np.diff(r_peaks) / fs
    avg_hr = int(np.mean(60.0 / rr_intervals))
    total_beats = len(r_peaks)
    anomalies_mask = y_pred != 'Normal Beat (N)'
    anomaly_count = int(np.sum(anomalies_mask))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average Heart Rate", f"{avg_hr} BPM")
    c2.metric("Total Beat Count", f"{total_beats} beats")
    c3.metric("Detected Arrhythmias", f"{anomaly_count} abnormal")
    c4.metric("Classifier Accuracy", f"{model_acc * 100:.1f}%")

    time_axis = np.arange(len(cleaned_ecg)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis, y=cleaned_ecg, mode='lines',
        name='Filtered ECG', line=dict(color='#1f77b4', width=1.4)
    ))

    for beat_type in np.unique(y_pred):
        idx = np.where(y_pred == beat_type)
        is_normal = (beat_type == 'Normal Beat (N)')
        fig.add_trace(go.Scatter(
            x=r_peaks[idx] / fs, y=cleaned_ecg[r_peaks[idx]],
            mode='markers', name=beat_type,
            marker=dict(
                color=COLOR_MAP.get(beat_type, 'red'),
                size=7 if is_normal else 13,
                symbol='circle' if is_normal else 'diamond'
            )
        ))

    fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Amplitude (mV)", height=450, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns()
    with col_left:
        st.subheader("Arrhythmia Breakdown")
        counts_df = pd.Series(y_pred).value_counts().reset_index()
        counts_df.columns = ["Beat Classification", "Count"]
        st.dataframe(counts_df, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Detected Abnormal Beats Log")
        if anomaly_count > 0:
            log_df = pd.DataFrame({
                "Time (sec)": np.round(r_peaks[anomalies_mask] / fs, 2),
                "Predicted": y_pred[anomalies_mask],
                "Ground Truth": y_true[anomalies_mask],
                "Pre-RR (sec)": np.round(X_patient[anomalies_mask, 0], 3)
            })
            st.dataframe(log_df, use_container_width=True, hide_index=True)
        else:
            st.info("No arrhythmias detected in this window.")

with tab2:
    st.subheader("Confusion Matrix (MIT-BIH Test Split)")
    labels_sorted = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, y_test_pred, labels=labels_sorted)
    fig_cm = px.imshow(
        cm, text_auto=True, x=labels_sorted, y=labels_sorted,
        labels=dict(x="Predicted Class", y="Actual MIT-BIH Class", color="Beat Count"),
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Classification Report (Precision, Recall, F1-Score)")
    report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)
