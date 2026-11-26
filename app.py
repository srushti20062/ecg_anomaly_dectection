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
st.title("ECG SIGNAL ANOMALY DETECTION")

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
    if len(valid_idx) >= 4:
        r_peaks = ann.sample[valid_idx]
        symbols = [ann.symbol[i] for i in valid_idx]
        features, labels, valid_peaks = [], [], []
        window = int(0.1 * fs)
        for i in range(1, len(r_peaks) - 1):
            pk = r_peaks[i]
            if pk - window < 0 or pk + window >= len(cleaned_ecg): continue
            beat_seg = cleaned_ecg[pk - window : pk + window]
            features.append([(r_peaks[i] - r_peaks[i - 1]) / fs, (r_peaks[i + 1] - r_peaks[i]) / fs, cleaned_ecg[pk], np.ptp(beat_seg), np.std(beat_seg)])
            labels.append(BEAT_CLASSES[symbols[i]])
            valid_peaks.append(pk)
        return cleaned_ecg, fs, np.array(valid_peaks), np.array(features), np.array(labels)
    else:
        _, valid_peaks, features = analyze_custom_signal(rec.p_signal[:, 0], fs=fs)
        return cleaned_ecg, fs, valid_peaks, features, None

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
    h, w = img_arr.shape
    if (h / float(w)) > 0.35:
        roi = img_arr[int(h * 0.36):int(h * 0.62), int(w * 0.10):int(w * 0.95)]
    else:
        roi = img_arr[int(h * 0.02):int(h * 0.98), int(w * 0.01):int(w * 0.99)]
    rh, rw = roi.shape
    inverted = np.clip(np.percentile(roi, 92) - roi, 0, None)
    trace_y = []
    for col in range(rw):
        col_vals = inverted[:, col]
        mx = np.max(col_vals)
        if mx > 15:
            y_indices = np.where(col_vals >= 0.80 * mx)
            trace_y.append(np.mean(y_indices))
        else:
            trace_y.append(trace_y[-1] if trace_y else rh / 2.0)

    raw_wave = (rh / 2.0) - np.array(trace_y)
    p98 = np.percentile(np.abs(raw_wave - np.median(raw_wave)), 98) + 1e-6
    raw_wave = ((raw_wave - np.median(raw_wave)) / p98) * 1.05
    return np.interp(np.linspace(0, 1, int(duration_sec * target_fs)), np.linspace(0, 1, len(raw_wave)), raw_wave)

def get_precautions_and_guidance(avg_hr, y_pred):
    unique_abnormal = [b for b in np.unique(y_pred) if b != 'Normal Beat (N)']
    if avg_hr < 60:
        hr_status = "Bradycardia (Low Resting Rate < 60 BPM)"
    elif avg_hr > 100:
        hr_status = "Tachycardia (Elevated Rate > 100 BPM)"
    else:
        hr_status = "Normal Resting Heart Rate (60-100 BPM)"

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
        detected_str = ", ".join(unique_abnormal) if unique_abnormal else "Heart Rate outside 60-100 BPM range"
        summary = f"Irregularities detected: **{detected_str}** | Rate Status: **{hr_status}**."

        if any("PVC" in b or "APB" in b for b in unique_abnormal):
            precautions.append("Limit stimulants such as caffeine, nicotine, energy drinks, and alcohol, which commonly trigger premature beats.")
            precautions.append("Prioritize 7-8 hours of quality sleep and practice stress-reduction techniques.")
        if any("Bundle Branch Block" in b for b in unique_abnormal):
            precautions.append("Schedule a 12-lead ECG and echocardiogram with a cardiologist to evaluate underlying conduction pathways.")
        if avg_hr > 100:
            precautions.append("Rest in a calm seated position, rehydrate slowly, and avoid strenuous exertion until heart rate stabilizes.")
        if avg_hr < 60:
            precautions.append("Monitor for dizziness, lightheadedness, or unusual fatigue, and review any rate-lowering medications with a doctor.")

        precautions.append("Consult a qualified cardiologist or physician for formal clinical evaluation - ‚¬â€seek urgent medical care if accompanied by chest pain, shortness of breath, or fainting.")

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
        selected_record = st.sidebar.selectbox("Select Patient Record:", ["100","101","102","103","104","105","106","107","108","109","111","112","113","114","115","116","117","118","119","121","122","123","124","200","201","202","203","205","207","208","209","210","212","213","214","215","217","219","220","221","222","223","228","230","231","232","233","234"])
        duration_sec = st.sidebar.slider("Signal Duration (seconds):", 10, 60, 25)
        cleaned_ecg, fs, r_peaks, X_patient, y_true = extract_mitbih_features(selected_record, max_samples=duration_sec * fs)
        data_ready = True

    elif "CSV" in input_mode:
        uploaded_csv = st.file_uploader("Upload ECG CSV File:", type=["csv"])
        fs = st.sidebar.number_input("Sampling Rate (Hz):", min_value=100, max_value=1000, value=360)
        if uploaded_csv is not None:
            df_csv = pd.read_csv(uploaded_csv)
            numeric_cols = df_csv.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                ecg_col = st.sidebar.selectbox("Select ECG Signal Column:", numeric_cols)
                cleaned_ecg, r_peaks, X_patient = analyze_custom_signal(df_csv[ecg_col].dropna().values, fs=fs)
                data_ready = True
        else:
            st.info("Upload a CSV file above (or click Download Sample Offline CSV in the sidebar).")

    elif "Image" in input_mode:
        uploaded_img = st.file_uploader("Upload ECG Strip Image (PNG/JPG):", type=["png", "jpg", "jpeg"])
        img_duration = st.sidebar.slider("Estimated Strip Duration (seconds):", 5, 30, 10)
        if uploaded_img is not None:
            st.image(uploaded_img, caption="Uploaded ECG Strip Image", use_container_width=True)
            raw_sig = digitize_ecg_image(uploaded_img, duration_sec=img_duration, target_fs=fs)
            cleaned_ecg, r_peaks, X_patient = analyze_custom_signal(raw_sig, fs=fs)
            data_ready = True
        else:
            st.info("Upload a clear ECG graph image (PNG/JPG) above to digitize and detect arrhythmias.")

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

        import json, base64
        import streamlit.components.v1 as components
        wav_b64 = base64.b64encode(generate_ecg_audio(r_peaks, y_pred, len(cleaned_ecg), ecg_fs=fs)).decode("utf-8")
        step = max(1, len(cleaned_ecg) // 4000)
        t_v = np.round(np.arange(0, len(cleaned_ecg), step) / fs, 4).tolist()
        y_v = np.round(cleaned_ecg[::step], 4).tolist()
        dur = round(len(cleaned_ecg) / fs, 3)
        pks = [{"t": round(float(pk / fs), 4), "y": round(float(cleaned_ecg[pk]), 4), "lbl": lbl, "norm": bool(lbl == 'Normal Beat (N)'), "col": COLOR_MAP.get(lbl, '#ef4444')} for pk, lbl in zip(r_peaks, y_pred)]
        components.html(f"""
        <div style="background:#0f172a;border:1px solid #334155;border-radius:10px;padding:14px;font-family:sans-serif;color:#f8fafc;">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
            <div>
              <button id="pBtn" style="background:#2563eb;color:#fff;border:none;padding:8px 18px;border-radius:6px;font-weight:bold;cursor:pointer;">â–&#9654; Play</button>
              <button id="rBtn" style="background:#334155;color:#fff;border:none;padding:8px 12px;border-radius:6px;margin-left:6px;cursor:pointer;">â†&#8634; Reset</button>
              <span id="tTxt" style="margin-left:10px;font-family:monospace;color:#94a3b8;">0.00s / {dur:.2f}s</span>
            </div>
            <span id="badge" style="padding:5px 12px;border-radius:16px;font-size:13px;font-weight:bold;background:#1e293b;color:#94a3b8;">Click â–&#9654; Play to Start</span>
          </div>
          <canvas id="cv" style="width:100%;height:350px;display:block;background:#020617;border-radius:8px;"></canvas>
          <audio id="aud" src="data:audio/wav;base64,{wav_b64}" preload="auto"></audio>
        </div>
        <script>
          const tD={json.dumps(t_v)},yD={json.dumps(y_v)},pks={json.dumps(pks)},dur={dur};
          const cv=document.getElementById("cv"),ctx=cv.getContext("2d"),aud=document.getElementById("aud");
          const pBtn=document.getElementById("pBtn"),rBtn=document.getElementById("rBtn"),tTxt=document.getElementById("tTxt"),badge=document.getElementById("badge");
          const yMin=Math.min(...yD)-0.15,yMax=Math.max(...yD)+0.2;
          function fit(){{const r=cv.getBoundingClientRect();cv.width=r.width*devicePixelRatio;cv.height=r.height*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);}}
          fit(); window.onresize=()=>{{fit();draw(aud.currentTime);}};
          function draw(ct){{
            const w=cv.getBoundingClientRect().width,h=cv.getBoundingClientRect().height,pL=40,pR=15,pT=15,pB=25,pW=w-pL-pR,pH=h-pT-pB;
            ctx.clearRect(0,0,w,h); ctx.strokeStyle="#1e293b"; ctx.lineWidth=1;
            for(let i=0;i<=4;i++){{let gy=pT+(pH*i)/4;ctx.beginPath();ctx.moveTo(pL,gy);ctx.lineTo(w-pR,gy);ctx.stroke();}}
            ctx.strokeStyle=ct>0?"rgba(56,189,248,0.25)":"#38bdf8"; ctx.lineWidth=ct>0?1.3:1.8; ctx.beginPath();
            for(let i=0;i<tD.length;i++){{let x=pL+(tD[i]/dur)*pW,y=pT+pH-((yD[i]-yMin)/(yMax-yMin))*pH;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}} ctx.stroke();
            if(ct>0){{ctx.save();ctx.strokeStyle="#38bdf8";ctx.shadowColor="#0ea5e9";ctx.shadowBlur=10;ctx.lineWidth=4.2;ctx.beginPath();let s=false;
              for(let i=0;i<tD.length;i++){{if(tD[i]>ct)break;let x=pL+(tD[i]/dur)*pW,y=pT+pH-((yD[i]-yMin)/(yMax-yMin))*pH;if(!s){{ctx.moveTo(x,y);s=true;}}else ctx.lineTo(x,y);}} ctx.stroke();ctx.restore();}}
            let hitB=null;
            for(const p of pks){{let px=pL+(p.t/dur)*pW,py=pT+pH-((p.y-yMin)/(yMax-yMin))*pH,hit=Math.abs(ct-p.t)<0.14&&ct>0;if(hit)hitB=p;
              ctx.fillStyle=p.col;ctx.strokeStyle="#fff";ctx.lineWidth=hit?2.5:1;
              if(p.norm){{ctx.beginPath();ctx.arc(px,py,hit?7.5:4.5,0,6.28);ctx.fill();if(hit)ctx.stroke();}}
              else{{let r=hit?11:7;ctx.beginPath();ctx.moveTo(px,py-r);ctx.lineTo(px+r,py);ctx.lineTo(px,py+r);ctx.lineTo(px-r,py);ctx.closePath();ctx.fill();ctx.stroke();}}}}
            let cx=pL+(Math.min(ct,dur)/dur)*pW;ctx.strokeStyle="#facc15";ctx.lineWidth=2.5;ctx.beginPath();ctx.moveTo(cx,pT);ctx.lineTo(cx,h-pB);ctx.stroke();
            tTxt.textContent=ct.toFixed(2)+"s / "+dur.toFixed(2)+"s";
            if(hitB){{badge.style.background=hitB.norm?"#14532d":"#7f1d1d";badge.style.color=hitB.norm?"#86efac":"#fca5a5";badge.textContent=hitB.norm?"â— Normal ("+hitB.t.toFixed(2)+"s)":"âš  ALARM: "+hitB.lbl;}}
          }}
          function loop(){{draw(aud.currentTime);if(!aud.paused&&!aud.ended)requestAnimationFrame(loop);else if(aud.ended)pBtn.textContent="â–&#9654; Play";}}
          pBtn.onclick=()=>{{if(aud.paused){{if(aud.ended)aud.currentTime=0;aud.play();pBtn.textContent="&#9208; Pause";requestAnimationFrame(loop);}}else{{aud.pause();pBtn.textContent="â–&#9654; Play";}}}};
          rBtn.onclick=()=>{{aud.pause();aud.currentTime=0;pBtn.textContent="â–&#9654; Play";badge.style.background="#1e293b";badge.style.color="#94a3b8";badge.textContent="Click â–&#9654; Play to Start";draw(0);}};
          draw(0);
        </script>
        """, height=430)

        st.subheader("Precautions & Assessment Based on Result")
        level, hr_status, summary, precautions = get_precautions_and_guidance(avg_hr, y_pred)
        if level == "normal":
            st.success(f"**Status:** {hr_status} Ã¢â‚¬â€ {summary}")
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
