import io
import streamlit as st
import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

st.set_page_config(page_title="Roblox Audio Safety Analyzer", page_icon="🎵", layout="wide")

# ============================== AUDIO FUNCTIONS ==============================

def measure_true_peak_db(y, oversample=4):
    """True Peak (dBTP) por sobremuestreo 4x, aproximación a ITU-R BS.1770."""
    y_os = resample_poly(y, oversample, 1)
    peak = np.max(np.abs(y_os))
    return 20 * np.log10(peak) if peak > 0 else -100.0


def apply_gain_db(y, gain_db):
    return y * (10 ** (gain_db / 20.0))


def compute_spectrum_bands(y, sr, n_fft=2048):
    """Devuelve el espectro normalizado y los valores promedio por banda."""
    S = np.abs(librosa.stft(y, n_fft=n_fft))
    S_norm = S / (np.max(S) + 1e-6)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mean_spectrum_db = 20 * np.log10(np.mean(S_norm, axis=1) + 1e-6)

    sub_bass_mask = (freqs >= 20) & (freqs <= 60)
    mids_mask = (freqs >= 500) & (freqs <= 2000)
    ultrasonic_mask = freqs >= 17000

    sub_bass_val = np.mean(mean_spectrum_db[sub_bass_mask]) if np.any(sub_bass_mask) else -100.0
    mids_val = np.mean(mean_spectrum_db[mids_mask]) if np.any(mids_mask) else -100.0
    ultrasonic_val = np.mean(mean_spectrum_db[ultrasonic_mask]) if np.any(ultrasonic_mask) else -100.0
    bass_dominance = sub_bass_val - mids_val

    return {
        "freqs": freqs, "mean_spectrum_db": mean_spectrum_db,
        "sub_bass_mask": sub_bass_mask, "mids_mask": mids_mask, "ultrasonic_mask": ultrasonic_mask,
        "sub_bass_val": sub_bass_val, "mids_val": mids_val, "ultrasonic_val": ultrasonic_val,
        "bass_dominance": bass_dominance,
    }


def apply_spectral_shaping(y, sr, sub_bass_limits, ultrasonic_limit_hz, bass_reduction_db, ultrasonic_reduction_db, n_fft=8192):
    """
    Reduce por EQ (en el dominio de la frecuencia) las bandas de subgraves
    y/o ultrasónicos, preservando la fase (solo se escala la magnitud).
    Usa un n_fft grande para tener resolución suficiente en 20-60 Hz,
    donde con un n_fft chico caben apenas 1-2 bins y cualquier suavizado
    diluye el recorte por completo.
    """
    if bass_reduction_db <= 0 and ultrasonic_reduction_db <= 0:
        return y

    freqs_hi = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    sub_bass_mask_hi = (freqs_hi >= sub_bass_limits[0]) & (freqs_hi <= sub_bass_limits[1])
    ultrasonic_mask_hi = freqs_hi >= ultrasonic_limit_hz

    gain = np.ones_like(freqs_hi)
    if bass_reduction_db > 0:
        factor = 10 ** (-bass_reduction_db / 20.0)
        gain[sub_bass_mask_hi] = np.minimum(gain[sub_bass_mask_hi], factor)
    if ultrasonic_reduction_db > 0:
        factor_u = 10 ** (-ultrasonic_reduction_db / 20.0)
        gain[ultrasonic_mask_hi] = np.minimum(gain[ultrasonic_mask_hi], factor_u)

    # suavizado ligero solo para evitar un corte 100% abrupto entre bins vecinos
    gain = uniform_filter1d(gain, size=3)

    D = librosa.stft(y, n_fft=n_fft)
    D_shaped = D * gain[:, None]
    y_shaped = librosa.istft(D_shaped, length=len(y))
    return y_shaped


def full_transform(y, sr, meter, target_lufs, target_tp_db, bass_dominance_limit, ultrasonic_limit, n_fft=2048):
    """Pipeline completo: 1) EQ de subgraves/ultrasónicos  2) ganancia a LUFS objetivo  3) techo True Peak."""
    bands_before = compute_spectrum_bands(y, sr, n_fft)
    bass_reduction_db = max(0.0, bands_before["bass_dominance"] - bass_dominance_limit)
    ultrasonic_reduction_db = max(0.0, bands_before["ultrasonic_val"] - ultrasonic_limit)

    y_eq = apply_spectral_shaping(
        y, sr, sub_bass_limits=(20, 60), ultrasonic_limit_hz=17000,
        bass_reduction_db=bass_reduction_db, ultrasonic_reduction_db=ultrasonic_reduction_db,
    )

    current_lufs = meter.integrated_loudness(y_eq)
    if not np.isfinite(current_lufs):
        current_lufs = -70.0
    gain_db = target_lufs - current_lufs
    y_gain = apply_gain_db(y_eq, gain_db)

    tp_after_gain = measure_true_peak_db(y_gain)
    if tp_after_gain > target_tp_db:
        y_gain = apply_gain_db(y_gain, -(tp_after_gain - target_tp_db))

    y_final = np.clip(y_gain, -1.0, 1.0)
    return y_final, bass_reduction_db, ultrasonic_reduction_db


# ============================== MODERN STYLES ==============================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp {
        background: radial-gradient(circle at top left, rgba(98, 103, 255, 0.2), transparent 30%),
                    radial-gradient(circle at bottom right, rgba(0, 214, 170, 0.18), transparent 25%),
                    #0b1020;
        color: #e5ecff;
        font-family: 'Inter', sans-serif;
    }
    .modern-shell {
        position: relative;
        overflow: hidden;
        border-radius: 22px;
        margin-bottom: 18px;
        background: linear-gradient(135deg, rgba(19,23,38,0.9), rgba(17,19,33,0.9));
        border: 1px solid rgba(140, 169, 255, 0.2);
        box-shadow: 0 30px 50px rgba(0, 0, 0, 0.25);
        padding: 24px 22px;
    }
    .aurora {
        position: absolute;
        border-radius: 50%;
        filter: blur(90px);
        opacity: 0.38;
        animation: floatPulse 9s ease-in-out infinite alternate;
    }
    .aurora-one { width: 180px; height: 180px; background: #7c6cff; top: -40px; right: 8%; }
    .aurora-two { width: 240px; height: 240px; background: #17c6c7; bottom: -90px; left: 6%; animation-delay: 2s; }
    @keyframes floatPulse {
        0% { transform: translateY(0px) scale(0.96); }
        100% { transform: translateY(-18px) scale(1.08); }
    }
    .hero-card {
        position: relative;
        z-index: 1;
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 7px 12px; border-radius: 999px;
        background: rgba(124, 108, 255, 0.12); border: 1px solid rgba(124, 108, 255, 0.25);
        color: #c2d0ff; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    }
    h1 {
        margin: 16px 0 8px 0; font-size: clamp(2rem, 4vw, 3.1rem); line-height: 1.08;
        font-weight: 800; letter-spacing: -0.04em; color: #f5f7ff;
    }
    .hero-subtitle {
        max-width: 760px; color: #b5c3e5; font-size: 1rem; line-height: 1.6; margin: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(24, 29, 43, 0.95), rgba(15, 19, 31, 0.95));
        border: 1px solid rgba(140, 169, 255, 0.18); border-radius: 18px; padding: 18px; margin-bottom: 15px;
        box-shadow: 0 12px 26px rgba(0,0,0,0.18); transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        position: relative; overflow: hidden;
    }
    .metric-card::before {
        content: ""; position: absolute; inset: 0 auto auto 0; width: 100%; height: 2px; background: linear-gradient(90deg, #6ea8fe, transparent);
    }
    .metric-card:hover { transform: translateY(-3px); border-color: rgba(110,168,254,0.7); box-shadow: 0 16px 36px rgba(89, 117, 255, 0.18); }
    .status-badge {
        font-size: 0.75rem; font-weight: 700; padding: 6px 10px; border-radius: 999px;
        display: inline-block; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.06em;
    }
    .badge-success { background-color: rgba(46, 160, 67, 0.18); color: #6ee7a8; border: 1px solid rgba(46, 160, 67, 0.35); }
    .badge-warning { background-color: rgba(210, 153, 34, 0.18); color: #ffca6d; border: 1px solid rgba(210, 153, 34, 0.35); }
    .badge-danger { background-color: rgba(248, 81, 73, 0.18); color: #ff8b8b; border: 1px solid rgba(248, 81, 73, 0.35); }
    .disclaimer-box {
        background: rgba(110, 168, 254, 0.07); border: 1px solid rgba(110, 168, 254, 0.18);
        border-left: 4px solid #6ea8fe; border-radius: 16px; padding: 14px 18px; font-size: 0.9rem;
        color: #cfdaf7; margin-bottom: 20px; box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .stAlert { border-radius: 16px; }
    div[data-testid="stSidebar"] { background: rgba(12,15,28,0.9); }
    [data-testid="stFileUploaderDropzone"] { border-radius: 18px; border: 1px dashed rgba(128,155,255,0.5) !important; }
    .stDownloadButton > button, .stButton > button {
        border-radius: 12px; font-weight: 700; transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .stDownloadButton > button:hover, .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(107, 124, 255, 0.25); }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="modern-shell">
  <div class="aurora aurora-one"></div>
  <div class="aurora aurora-two"></div>
  <div class="hero-card">
    <div class="hero-badge">🎵 Audio safety studio</div>
    <h1>Roblox Audio Safety Analyzer</h1>
    <p class="hero-subtitle">Diagnose loudness, true peak, sub-bass excess, and ultrasonic artifacts before uploading to Roblox, with a cleaner workflow and a more modern experience.</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.components.v1.html("""
<script>
  let audioCtx = null;

  function ensureAudio() {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return null;
    if (!audioCtx) {
      audioCtx = new AudioCtx();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    return audioCtx;
  }

  function playTone({ frequency = 540, duration = 0.08, gain = 0.02, type = 'sine' } = {}) {
    try {
      const ctx = ensureAudio();
      if (!ctx) return;
      const osc = ctx.createOscillator();
      const gainNode = ctx.createGain();
      osc.type = type;
      osc.frequency.value = frequency;
      gainNode.gain.value = gain;
      osc.connect(gainNode);
      gainNode.connect(ctx.destination);
      osc.start();
      const stopAt = ctx.currentTime + duration;
      gainNode.gain.setValueAtTime(gain, ctx.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.0001, stopAt);
      osc.stop(stopAt);
    } catch (e) {}
  }

  function bindSoundTo(selector, hoverFreq, clickFreq, type = 'sine') {
    document.querySelectorAll(selector).forEach((el) => {
      if (el.dataset.soundBound === 'true') return;
      el.dataset.soundBound = 'true';

      el.addEventListener('pointerover', () => {
        playTone({ frequency: hoverFreq, duration: 0.06, gain: 0.012, type });
      }, { passive: true });

      el.addEventListener('pointerdown', () => {
        playTone({ frequency: clickFreq, duration: 0.12, gain: 0.02, type: 'triangle' });
      }, { passive: true });
    });
  }

  function bindAll() {
    bindSoundTo('button', 520, 700, 'sine');
    bindSoundTo('[data-testid="stFileUploaderDropzone"]', 430, 620, 'triangle');
    bindSoundTo('.metric-card', 610, 780, 'triangle');
    bindSoundTo('input[type="checkbox"]', 560, 740, 'square');
  }

  window.addEventListener('pointerdown', ensureAudio, { once: true });
  window.addEventListener('pointerover', ensureAudio, { once: true });

  const observer = new MutationObserver(() => {
    bindAll();
  });

  observer.observe(document.body, { childList: true, subtree: true });
  bindAll();
</script>
""", height=0)

st.markdown("""
<div class="disclaimer-box">
⚠️ <b>Roblox does not publish the exact thresholds of its moderation system</b>, so no external tool — including this one — can guarantee a 100% accurate result. This app uses real industry standards (ITU-R BS.1770 / EBU R128) and lets you set your own objectives below. Roblox can also reject audio for reasons no loudness tool can fix: copyright, voice/crying, or explicit content.
</div>
""", unsafe_allow_html=True)

# ============================== SIDEBAR: OBJETIVOS ==============================
DEFAULTS = {
    "lufs_target": -14.0,
    "tp_limit": -1.0,
    "momentary_limit": -8.0,
    "bass_dominance_limit": 26.0,
    "ultrasonic_limit": -30.0,
    "dynamic_range_min": 6.0,
}

with st.sidebar:
    st.header("🎚️ Audio targets")
    strict_mode = st.toggle("🔒 Strict mode (recommended values)", value=True,
                             help="Locks the threshold values to the recommended industry-safe defaults so you do not accidentally relax the analysis and get a false green light.")

    if not strict_mode:
        st.warning("⚠️ You are in custom mode. A result marked as 'safe' with looser thresholds does not mean Roblox will accept it — it only changes what this app displays, not Roblox's actual moderation rules.")

    st.caption("These values are used to: (1) determine whether your audio is at risk, and (2) set the target for the 'Generate corrected version' step.")

    lufs_target = st.slider(
        "Integrated LUFS target", -30.0, -6.0, DEFAULTS["lufs_target"], 0.5, disabled=strict_mode,
        help="Measures the perceived average loudness over the whole clip, not just the peak. -14 LUFS is the Spotify/YouTube reference; -23 LUFS is the stricter EBU R128 broadcast standard. When transforming, the app raises or lowers gain across the full track until it lands near this value. If you set it too high (for example -6), the track will be louder and more likely to hit the True Peak ceiling below."
    )
    tp_limit = st.slider(
        "Maximum True Peak (dBTP)", -6.0, 0.0, DEFAULTS["tp_limit"], 0.1, disabled=strict_mode,
        help="The real peak of the waveform, including inter-sample peaks that appear when audio is reconstructed on a DAC or re-encoded. If True Peak is too high (close to 0 dBTP), the audio can distort on playback or re-encoding. When applying a correction, if boosting volume to hit the LUFS target would violate this limit, the app reduces gain as needed to respect it. That is why the final LUFS may not land exactly on the target."
    )
    momentary_limit = st.slider(
        "Maximum momentary loudness (LUFS, bursts)", -20.0, -3.0, DEFAULTS["momentary_limit"], 0.5, disabled=strict_mode,
        help="Loudness measured in short 400 ms windows instead of the full file. This catches sudden bursts like a shout or impact that may trigger 'Disruptive Audio' even when the overall average is moderate. This value is used only for diagnosis; the auto-correction does not apply a separate burst limiter, only the general True Peak ceiling."
    )
    bass_dominance_limit = st.slider(
        "Maximum bass dominance (dB vs mids)", 10.0, 35.0, DEFAULTS["bass_dominance_limit"], 1.0, disabled=strict_mode,
        help="Compares how strong the 20-60 Hz sub-bass band is relative to the 500-2000 Hz midrange band, where most vocals and instruments live. A high value means exaggerated low-end like a car subwoofer. During transformation, if your file exceeds this limit, the app applies EQ to reduce the 20-60 Hz band until the dominance is under the selected value without changing the rest of the spectrum."
    )
    ultrasonic_limit = st.slider(
        "Maximum ultrasonic noise (dB, >17kHz)", -50.0, -10.0, DEFAULTS["ultrasonic_limit"], 1.0, disabled=strict_mode,
        help="Energy above 17 kHz, often inaudible to most listeners but sometimes caused by compression artifacts or faulty microphone capture. If your file exceeds this limit, the app applies a filter that attenuates frequencies above 17 kHz — it is generally safe to remove because most people cannot hear it."
    )
    dynamic_range_min = st.slider(
        "Minimum dynamic range (dB)", 1.0, 15.0, DEFAULTS["dynamic_range_min"], 0.5, disabled=strict_mode,
        help="The difference between the loudest peak and the average level. A low value means the audio is heavily compressed or flattened, sounding like a constant wall of noise without variation. This is for DIAGNOSTIC use only. If the original is already heavily compressed, this app cannot recover the lost dynamic range; that information is no longer in the signal. The real fix is to start from a less compressed source."
    )

    if strict_mode:
        lufs_target = DEFAULTS["lufs_target"]
        tp_limit = DEFAULTS["tp_limit"]
        momentary_limit = DEFAULTS["momentary_limit"]
        bass_dominance_limit = DEFAULTS["bass_dominance_limit"]
        ultrasonic_limit = DEFAULTS["ultrasonic_limit"]
        dynamic_range_min = DEFAULTS["dynamic_range_min"]

uploaded_file = st.file_uploader("Upload your audio file (MP3 / WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("Running loudness, true peak, and spectrum scan..."):
        y, sr = librosa.load(uploaded_file, sr=None, mono=True)

        meter = pyln.Meter(sr)
        try:
            integrated_lufs = meter.integrated_loudness(y)
        except Exception:
            integrated_lufs = -70.0
        if not np.isfinite(integrated_lufs):
            integrated_lufs = -70.0

        win_samples = int(0.4 * sr)
        hop_samples = int(0.1 * sr)
        momentary_values = []
        if len(y) > win_samples:
            for start in range(0, len(y) - win_samples, hop_samples):
                chunk = y[start:start + win_samples]
                try:
                    val = meter.integrated_loudness(chunk)
                    if np.isfinite(val):
                        momentary_values.append(val)
                except Exception:
                    pass
        max_momentary = max(momentary_values) if momentary_values else integrated_lufs

        true_peak_db = measure_true_peak_db(y)
        sample_peak_amp = np.max(np.abs(y))
        sample_peak_db = 20 * np.log10(sample_peak_amp) if sample_peak_amp > 0 else -100.0
        dynamic_range = true_peak_db - integrated_lufs

        bands = compute_spectrum_bands(y, sr)
        freqs = bands["freqs"]
        mean_spectrum_db = bands["mean_spectrum_db"]
        bass_dominance = bands["bass_dominance"]
        ultrasonic_val = bands["ultrasonic_val"]

        reasons = []
        is_no_apto = False
        is_riesgo = False

        if true_peak_db > tp_limit:
            is_no_apto = True
            reasons.append(f"True Peak of {true_peak_db:.1f} dBTP exceeds the limit of {tp_limit:.1f} dBTP.")
        if max_momentary > momentary_limit:
            is_no_apto = True
            reasons.append(f"Momentary loudness burst of {max_momentary:.1f} LUFS (threshold: {momentary_limit:.1f} LUFS).")
        if integrated_lufs > lufs_target:
            is_riesgo = True
            reasons.append(f"Integrated loudness of {integrated_lufs:.1f} LUFS is above the target of {lufs_target:.1f} LUFS.")
        if bass_dominance > bass_dominance_limit:
            is_no_apto = True
            reasons.append("Excessive sub-bass presence relative to the midrange frequencies.")
        if ultrasonic_val > ultrasonic_limit:
            is_riesgo = True
            reasons.append("Ultrasonic noise (>17kHz) detected in the mix.")
        if dynamic_range < dynamic_range_min:
            is_riesgo = True
            reasons.append("Very compressed dynamic range (constant 'wall of noise' sound).")

        st.subheader("📊 Diagnostic result")
        if is_no_apto:
            st.error("❌ **HIGH RISK — parameters fall outside industry-safe standards**")
        elif is_riesgo:
            st.warning("⚠️ **MODERATE RISK — some parameters are near the limit**")
        else:
            st.success("✅ **WITHIN SAFE RANGE — loudness, peaks, and spectrum are inside healthy limits**")
            st.caption("This means the audio follows solid mastering practices for LUFS, True Peak, and spectral balance. **It is not a guarantee that Roblox will accept it** — the moderation system also analyzes sound content such as voices, screams, and perceptual distortion in ways this app cannot fully replicate.")

        if reasons:
            st.write("**Detected issues:**")
            for r in reasons:
                st.markdown(f"- {r}")

        with st.expander("📌 Rejection causes this tool cannot detect or fix"):
            st.write("""
            * **Copyright / Content ID:** registered songs, remixes, or samples can be removed even when the audio is technically clean.
            * **Voice, screams, or profanity:** moderation can flag spoken or sung content, harsh screams, or explicit language.
            * **Context of the sound:** gunshots, horror screams, sirens, or similar effects may be flagged based on their nature even when volume is moderate.
            """)

        st.write("### 🔍 Technical parameters")
        col1, col2, col3 = st.columns(3)

        with col1:
            if true_peak_db > tp_limit:
                badge, rec = '<span class="status-badge badge-danger">True Peak exceeded</span>', f"Reduce gain to keep True Peak at or below {tp_limit:.1f} dBTP."
            elif max_momentary > momentary_limit:
                badge, rec = '<span class="status-badge badge-danger">Burst detected</span>', "Apply limiting to smooth sudden transient spikes."
            else:
                badge, rec = '<span class="status-badge badge-success">Safe</span>', f"Sample Peak: {sample_peak_db:.1f} dBFS"
            st.markdown(f"""<div class="metric-card">{badge}<h4>True Peak</h4>
            <h2 style="color:#58a6ff;">{true_peak_db:.1f} <small style="font-size:14px;">dBTP</small></h2>
            <hr style="border-color:#30363d; margin:10px 0;">
            <p style="font-size:0.85rem;"><b>Limit:</b> ≤ {tp_limit:.1f} dBTP</p>
            <p style="font-size:0.85rem;">{rec}</p></div>""", unsafe_allow_html=True)

        with col2:
            if integrated_lufs > lufs_target:
                badge, rec = '<span class="status-badge badge-warning">Too loud</span>', "Reduce the overall gain or use less aggressive compression."
            else:
                badge, rec = '<span class="status-badge badge-success">In range</span>', f"Max momentary: {max_momentary:.1f} LUFS"
            st.markdown(f"""<div class="metric-card">{badge}<h4>Integrated Loudness (LUFS)</h4>
            <h2 style="color:#a5d6ff;">{integrated_lufs:.1f} <small style="font-size:14px;">LUFS</small></h2>
            <hr style="border-color:#30363d; margin:10px 0;">
            <p style="font-size:0.85rem;"><b>Target:</b> ≤ {lufs_target:.1f} LUFS</p>
            <p style="font-size:0.85rem;">{rec}</p></div>""", unsafe_allow_html=True)

        with col3:
            if dynamic_range < dynamic_range_min:
                badge, rec = '<span class="status-badge badge-warning">Low dynamic range</span>', "Not recoverable with post-processing; requires a less compressed source."
            else:
                badge, rec = '<span class="status-badge badge-success">Healthy dynamics</span>', "The audio breathes correctly."
            st.markdown(f"""<div class="metric-card">{badge}<h4>Dynamic Range</h4>
            <h2 style="color:#d2a8ff;">{dynamic_range:.1f} <small style="font-size:14px;">dB</small></h2>
            <hr style="border-color:#30363d; margin:10px 0;">
            <p style="font-size:0.85rem;"><b>Minimum:</b> {dynamic_range_min:.1f} dB</p>
            <p style="font-size:0.85rem;">{rec}</p></div>""", unsafe_allow_html=True)

        col4, col5 = st.columns(2)
        with col4:
            badge = '<span class="status-badge badge-danger">Exceeded</span>' if bass_dominance > bass_dominance_limit else '<span class="status-badge badge-success">Balanced</span>'
            st.markdown(f"""<div class="metric-card">{badge}<h4>Bass Dominance</h4>
            <h2 style="color:#f0883e;">+{bass_dominance:.1f} <small style="font-size:14px;">dB vs mids</small></h2>
            <hr style="border-color:#30363d; margin:10px 0;">
            <p style="font-size:0.85rem;"><b>Limit:</b> ≤ {bass_dominance_limit:.1f} dB</p></div>""", unsafe_allow_html=True)
        with col5:
            badge = '<span class="status-badge badge-danger">Exceeded</span>' if ultrasonic_val > ultrasonic_limit else '<span class="status-badge badge-success">Clean</span>'
            st.markdown(f"""<div class="metric-card">{badge}<h4>Ultrasonic Noise</h4>
            <h2 style="color:#79c0ff;">{ultrasonic_val:.1f} <small style="font-size:14px;">dB</small></h2>
            <hr style="border-color:#30363d; margin:10px 0;">
            <p style="font-size:0.85rem;"><b>Limit:</b> ≤ {ultrasonic_limit:.1f} dB</p></div>""", unsafe_allow_html=True)

        # ============================== TRANSFORMATION ==============================
        st.write("### 🎛️ Generate corrected version")
        st.caption(
            "Applies, in this order: 1) EQ that reduces sub-bass and/or ultrasonic energy when they exceed the sidebar limits, "
            "2) gain adjustment to move the LUFS toward the target, 3) a True Peak ceiling that trims gain if needed. "
            "Dynamic range is not corrected — if the file is already heavily compressed, there is no way to recover the lost variation."
        )

        if st.button("🎚️ Generate corrected version"):
            with st.spinner("Applying EQ, gain, and True Peak ceiling..."):
                y_final, bass_cut, ultrasonic_cut = full_transform(
                    y, sr, meter, lufs_target, tp_limit, bass_dominance_limit, ultrasonic_limit
                )

                final_lufs = meter.integrated_loudness(y_final)
                if not np.isfinite(final_lufs):
                    final_lufs = -70.0
                final_tp = measure_true_peak_db(y_final)
                final_bands = compute_spectrum_bands(y_final, sr)

                buf = io.BytesIO()
                sf.write(buf, y_final, sr, format="WAV", subtype="PCM_16")
                buf.seek(0)

            st.success("Done — this is how the audio looks after correction:")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Final LUFS", f"{final_lufs:.1f}", f"target {lufs_target:.1f}")
            r2.metric("Final True Peak", f"{final_tp:.1f} dBTP", f"limit {tp_limit:.1f}")
            r3.metric("Bass reduced", f"-{bass_cut:.1f} dB" if bass_cut > 0 else "unchanged")
            r4.metric("Ultrasonic reduced", f"-{ultrasonic_cut:.1f} dB" if ultrasonic_cut > 0 else "unchanged")

            if final_bands["bass_dominance"] - bass_dominance_limit > 0.5:
                st.warning("Bass dominance is still slightly above the limit; it may need another pass or a more aggressive EQ profile.")

            st.audio(buf, format="audio/wav")
            st.download_button(
                "⬇️ Download corrected audio (.wav)",
                data=buf, file_name="audio_corrected.wav", mime="audio/wav",
            )
            st.caption("The file is exported as WAV to preserve quality. If you need MP3 for Roblox, convert it externally with Audacity or another converter before uploading.")

        # ============================== FREQUENCY SPECTRUM ==============================
        st.write("### 📈 Normalized frequency spectrum")
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 3.8))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#0d1117')
        ax.plot(freqs, mean_spectrum_db, color='#bc8cff', linewidth=1.8, label='Respuesta de Frecuencia')
        ax.set_xscale('log')
        ax.set_xlim(20, sr // 2)
        ax.set_ylim(-90, 5)
        ax.axvspan(20, 60, color='#f85149', alpha=0.18, label='Zona Subgraves (<60Hz)')
        ax.axvspan(500, 2000, color='#3fb950', alpha=0.10, label='Zona Medios (Referencia)')
        ax.axvspan(17000, sr // 2, color='#d29922', alpha=0.12, label='Zona Ultrasónica (>17kHz)')
        ax.set_title("Frecuencia (Hz) vs Amplitud (dBFS)", fontsize=11, color='#c9d1d9', pad=12)
        ax.set_xlabel("Frecuencia (Hz)", fontsize=9, color='#8b949e')
        ax.set_ylabel("Amplitud Relativa (dB)", fontsize=9, color='#8b949e')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.grid(True, which="both", ls="--", color='#21262d', alpha=0.7)
        ax.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=8)
        st.pyplot(fig)
