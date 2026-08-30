import streamlit as st
import numpy as np
import librosa
import pyloudnorm as pyln
from scipy.signal import resample_poly
import matplotlib.pyplot as plt

st.set_page_config(page_title="Roblox Audio Moderation Checker Pro", page_icon="🎵", layout="wide")

# --- ESTILOS CSS CUSTOM ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-3px); border-color: #58a6ff; }
    .status-badge {
        font-size: 0.85rem; font-weight: 700; padding: 4px 10px; border-radius: 20px;
        display: inline-block; margin-bottom: 8px; text-transform: uppercase;
    }
    .badge-success { background-color: rgba(46, 160, 67, 0.2); color: #3fb950; border: 1px solid #2ea643; }
    .badge-warning { background-color: rgba(210, 153, 34, 0.2); color: #d29922; border: 1px solid #bb8009; }
    .badge-danger { background-color: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid #f85149; }
    .disclaimer-box {
        background: rgba(88, 166, 255, 0.08); border: 1px solid #30363d; border-left: 4px solid #58a6ff;
        border-radius: 8px; padding: 14px 18px; font-size: 0.88rem; color: #8b949e; margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Roblox Audio Moderation Checker Pro")
st.caption("Análisis técnico de LUFS, True Peak y espectro para reducir el riesgo de rechazo por *Disruptive Audio*.")

st.markdown("""
<div class="disclaimer-box">
⚠️ <b>Roblox no publica los umbrales exactos de su sistema de moderación</b>, por lo que ninguna herramienta externa
—esta incluida— puede garantizar un resultado 100% preciso. Este checker usa estándares reales de la industria
(ITU-R BS.1770 / EBU R128: LUFS y True Peak) en vez de aproximaciones caseras, y te deja ajustar los umbrales
en la barra lateral. Úsalo como guía técnica, no como veredicto oficial. Además, Roblox también rechaza audio por
motivos que esta herramienta <b>no puede detectar</b>: copyright, letras/voces inapropiadas, gritos, o contenido explícito.
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: UMBRALES AJUSTABLES ---
with st.sidebar:
    st.header("⚙️ Umbrales de análisis")
    st.caption("Valores por defecto basados en referencias de la industria (Spotify/YouTube ≈ -14 LUFS, EBU R128 ≈ -23 LUFS, True Peak ≤ -1 dBTP). Ajusta si tienes evidencia de que tus audios pasan o fallan con otros valores.")
    lufs_target = st.slider("LUFS integrado máximo recomendado", -30.0, -6.0, -14.0, 0.5)
    tp_limit = st.slider("True Peak máximo (dBTP)", -6.0, 0.0, -1.0, 0.1)
    momentary_limit = st.slider("Loudness momentáneo máx. (LUFS, ráfagas)", -20.0, -3.0, -8.0, 0.5)
    bass_dominance_limit = st.slider("Dominancia de subgraves máx. (dB vs medios)", 10.0, 35.0, 26.0, 1.0)
    ultrasonic_limit = st.slider("Ruido ultrasónico máx. (dB, >17kHz)", -50.0, -10.0, -30.0, 1.0)
    dynamic_range_min = st.slider("Rango dinámico mínimo (dB)", 1.0, 15.0, 6.0, 0.5)

uploaded_file = st.file_uploader("Sube tu archivo de audio (MP3 / WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("Ejecutando escaneo de loudness, true peak y espectro..."):
        y, sr = librosa.load(uploaded_file, sr=None, mono=True)

        # --- 1. LUFS INTEGRADO REAL (ITU-R BS.1770 vía pyloudnorm) ---
        meter = pyln.Meter(sr)
        try:
            integrated_lufs = meter.integrated_loudness(y)
        except Exception:
            integrated_lufs = -70.0
        if not np.isfinite(integrated_lufs):
            integrated_lufs = -70.0

        # --- 2. LOUDNESS MOMENTÁNEO MÁXIMO (ventanas de 400ms, detecta ráfagas) ---
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

        # --- 3. TRUE PEAK (dBTP) vía sobremuestreo 4x (aproximación ITU-R BS.1770) ---
        oversample_factor = 4
        y_os = resample_poly(y, oversample_factor, 1)
        true_peak_amp = np.max(np.abs(y_os))
        true_peak_db = 20 * np.log10(true_peak_amp) if true_peak_amp > 0 else -100.0

        sample_peak_amp = np.max(np.abs(y))
        sample_peak_db = 20 * np.log10(sample_peak_amp) if sample_peak_amp > 0 else -100.0

        dynamic_range = true_peak_db - integrated_lufs

        # --- 4. ANÁLISIS ESPECTRAL ---
        n_fft = 2048
        S = np.abs(librosa.stft(y, n_fft=n_fft))
        S_norm = S / (np.max(S) + 1e-6)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        mean_spectrum_db = 20 * np.log10(np.mean(S_norm, axis=1) + 1e-6)

        sub_bass_mask = (freqs >= 20) & (freqs <= 60)
        mids_mask = (freqs >= 500) & (freqs <= 2000)
        ultrasonic_mask = freqs >= 17000

        sub_bass_val = np.mean(mean_spectrum_db[sub_bass_mask]) if np.any(sub_bass_mask) else -100
        mids_val = np.mean(mean_spectrum_db[mids_mask]) if np.any(mids_mask) else -100
        ultrasonic_val = np.mean(mean_spectrum_db[ultrasonic_mask]) if np.any(ultrasonic_mask) else -100
        bass_dominance = sub_bass_val - mids_val

        # --- REGLAS DE DETECCIÓN (basadas en umbrales configurables) ---
        reasons = []
        is_no_apto = False
        is_riesgo = False

        if true_peak_db > tp_limit:
            is_no_apto = True
            reasons.append(f"True Peak de {true_peak_db:.1f} dBTP supera el límite de {tp_limit:.1f} dBTP (riesgo de distorsión inter-muestra, sobre todo al re-codificar a MP3).")
        if max_momentary > momentary_limit:
            is_no_apto = True
            reasons.append(f"Ráfaga de loudness momentáneo de {max_momentary:.1f} LUFS (umbral: {momentary_limit:.1f} LUFS) — posible detonante de 'Disruptive Audio' por impacto repentino.")
        if integrated_lufs > lufs_target:
            is_riesgo = True
            reasons.append(f"Loudness integrado de {integrated_lufs:.1f} LUFS por encima del objetivo de {lufs_target:.1f} LUFS.")
        if bass_dominance > bass_dominance_limit:
            is_no_apto = True
            reasons.append("Exceso destructivo de subgraves respecto a las frecuencias medias.")
        if ultrasonic_val > ultrasonic_limit:
            is_riesgo = True
            reasons.append("Ruido ultrasónico (>17kHz) detectado en la mezcla.")
        if dynamic_range < dynamic_range_min:
            is_riesgo = True
            reasons.append("Rango dinámico muy comprimido (sonido tipo 'muro de ruido' constante).")

        # --- RESULTADO VISUAL ---
        st.subheader("📊 Resultado del Diagnóstico")

        if is_no_apto:
            st.error("❌ **RIESGO ALTO — parámetros fuera de estándares de la industria**")
        elif is_riesgo:
            st.warning("⚠️ **RIESGO MODERADO — algunos parámetros al límite**")
        else:
            st.success("✅ **DENTRO DE ESTÁNDARES — loudness, picos y espectro dentro de rangos seguros**")

        if reasons:
            with st.container():
                st.write("**Motivos detectados:**")
                for r in reasons:
                    st.markdown(f"- {r}")

        with st.expander("📌 Otras causas de rechazo que esta herramienta NO puede detectar", expanded=False):
            st.write("""
            * **Derechos de autor (Content ID):** canciones, remixes o samples registrados se eliminan aunque el audio esté técnicamente limpio.
            * **Voz, gritos o palabras malsonantes:** moderación de contenido hablado o cantado, gritos agudos o lenguaje explícito.
            * **Compresión MP3 defectuosa:** exportar con muy poco headroom puede generar distorsión al transcodificar; deja al menos -1 dBTP de margen.
            * **Contexto del sonido:** efectos como disparos, gritos de terror o sirenas pueden marcarse por su naturaleza aunque el volumen sea moderado.
            """)

        # --- MÉTRICAS ---
        st.write("### 🔍 Parámetros Técnicos")
        col1, col2, col3 = st.columns(3)

        with col1:
            if true_peak_db > tp_limit:
                badge, rec = '<span class="status-badge badge-danger">True Peak excedido</span>', f"**Ajuste:** baja la ganancia para dejar True Peak ≤ {tp_limit:.1f} dBTP."
            elif max_momentary > momentary_limit:
                badge, rec = '<span class="status-badge badge-danger">Ráfaga detectada</span>', "**Ajuste:** aplica un limitador para suavizar picos repentinos."
            else:
                badge, rec = '<span class="status-badge badge-success">Seguro</span>', f"Sample Peak: {sample_peak_db:.1f} dBFS"
            st.markdown(f"""
            <div class="metric-card">{badge}<h4>True Peak</h4>
            <h2 style="color:#58a6ff;">{true_peak_db:.1f} <small style="font-size:14px;">dBTP</small></h2>
            <hr style="border-color:#30363d; margin: 10px 0;">
            <p style="font-size:0.85rem;"><b>Límite:</b> ≤ {tp_limit:.1f} dBTP</p>
            <p style="font-size:0.85rem;">{rec}</p></div>
            """, unsafe_allow_html=True)

        with col2:
            if integrated_lufs > lufs_target:
                badge, rec = '<span class="status-badge badge-warning">Muy fuerte</span>', "**Ajuste:** baja la ganancia general o aplica menos compresión."
            else:
                badge, rec = '<span class="status-badge badge-success">Dentro de rango</span>', f"Momentáneo máx: {max_momentary:.1f} LUFS"
            st.markdown(f"""
            <div class="metric-card">{badge}<h4>Loudness Integrado (LUFS)</h4>
            <h2 style="color:#a5d6ff;">{integrated_lufs:.1f} <small style="font-size:14px;">LUFS</small></h2>
            <hr style="border-color:#30363d; margin: 10px 0;">
            <p style="font-size:0.85rem;"><b>Objetivo:</b> ≤ {lufs_target:.1f} LUFS</p>
            <p style="font-size:0.85rem;">{rec}</p></div>
            """, unsafe_allow_html=True)

        with col3:
            if dynamic_range < dynamic_range_min:
                badge, rec = '<span class="status-badge badge-warning">Sin dinámica</span>', "**Ajuste:** reduce la compresión/limitación."
            else:
                badge, rec = '<span class="status-badge badge-success">Buena dinámica</span>', "El audio respira correctamente."
            st.markdown(f"""
            <div class="metric-card">{badge}<h4>Rango Dinámico</h4>
            <h2 style="color:#d2a8ff;">{dynamic_range:.1f} <small style="font-size:14px;">dB</small></h2>
            <hr style="border-color:#30363d; margin: 10px 0;">
            <p style="font-size:0.85rem;"><b>Mínimo:</b> {dynamic_range_min:.1f} dB</p>
            <p style="font-size:0.85rem;">{rec}</p></div>
            """, unsafe_allow_html=True)

        # --- GRÁFICA DE ESPECTRO ---
        st.write("### 📈 Espectro de Frecuencia Normalizado")
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
