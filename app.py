import io
import streamlit as st
import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

st.set_page_config(page_title="Roblox Audio Moderation", page_icon="🎵", layout="wide")

# ============================== FUNCIONES DE AUDIO ==============================

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


# ============================== ESTILOS ==============================
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border: 1px solid #30363d; border-radius: 12px; padding: 18px; margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); transition: transform 0.2s ease, border-color 0.2s ease;
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
st.caption("Analiza y corrige LUFS, True Peak, subgraves y ultrasónicos para reducir el riesgo de rechazo por *Disruptive Audio*.")

st.markdown("""
<div class="disclaimer-box">
⚠️ <b>Roblox no publica los umbrales exactos de su sistema de moderación</b>, por lo que ninguna herramienta externa
—esta incluida— puede garantizar un resultado 100% preciso. Esta app usa estándares reales de la industria
(ITU-R BS.1770 / EBU R128) y te deja fijar tus propios objetivos abajo. Además, Roblox también rechaza audio por
motivos que ningún procesamiento de volumen puede corregir: copyright, voces/gritos, o contenido explícito.
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
    st.header("🎚️ Objetivos de audio")
    strict_mode = st.toggle("🔒 Modo estricto (valores recomendados)", value=True,
                             help="Bloquea los umbrales en los valores recomendados por estándares de la industria, para evitar que se aflojen sin querer y la app diga 'apto' cuando no debería.")

    if not strict_mode:
        st.warning("⚠️ Estás en modo personalizado. Un resultado 'apto' con umbrales aflojados **no significa que Roblox lo vaya a aceptar** — solo cambia lo que esta app te muestra, no el criterio real de Roblox.")

    st.caption("Estos valores se usan para: (1) decidir si tu audio está en riesgo, y (2) como META cuando le des a 'Generar versión corregida'.")

    lufs_target = st.slider(
        "LUFS integrado objetivo", -30.0, -6.0, DEFAULTS["lufs_target"], 0.5, disabled=strict_mode,
        help="Mide el volumen PROMEDIO percibido de todo el audio (no solo el pico). -14 LUFS es la referencia de Spotify/YouTube; -23 LUFS es el estándar de TV (EBU R128), mucho más bajo. Al transformar, la app sube o baja la ganancia de TODO el audio para que el promedio caiga en este número. Si lo pones muy alto (ej. -6), el audio quedará muy fuerte y es más probable que choque con el límite de True Peak de abajo."
    )
    tp_limit = st.slider(
        "True Peak máximo (dBTP)", -6.0, 0.0, DEFAULTS["tp_limit"], 0.1, disabled=strict_mode,
        help="El pico REAL de la onda, incluyendo los picos 'entre muestras' que aparecen al reconstruir el audio en un DAC o al convertir a MP3 (por eso es distinto del pico normal que ves en un editor). Si el True Peak es muy alto (cercano a 0 dBTP), el audio puede distorsionarse al reproducirse o al re-codificarse. Al transformar, si subir el volumen para llegar al LUFS objetivo haría que el True Peak superara este límite, la app reduce la ganancia lo necesario para respetarlo — por eso a veces el LUFS final no llega exacto al objetivo."
    )
    momentary_limit = st.slider(
        "Loudness momentáneo máx. (LUFS, ráfagas)", -20.0, -3.0, DEFAULTS["momentary_limit"], 0.5, disabled=strict_mode,
        help="Loudness medido en ventanas cortas (400ms) en vez de todo el audio. Detecta RÁFAGAS o impactos repentinos (un grito, un golpe fuerte) que pueden disparar 'Disruptive Audio' aunque el promedio general del audio sea bajo. Este valor solo se usa para el DIAGNÓSTICO (marcar riesgo); la corrección automática no aplica un limitador de ráfagas independiente, solo el techo de True Peak general."
    )
    bass_dominance_limit = st.slider(
        "Dominancia de subgraves máx. (dB vs medios)", 10.0, 35.0, DEFAULTS["bass_dominance_limit"], 1.0, disabled=strict_mode,
        help="Compara qué tan fuerte está la banda de 20-60 Hz (subgraves) frente a la banda de 500-2000 Hz (medios, donde está la mayoría de la voz e instrumentos). Un valor alto significa graves exagerados tipo 'bocina de auto'. Al transformar, si tu audio excede este límite, la app aplica un EQ que atenúa específicamente la banda de 20-60 Hz hasta bajar la dominancia a este número — no toca el resto del espectro."
    )
    ultrasonic_limit = st.slider(
        "Ruido ultrasónico máx. (dB, >17kHz)", -50.0, -10.0, DEFAULTS["ultrasonic_limit"], 1.0, disabled=strict_mode,
        help="Energía por encima de 17kHz, normalmente inaudible para la mayoría de las personas pero que puede colarse como ruido de compresión o de un micrófono defectuoso. Al transformar, si tu audio excede este límite, la app aplica un filtro que atenúa específicamente las frecuencias por encima de 17kHz — es seguro quitarlo porque casi nadie lo escucha de todas formas."
    )
    dynamic_range_min = st.slider(
        "Rango dinámico mínimo (dB)", 1.0, 15.0, DEFAULTS["dynamic_range_min"], 0.5, disabled=strict_mode,
        help="Diferencia entre el pico más alto y el volumen promedio. Un valor bajo significa que el audio está muy comprimido/'aplastado' (suena como un muro de ruido constante, sin variación). IMPORTANTE: esto solo se usa para DIAGNÓSTICO. Si tu audio original ya viene muy comprimido, esta app NO puede 'devolverle' la dinámica perdida — esa información ya no existe en la señal. La única solución real es partir de una versión menos comprimida del audio original."
    )

    if strict_mode:
        lufs_target = DEFAULTS["lufs_target"]
        tp_limit = DEFAULTS["tp_limit"]
        momentary_limit = DEFAULTS["momentary_limit"]
        bass_dominance_limit = DEFAULTS["bass_dominance_limit"]
        ultrasonic_limit = DEFAULTS["ultrasonic_limit"]
        dynamic_range_min = DEFAULTS["dynamic_range_min"]

uploaded_file = st.file_uploader("Sube tu archivo de audio (MP3 / WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("Ejecutando escaneo de loudness, true peak y espectro..."):
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
            reasons.append(f"True Peak de {true_peak_db:.1f} dBTP supera el límite de {tp_limit:.1f} dBTP.")
        if max_momentary > momentary_limit:
            is_no_apto = True
            reasons.append(f"Ráfaga de loudness momentáneo de {max_momentary:.1f} LUFS (umbral: {momentary_limit:.1f} LUFS).")
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

        st.subheader("📊 Resultado del Diagnóstico")
        if is_no_apto:
            st.error("❌ **RIESGO ALTO — parámetros fuera de estándares de la industria**")
        elif is_riesgo:
            st.warning("⚠️ **RIESGO MODERADO — algunos parámetros al límite**")
        else:
            st.success("✅ **DENTRO DE ESTÁNDARES — loudness, picos y espectro dentro de rangos seguros**")
            st.caption("Esto significa que el audio cumple buenas prácticas de mastering (LUFS, True Peak, espectro). **No es una garantía de que Roblox lo vaya a aceptar** — su moderación también analiza el contenido del sonido (voces, gritos, distorsión perceptual) de una forma que esta app no puede replicar.")

        if reasons:
            st.write("**Motivos detectados:**")
            for r in reasons:
                st.markdown(f"- {r}")

        with st.expander("📌 Causas de rechazo que esta herramienta NO puede detectar ni corregir"):
            st.write("""
            * **Derechos de autor (Content ID):** canciones, remixes o samples registrados se eliminan aunque el audio esté técnicamente limpio.
            * **Voz, gritos o palabras malsonantes:** moderación de contenido hablado o cantado, gritos agudos o lenguaje explícito.
            * **Contexto del sonido:** disparos, gritos de terror o sirenas pueden marcarse por su naturaleza aunque el volumen sea moderado.
            """)

        st.write("### 🔍 Parámetros Técnicos")
        col1, col2, col3 = st.columns(3)

        with col1:
            if true_peak_db > tp_limit:
                badge, rec = '<span class="status-badge badge-danger">True Peak excedido</span>', f"Baja la ganancia para dejar True Peak ≤ {tp_limit:.1f} dBTP."
            elif max_momentary > momentary_limit:
                badge, rec = '<span class="status-badge badge-danger">Ráfaga detectada</span>', "Aplica un limitador para suavizar picos repentinos."
            else:
                badge, rec = '<span class="status-badge badge-success">Seguro</span>', f"Sample Peak: {sample_peak_db:.1f} dBFS"
            st.markdown(f"""<div class="metric-card">{badge}<h4>True Peak</h4>
            <h2 style="color:#58a6ff;">{true_peak_db:.1f} <small style="font-size:14px;">dBTP</small></h2>
            <hr style="border-color:#30363d; margin:10px 0;">
            <p style="font-size:0.85rem;"><b>Límite:</b> ≤ {tp_limit:.1f} dBTP</p>
            <p style="font-size:0.85rem;">{rec}</p></div>""", unsafe_allow_html=True)

        with col2:
            if integrated_lufs > lufs_target:
                badge, rec = '<span class="status-badge badge-warning">Muy fuerte</span>', "Baja la ganancia general o aplica menos compresión."
            else:
                badge, rec = '<span class="status-badge badge-success">Dentro de rango</span>', f"Momentáneo máx: {max_momentary:.1f} LUFS"
            st.markdown(f"""<div class="metric-card">{badge}<h4>Loudness Integrado (LUFS)</h4>
            <h2 style="color:#a5d6ff;">{integrated_lufs:.1f} <small style="font-size:14px;">LUFS</small></h2>
            <hr style="border-color:#30363d; margin:10px 0;">
            <p style="font-size:0.85rem;"><b>Objetivo:</b> ≤ {lufs_target:.1f} LUFS</p>
            <p style="font-size:0.85rem;">{rec}</p></div>""", unsafe_allow_html=True)

        with col3:
            if dynamic_range < dynamic_range_min:
                badge, rec = '<span class="status-badge badge-warning">Sin dinámica</span>', "No recuperable por procesamiento; requiere un original menos comprimido."
            else:
                badge, rec = '<span class="status-badge badge-success">Buena dinámica</span>', "El audio respira correctamente."
            st.markdown(f"""<div class="metric-card">{badge}<h4>Rango Dinámico</h4>
            <h2 style="color:#d2a8ff;">{dynamic_range:.1f} <small style="font-size:14px;">dB</small></h2>
            <hr style="border-color:#30363d; margin:10px 0;">
            <p style="font-size:0.85rem;"><b>Mínimo:</b> {dynamic_range_min:.1f} dB</p>
            <p style="font-size:0.85rem;">{rec}</p></div>""", unsafe_allow_html=True)

        col4, col5 = st.columns(2)
        with col4:
            badge = '<span class="status-badge badge-danger">Excedido</span>' if bass_dominance > bass_dominance_limit else '<span class="status-badge badge-success">Balanceado</span>'
            st.markdown(f"""<div class="metric-card">{badge}<h4>Dominancia de Subgraves</h4>
            <h2 style="color:#f0883e;">+{bass_dominance:.1f} <small style="font-size:14px;">dB vs medios</small></h2>
            <hr style="border-color:#30363d; margin:10px 0;">
            <p style="font-size:0.85rem;"><b>Límite:</b> ≤ {bass_dominance_limit:.1f} dB</p></div>""", unsafe_allow_html=True)
        with col5:
            badge = '<span class="status-badge badge-danger">Excedido</span>' if ultrasonic_val > ultrasonic_limit else '<span class="status-badge badge-success">Limpio</span>'
            st.markdown(f"""<div class="metric-card">{badge}<h4>Ruido Ultrasónico</h4>
            <h2 style="color:#79c0ff;">{ultrasonic_val:.1f} <small style="font-size:14px;">dB</small></h2>
            <hr style="border-color:#30363d; margin:10px 0;">
            <p style="font-size:0.85rem;"><b>Límite:</b> ≤ {ultrasonic_limit:.1f} dB</p></div>""", unsafe_allow_html=True)

        # ============================== TRANSFORMACIÓN ==============================
        st.write("### 🎛️ Generar versión corregida")
        st.caption(
            "Aplica, en este orden: 1) EQ que atenúa subgraves y/o ultrasónicos si exceden los límites de la barra "
            "lateral, 2) ganancia para acercar el LUFS al objetivo, 3) un techo de True Peak que reduce la ganancia "
            "si hiciera falta. El rango dinámico NO se corrige — si ya está muy comprimido, no hay forma de recuperarlo."
        )

        if st.button("🎚️ Generar versión corregida"):
            with st.spinner("Aplicando EQ, ganancia y límite de True Peak..."):
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

            st.success("Listo — así quedó el audio después de la corrección:")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("LUFS final", f"{final_lufs:.1f}", f"objetivo {lufs_target:.1f}")
            r2.metric("True Peak final", f"{final_tp:.1f} dBTP", f"límite {tp_limit:.1f}")
            r3.metric("Subgraves atenuados", f"-{bass_cut:.1f} dB" if bass_cut > 0 else "sin cambio")
            r4.metric("Ultrasónicos atenuados", f"-{ultrasonic_cut:.1f} dB" if ultrasonic_cut > 0 else "sin cambio")

            if final_bands["bass_dominance"] - bass_dominance_limit > 0.5:
                st.warning("La dominancia de subgraves quedó ligeramente por encima del límite; puede requerir varias pasadas o un filtro más agresivo.")

            st.audio(buf, format="audio/wav")
            st.download_button(
                "⬇️ Descargar audio corregido (.wav)",
                data=buf, file_name="audio_corregido.wav", mime="audio/wav",
            )
            st.caption("Se exporta en WAV para no perder calidad. Si necesitas MP3 para subir a Roblox, conviértelo con Audacity o un conversor externo antes de subirlo.")

        # ============================== ESPECTRO ==============================
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
