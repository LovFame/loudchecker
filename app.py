import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

st.set_page_config(page_title="Roblox Audio Moderation Checker Pro", page_icon="🎵", layout="wide")

# --- ESTILOS CSS CUSTOM ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #58a6ff;
    }
    .status-badge {
        font-size: 0.85rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 8px;
        text-transform: uppercase;
    }
    .badge-success { background-color: rgba(46, 160, 67, 0.2); color: #3fb950; border: 1px solid #2ea643; }
    .badge-warning { background-color: rgba(210, 153, 34, 0.2); color: #d29922; border: 1px solid #bb8009; }
    .badge-danger { background-color: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid #f85149; }
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Roblox Audio Moderation Checker Pro")
st.caption("Verificador avanzado de ganancia, ráfagas transitorias, densidad LUFS y prevención de baneo por *Disruptive Audio*.")

uploaded_file = st.file_uploader("Sube tu archivo de audio (MP3 / WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    with st.spinner("Ejecutando escaneo profundo de audio y espectro..."):
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # 1. Peak & True Peak Estimación
        max_amplitude = np.max(np.abs(y))
        peak_db = 20 * np.log10(max_amplitude) if max_amplitude > 0 else -100.0
        
        # 2. Medición RMS y Densidad Energética (Proxy de LUFS)
        rms = librosa.feature.rms(y=y)[0]
        mean_rms_db = 20 * np.log10(np.mean(rms)) if np.mean(rms) > 0 else -100.0
        max_rms_db = 20 * np.log10(np.max(rms)) if np.max(rms) > 0 else -100.0
        dynamic_range = peak_db - mean_rms_db

        # 3. Análisis Espectral y Ultrasónico
        n_fft = 2048
        S = np.abs(librosa.stft(y, n_fft=n_fft))
        S_norm = S / (np.max(S) + 1e-6)
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        mean_spectrum_db = 20 * np.log10(np.mean(S_norm, axis=1) + 1e-6)
        
        sub_bass_mask = (freqs >= 20) & (freqs <= 60)
        mids_mask = (freqs >= 500) & (freqs <= 2000)
        highs_mask = (freqs >= 10000) & (freqs <= 16000)
        ultrasonic_mask = freqs >= 17000

        sub_bass_val = np.mean(mean_spectrum_db[sub_bass_mask]) if np.any(sub_bass_mask) else -100
        mids_val = np.mean(mean_spectrum_db[mids_mask]) if np.any(mids_mask) else -100
        highs_val = np.mean(mean_spectrum_db[highs_mask]) if np.any(highs_mask) else -100
        ultrasonic_val = np.mean(mean_spectrum_db[ultrasonic_mask]) if np.any(ultrasonic_mask) else -100

        bass_dominance = sub_bass_val - mids_val

        # --- REGLAS DE DETECCIÓN ESTRICTAS ---
        reasons = []
        is_no_apto = False
        is_riesgo = False

        if max_rms_db > -2.5:
            is_no_apto = True
            reasons.append("Pico RMS transitorio excesivo (Disruptive Audio por impacto repentino).")
        if mean_rms_db > -9.0:
            is_no_apto = True
            reasons.append("Audio ultra-comprimido/luid (Loudness RMS sostenido demasiado alto).")
        if bass_dominance > 26.0:
            is_no_apto = True
            reasons.append("Exceso destructivo de subgraves.")
        if ultrasonic_val > -30.0:
            is_riesgo = True
            reasons.append("Ruido ultrasónico (>17kHz) detectado en la mezcla.")
        if dynamic_range < 5.0:
            is_riesgo = True
            reasons.append("Rango dinámico extremadamente aplastado (bloque de ruido constante).")

        # --- RESULTADO VISUAL ---
        st.subheader("📊 Resultado del Diagnóstico de Ganancia")
        
        if is_no_apto:
            sound_url = "https://cdn.pixabay.com/download/audio/2021/08/04/audio_c6ccf3232f.mp3?filename=negative_beeps-6008.mp3"
            st.error("❌ **ESTADO: NO APTO (Alto riesgo de baneo por Disruptive Audio)**")
        elif is_riesgo:
            sound_url = "https://cdn.pixabay.com/download/audio/2022/03/15/audio_c8c8a73467.mp3?filename=system-notification-129219.mp3"
            st.warning("⚠️ **ESTADO: EN RIESGO (Parámetros al límite)**")
        else:
            sound_url = "https://cdn.pixabay.com/download/audio/2021/08/04/audio_bb630cc098.mp3?filename=success-1-6297.mp3"
            st.success("✅ **ESTADO: APTO TÉCNICAMENTE (Frecuencia y Ganancia limpias)**")

        components.html(f"""
            <audio autoplay style="display:none;">
                <source src="{sound_url}" type="audio/mpeg">
            </audio>
        """, height=0)

        # --- CHECKLIST DE MODERACIÓN EXTRA (RAZONES DE BANEO NON-WAVE) ---
        st.write("### 🚨 Checklist de Causas de Rechazo en Roblox")
        
        with st.expander("📌 ¿Por qué Roblox puede tumbarlo si la gráfica sale limpia?", expanded=True):
            st.write("""
            Si este analizador dice **APTO** o **SEGURO** pero Roblox sigue eliminando el audio, revisa estas causas externas al volumen:
            
            * **Derechos de Autor (Copyright / Content ID):** Si es una canción famosa, remix o incluye una base registrada, el bot automatizado de Roblox la borra al instante.
            * **Palabras malsonantes / Letra explícita:** Moderación de texto/voz por groserías, referencias no aptas o violencia en la letra.
            * **Compresión MP3 defectuosa (True Peak):** Al convertir a MP3, los picos pueden distorsionar al reproducirse en Roblox. Baja el volumen en Audacity a **-3.0 dB** antes de exportar.
            * **Falta de corte en subs (Bajos fantasma):** Frecuencias por debajo de 35 Hz que hacen vibrar el reproductor de Roblox.
            """)

        # --- MÉTRICAS ---
        st.write("### 🔍 Parámetros Técnicos y Recomendaciones")
        col1, col2, col3 = st.columns(3)

        # 1. POTENCIA RMS / RÁFAGA
        with col1:
            if max_rms_db > -2.5:
                badge = '<span class="status-badge badge-danger">Ráfaga Crítica</span>'
                rec = "**Ajuste:** Aplica *Limitador Duro* a -3 dB en Audacity."
            elif mean_rms_db > -9.0:
                badge = '<span class="status-badge badge-warning">Demasiado Fuerte</span>'
                rec = "**Ajuste:** Baja la ganancia general -3 dB."
            else:
                badge = '<span class="status-badge badge-success">Seguro</span>'
                rec = f"**Peak:** {peak_db:.2f} dBFS | **LUFS Proxy:** {mean_rms_db:.1f} dB"

            st.markdown(f"""
            <div class="metric-card">
                {badge}
                <h4>Potencia RMS (Loudness)</h4>
                <h2 style="color:#58a6ff;">{mean_rms_db:.1f} <small style="font-size:14px;">dB RMS</small></h2>
                <hr style="border-color:#30363d; margin: 10px 0;">
                <p style="font-size:0.85rem;"><b>Objetivo Apto:</b> &lt; -10.0 dB RMS</p>
                <p style="font-size:0.85rem;">{rec}</p>
            </div>
            """, unsafe_allow_html=True)

        # 2. SUBGRAVES
        with col2:
            if bass_dominance > 26.0:
                badge = '<span class="status-badge badge-danger">Graves Excesivos</span>'
                diff = bass_dominance - 22.0
                rec = f"**Ajuste:** Aplica filtro paso alto en 40Hz."
            elif bass_dominance > 20.0:
                badge = '<span class="status-badge badge-warning">Graves Pesados</span>'
                rec = "**Nota:** Rango pesado pero generalmente aceptado."
            else:
                badge = '<span class="status-badge badge-success">Equilibrado</span>'
                rec = "**Apto:** Balance correcto de graves."

            st.markdown(f"""
            <div class="metric-card">
                {badge}
                <h4>Dominancia Subgraves</h4>
                <h2 style="color:#a5d6ff;">+{bass_dominance:.1f} <small style="font-size:14px;">dB vs Medios</small></h2>
                <hr style="border-color:#30363d; margin: 10px 0;">
                <p style="font-size:0.85rem;"><b>Objetivo Apto:</b> &le; 26.0 dB</p>
                <p style="font-size:0.85rem;">{rec}</p>
            </div>
            """, unsafe_allow_html=True)

        # 3. RANGO DINÁMICO
        with col3:
            if dynamic_range < 5.0:
                badge = '<span class="status-badge badge-warning">Sin Dinámica</span>'
                rec = "**Ajuste:** Audio demasiado aplastado/comprimido."
            else:
                badge = '<span class="status-badge badge-success">Dinámica Buena</span>'
                rec = "**Apto:** Respira correctamente."

            st.markdown(f"""
            <div class="metric-card">
                {badge}
                <h4>Rango Dinámico</h4>
                <h2 style="color:#d2a8ff;">{dynamic_range:.1f} <small style="font-size:14px;">dB</small></h2>
                <hr style="border-color:#30363d; margin: 10px 0;">
                <p style="font-size:0.85rem;"><b>Objetivo Apto:</b> &gt; 6.0 dB</p>
                <p style="font-size:0.85rem;">{rec}</p>
            </div>
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
        
        ax.set_title("Frecuencia (Hz) vs Amplitud (dBFS)", fontsize=11, color='#c9d1d9', pad=12)
        ax.set_xlabel("Frecuencia (Hz)", fontsize=9, color='#8b949e')
        ax.set_ylabel("Amplitud Relativa (dB)", fontsize=9, color='#8b949e')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.grid(True, which="both", ls="--", color='#21262d', alpha=0.7)
        ax.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=8)

        st.pyplot(fig)
