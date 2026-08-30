import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

st.set_page_config(page_title="Roblox Audio Moderation Checker Pro", page_icon="🎵", layout="wide")

# --- ESTILOS CSS CUSTOM (DASHBOARD MODERNO) ---
st.markdown("""
    <style>
    /* Fondo general y fuentes */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Tarjetas de Métricas */
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
    
    /* Indicadores de Estado */
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

    /* Estilo de la gráfica */
    .stPlotlyChart, div.ElementContainer {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Roblox Audio Moderation Checker Pro")
st.caption("Verificador avanzado de saturación, balance espectral y prevención de baneo por *Disruptive Audio*.")

uploaded_file = st.file_uploader("Sube tu archivo de audio (MP3 / WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    with st.spinner("Analizando espectro de audio y picos de ganancia..."):
        # Cargar audio
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # 1. Pico Máximo (dBFS)
        max_amplitude = np.max(np.abs(y))
        peak_db = 20 * np.log10(max_amplitude) if max_amplitude > 0 else -100.0
        
        # 2. Análisis Espectral (STFT Normalizada)
        n_fft = 2048
        S = np.abs(librosa.stft(y, n_fft=n_fft))
        S_norm = S / (np.max(S) + 1e-6)
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        mean_spectrum_db = 20 * np.log10(np.mean(S_norm, axis=1) + 1e-6)
        
        # Medición por bandas de frecuencia
        sub_bass_mask = (freqs >= 20) & (freqs <= 60)
        mids_mask = (freqs >= 500) & (freqs <= 2000)
        highs_mask = freqs >= 10000

        sub_bass_val = np.mean(mean_spectrum_db[sub_bass_mask]) if np.any(sub_bass_mask) else -100
        mids_val = np.mean(mean_spectrum_db[mids_mask]) if np.any(mids_mask) else -100
        highs_val = np.mean(mean_spectrum_db[highs_mask]) if np.any(highs_mask) else -100

        # Dominancia de Subgraves vs Medios
        bass_dominance = sub_bass_val - mids_val

        # --- REGLAS DE EVALUACIÓN ---
        is_no_apto = False
        is_riesgo = False

        if peak_db > 8.9 or bass_dominance > 14.0:
            is_no_apto = True
        elif (0.0 <= peak_db <= 8.9) or (9.5 < bass_dominance <= 14.0) or (highs_val < -65.0):
            is_riesgo = True

        # --- SONIDOS Y ALERTAS VISUALES ---
        st.subheader("📊 Resultado del Diagnóstico")
        
        if is_no_apto:
            sound_url = "https://assets.mixkit.co/active_storage/sfx/2874/2874-preview.mp3" # Error sound
            st.error("❌ **ESTADO: NO APTO** — Este archivo excede los parámetros seguros de Roblox y probablemente será rechazado por *Disruptive Audio* o saturación.")
        elif is_riesgo:
            sound_url = "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" # Warning sound
            st.warning("⚠️ **ESTADO: PUEDE SER NO APTO** — El audio está en zona límite. Te sugerimos realizar los ajustes indicados antes de subirlo.")
        else:
            sound_url = "https://assets.mixkit.co/active_storage/sfx/1435/1435-preview.mp3" # Success sound
            st.success("✅ **ESTADO: APTO** — El archivo está correctamente balanceado y listo para subirse a Roblox.")

        # Reproducir efecto de sonido ambiental en la Web App
        components.html(f"""
            <audio autoplay style="display:none;">
                <source src="{sound_url}" type="audio/mpeg">
            </audio>
        """, height=0)

        # --- TARJETAS DE MÉTRICAS DETALLADAS ---
        st.write("### 🔍 Parámetros Técnicos y Recomendaciones")
        col1, col2, col3 = st.columns(3)

        # 1. TARJETA PICO MÁXIMO
        with col1:
            if peak_db > 8.9:
                badge = '<span class="status-badge badge-danger">Saturación Crítica</span>'
                rec = f"**Ajuste:** Reduce **{-peak_db:.1f} dB** en Audacity."
            elif peak_db >= 0.0:
                badge = '<span class="status-badge badge-warning">Zona Límite</span>'
                rec = f"**Recomendación:** Recomendado bajar a **-0.5 dB**."
            else:
                badge = '<span class="status-badge badge-success">Seguro</span>'
                rec = "**Apto:** Sin riesgo de clipeo."

            st.markdown(f"""
            <div class="metric-card">
                {badge}
                <h4>Pico Máximo</h4>
                <h2 style="color:#58a6ff;">{peak_db:.2f} <small style="font-size:14px;">dBFS</small></h2>
                <hr style="border-color:#30363d; margin: 10px 0;">
                <p style="font-size:0.85rem;"><b>Objetivo Apto:</b> &lt; 0.0 dB (Max 8.9 dB)</p>
                <p style="font-size:0.85rem;">{rec}</p>
            </div>
            """, unsafe_allow_html=True)

        # 2. TARJETA SUBGRAVES
        with col2:
            if bass_dominance > 14.0:
                badge = '<span class="status-badge badge-danger">Graves Excesivos</span>'
                diff = bass_dominance - 9.5
                rec = f"**Ajuste:** Aplica *High-Pass Filter* a 60Hz o baja graves **-{diff:.1f} dB**."
            elif bass_dominance > 9.5:
                badge = '<span class="status-badge badge-warning">Graves Elevados</span>'
                diff = bass_dominance - 9.5
                rec = f"**Sugerencia:** Reduce graves **-{diff:.1f} dB** para mayor seguridad."
            else:
                badge = '<span class="status-badge badge-success">Equilibrado</span>'
                rec = "**Apto:** Balance perfecto con medios."

            st.markdown(f"""
            <div class="metric-card">
                {badge}
                <h4>Dominancia Subgraves</h4>
                <h2 style="color:#a5d6ff;">+{bass_dominance:.1f} <small style="font-size:14px;">dB vs Medios</small></h2>
                <hr style="border-color:#30363d; margin: 10px 0;">
                <p style="font-size:0.85rem;"><b>Objetivo Apto:</b> &le; 9.5 dB (Max 14.0 dB)</p>
                <p style="font-size:0.85rem;">{rec}</p>
            </div>
            """, unsafe_allow_html=True)

        # 3. TARJETA AGUDOS
        with col3:
            if highs_val < -65.0:
                badge = '<span class="status-badge badge-warning">Audio Ahogado</span>'
                rec = "**Ajuste:** Faltan agudos en la mezcla (>10 kHz)."
            elif highs_val > -20.0:
                badge = '<span class="status-badge badge-warning">Agudos Muy Altos</span>'
                rec = "**Ajuste:** Reduce altas frecuencias para evitar estridencia."
            else:
                badge = '<span class="status-badge badge-success">Equilibrado</span>'
                rec = "**Apto:** Rango de agudos limpio."

            st.markdown(f"""
            <div class="metric-card">
                {badge}
                <h4>Agudos (&gt;10 kHz)</h4>
                <h2 style="color:#d2a8ff;">{highs_val:.1f} <small style="font-size:14px;">dB Relativo</small></h2>
                <hr style="border-color:#30363d; margin: 10px 0;">
                <p style="font-size:0.85rem;"><b>Objetivo Apto:</b> -65.0 dB a -20.0 dB</p>
                <p style="font-size:0.85rem;">{rec}</p>
            </div>
            """, unsafe_allow_html=True)

        # --- GRÁFICA DE ESPECTRO ---
        st.write("### 📈 Espectro de Frecuencia Normalizado")
        
        # Tema Oscuro para Matplotlib
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 3.8))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#0d1117')

        ax.plot(freqs, mean_spectrum_db, color='#bc8cff', linewidth=1.8, label='Respuesta de Frecuencia')
        ax.set_xscale('log')
        ax.set_xlim(20, sr // 2)
        ax.set_ylim(-90, 5)
        
        # Zonas destacadas
        ax.axvspan(20, 60, color='#f85149', alpha=0.18, label='Zona Subgraves (<60Hz)')
        ax.axvspan(500, 2000, color='#3fb950', alpha=0.10, label='Zona Medios (Referencia)')
        
        ax.set_title("Frecuencia (Hz) vs Amplitud (dBFS)", fontsize=11, color='#c9d1d9', pad=12)
        ax.set_xlabel("Frecuencia (Hz)", fontsize=9, color='#8b949e')
        ax.set_ylabel("Amplitud Relativa (dB)", fontsize=9, color='#8b949e')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.grid(True, which="both", ls="--", color='#21262d', alpha=0.7)
        ax.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=8)

        st.pyplot(fig)
