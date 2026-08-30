import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt

st.set_page_config(page_title="Roblox Audio Moderation Checker", page_icon="🎵")

st.title("🎵 Detector de Moderación de Audio para Roblox")
st.write("Sube tu archivo de audio para analizar picos, saturación y espectro de frecuencias.")

uploaded_file = st.file_uploader("Selecciona tu archivo de audio (MP3 o WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    with st.spinner("Analizando audio completo..."):
        # Cargar audio
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # 1. Medición de Pico Máximo Real (dBFS)
        max_amplitude = np.max(np.abs(y))
        peak_db = 20 * np.log10(max_amplitude) if max_amplitude > 0 else -100.0
        
        # 2. Análisis Espectral (STFT absoluto)
        n_fft = 2048
        S = np.abs(librosa.stft(y, n_fft=n_fft))
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        mean_spectrum = np.mean(S, axis=1)
        # Convertir a dBFS absolutos
        mean_spectrum_db = 20 * np.log10(mean_spectrum + 1e-6)
        
        # Filtros de banda de frecuencia
        sub_bass_mask = (freqs >= 20) & (freqs <= 60)
        sub_bass_mean = np.mean(mean_spectrum_db[sub_bass_mask]) if np.any(sub_bass_mask) else -100
        
        ultra_high_mask = freqs >= 15000
        ultra_high_mean = np.mean(mean_spectrum_db[ultra_high_mask]) if np.any(ultra_high_mask) else -100

        # --- EVALUACIÓN GENERAL DE ESTADO ---
        is_no_apto = peak_db > 8.9 or sub_bass_mean > -18.0
        is_riesgo = (0.0 <= peak_db <= 8.9) or (-25.0 < sub_bass_mean <= -18.0) or (ultra_high_mean > -20.0)

        st.subheader("📊 Resultado de Evaluación")

        if is_no_apto:
            st.error("❌ **ESTADO: NO APTO**\n\nEste archivo superó los límites permitidos y corre alto riesgo de ser rechazado.")
        elif is_riesgo:
            st.warning("⚠️ **ESTADO: PUEDE SER NO APTO!**\n\nEl audio está en zona límite de moderación.")
        else:
            st.success("✅ **ESTADO: APTO**\n\nEl audio cumple con los márgenes de volumen y espectro seguro.")

        # --- MOSTRAR SIEMPRE TODOS LOS DATOS TÉCNICOS ---
        st.write("**Desglose detallado del análisis:**")

        # Registro 1: Pico y Saturación
        if peak_db > 8.9:
            st.markdown(f"🔴 **Pico Máximo (dBFS):** `{peak_db:.2f} dB` — **Saturación Crítica** (Supera el límite de 8.9 dB).")
        elif peak_db >= 0.0:
            st.markdown(f"🟡 **Pico Máximo (dBFS):** `{peak_db:.2f} dB` — **Riesgo de Saturación** (Entre 0.0 y 8.9 dB).")
        else:
            st.markdown(f"🟢 **Pico Máximo (dBFS):** `{peak_db:.2f} dB` — **Nivel Seguro** (Por debajo de 0.0 dB).")

        # Registro 2: Subgraves
        if sub_bass_mean > -18.0:
            st.markdown(f"🔴 **Subgraves (<60 Hz):** `{sub_bass_mean:.1f} dB` medio — **Excesivo** (Riesgo alto de Disruptive Audio).")
        elif sub_bass_mean > -25.0:
            st.markdown(f"🟡 **Subgraves (<60 Hz):** `{sub_bass_mean:.1f} dB` medio — **Elevado**.")
        else:
            st.markdown(f"🟢 **Subgraves (<60 Hz):** `{sub_bass_mean:.1f} dB` medio — **Normal**.")

        # Registro 3: Agudos
        if ultra_high_mean > -20.0:
            st.markdown(f"🟡 **Agudos (>15 kHz):** `{ultra_high_mean:.1f} dB` medio — **Muy Altos**.")
        else:
            st.markdown(f"🟢 **Agudos (>15 kHz):** `{ultra_high_mean:.1f} dB` medio — **Equilibrados**.")

        # --- GRÁFICA DE ESPECTRO ---
        st.subheader("📈 Análisis de Espectro (Frecuencias)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs, mean_spectrum_db, color='#8a2be2')
        ax.set_xscale('log')
        ax.set_xlim(20, sr // 2)
        ax.axvspan(20, 60, color='red', alpha=0.15, label='Zona de riesgo por Subgraves')
        ax.set_title("Espectro de Frecuencia (Hz vs dBFS)")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Amplitud Real (dBFS)")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        st.pyplot(fig)
