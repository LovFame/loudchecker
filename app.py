import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt

st.set_page_config(page_title="Roblox Audio Moderation Checker", page_icon="🎵")

st.title("🎵 Detector de Moderación de Audio para Roblox")
st.write("Sube tu archivo de audio para analizar picos (dBFS), saturación y distribución de frecuencias.")

uploaded_file = st.file_uploader("Selecciona tu archivo de audio (MP3 o WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    with st.spinner("Analizando espectro de sonido y niveles de pico..."):
        # Cargar audio con Librosa
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # 1. Cálculo de Pico Máximo (dBFS)
        max_amplitude = np.max(np.abs(y))
        peak_db = 20 * np.log10(max_amplitude) if max_amplitude > 0 else -100
        
        # 2. Análisis del Espectro de Frecuencias (FFT)
        n_fft = 2048
        S = np.abs(librosa.stft(y, n_fft=n_fft))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        mean_spectrum = np.mean(S, axis=1)
        mean_spectrum_db = 20 * np.log10(mean_spectrum + 1e-6)
        
        # Frecuencias extremas
        sub_bass_mask = freqs <= 45
        sub_bass_max = np.max(mean_spectrum_db[sub_bass_mask]) if np.any(sub_bass_mask) else -100
        
        ultra_high_mask = freqs >= 16000
        ultra_high_max = np.max(mean_spectrum_db[ultra_high_mask]) if np.any(ultra_high_mask) else -100

        # --- EVALUACIÓN DE APTITUD ---
        st.subheader("📊 Resultado de Evaluación")
        
        reasons = []
        status = "APTO"

        if peak_db > -0.5:
            status = "NO APTO"
            reasons.append(f"🔴 **Saturación Digital/Clipping:** El pico alcanza **{peak_db:.2f} dBFS** (Supera el límite de -0.5 dBFS). El bot lo detectará como ruido disruptivo.")
        elif peak_db > -1.0:
            if status != "NO APTO": status = "POSIBLEMENTE APTO (RIESGO MEDIO)"
            reasons.append(f"🟡 **Pico Ajustado:** El pico alcanza **{peak_db:.2f} dBFS**. Está cerca de 0 dBFS; se recomienda bajarlo a -1.0 dBFS.")

        if sub_bass_max > -10:
            if status != "NO APTO": status = "POSIBLEMENTE APTO (RIESGO MEDIO)"
            reasons.append(f"🟡 **Subgraves Excesivos:** Hay una acumulación fuerte en frecuencias menores a 45 Hz ({sub_bass_max:.1f} dB).")

        if ultra_high_max > -25:
            if status != "NO APTO": status = "POSIBLEMENTE APTO (RIESGO MEDIO)"
            reasons.append(f"🟡 **Agudos Estridentes:** Hay picos altos por encima de 16,000 Hz ({ultra_high_max:.1f} dB).")

        # Mostrar estado
        if status == "APTO":
            st.success("✅ **ESTADO: APTO**\n\nEl audio no presenta clipping agresivo y tiene un espectro equilibrado. Pasa los controles estándar de moderación.")
        elif status == "POSIBLEMENTE APTO (RIESGO MEDIO)":
            st.warning("⚠️ **ESTADO: POSIBLEMENTE APTO (RIESGO MEDIO)**\n\nEl audio podría pasar, pero tiene parámetros al límite de lo permitido.")
        else:
            st.error("❌ **ESTADO: NO APTO**\n\nEste archivo tiene altas probabilidades de ser rechazado o marcado como Disruptive Audio.")

        if reasons:
            st.write("**Detalles encontrados:**")
            for r in reasons:
                st.markdown(r)

        # --- GRÁFICA DEL ESPECTRO ---
        st.subheader("📈 Análisis de Espectro (Frecuencias)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs, mean_spectrum_db, color='#8a2be2')
        ax.set_xscale('log')
        ax.set_xlim(20, sr // 2)
        ax.set_title("Espectro de Frecuencia (Hz vs dB)")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Amplitud (dB)")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        st.pyplot(fig)
