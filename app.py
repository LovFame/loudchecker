import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt

st.set_page_config(page_title="Roblox Audio Moderation Checker", page_icon="🎵")

st.title("🎵 Roblox Mod Detection!")
st.write("Sube tu archivo de audio para analizar picos (dBFS), saturación y distribución de frecuencias.")

uploaded_file = st.file_uploader("Selecciona tu archivo de audio (MP3 o WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    with st.spinner("Analizando audio..."):
        # Cargar audio normalizado en rango [-1.0, 1.0]
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # 1. Cálculo real de Pico Máximo en dBFS (Escala digital correcta: máx 0 dBFS)
        max_amplitude = np.max(np.abs(y))
        peak_db = 20 * np.log10(max_amplitude) if max_amplitude > 0 else -100.0
        
        # 2. Análisis del Espectro de Frecuencias (FFT Normalizada)
        n_fft = 2048
        S = np.abs(librosa.stft(y, n_fft=n_fft))
        # Normalizar espectro respecto al pico máximo
        S_norm = S / (np.max(S) + 1e-6)
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        mean_spectrum = np.mean(S_norm, axis=1)
        mean_spectrum_db = 20 * np.log10(mean_spectrum + 1e-6)
        
        # Frecuencias extremas (Evaluación en escala dBFS relativa)
        sub_bass_mask = (freqs >= 20) & (freqs <= 45)
        sub_bass_max = np.max(mean_spectrum_db[sub_bass_mask]) if np.any(sub_bass_mask) else -100
        
        ultra_high_mask = freqs >= 16000
        ultra_high_max = np.max(mean_spectrum_db[ultra_high_mask]) if np.any(ultra_high_mask) else -100

# --- EVALUACIÓN CORREGIDA ---
        st.subheader("📊 Resultado de Evaluación")
        
        reasons = []

        # Regla de evaluación según el valor del pico (peak_db)
        if peak_db > 8.9:
            status = "NO APTO"
            st.error(f"❌ **ESTADO: NO APTO**\n\nEl pico alcanza **{peak_db:.2f} dB**, superando el límite crítico de 8.9 dB. Este archivo tiene altas probabilidades de ser rechazado por Disruptive Audio.")
            reasons.append(f"🔴 **Exceso de volumen/saturación:** El nivel medido es de **{peak_db:.2f} dB** (mayor a 8.9 dB).")

        elif peak_db >= 0.0 and peak_db <= 8.9:
            status = "PUEDE SER NO APTO!"
            st.warning(f"⚠️ **ESTADO: PUEDE SER NO APTO!**\n\nEl pico está entre 0.0 y 8.9 dB (**{peak_db:.2f} dB**). Dependiendo del tipo de sonido y los graves, el bot podría marcarlo o pasarlo.")
            reasons.append(f"🟡 **Zona de Riesgo:** El nivel medido es de **{peak_db:.2f} dB** (está en el rango de 0.0 a 8.9 dB).")

        else: # Menor a 0.0 dB
            status = "APTO"
            st.success(f"✅ **ESTADO: APTO**\n\nEl pico está por debajo de 0.0 dB (**{peak_db:.2f} dB**). Está dentro de los niveles digitales seguros.")

        # Advertencias secundarias de frecuencias
        if sub_bass_max > -3.0:
            reasons.append(f"🟡 **Subgraves Fuertes:** Acumulación alta en frecuencias menores a 45 Hz ({sub_bass_max:.1f} dB rel).")

        if ultra_high_max > -15.0:
            reasons.append(f"🟡 **Agudos Altos:** Frecuencias altas por encima de 16,000 Hz ({ultra_high_max:.1f} dB rel).")

        if reasons:
            st.write("**Detalles del análisis:**")
            for r in reasons:
                st.markdown(r)
                
        # --- GRÁFICA CORREGIDA ---
        st.subheader("📈 Análisis de Espectro (Frecuencias)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs, mean_spectrum_db, color='#8a2be2')
        ax.set_xscale('log')
        ax.set_xlim(20, sr // 2)
        ax.set_ylim(-80, 5)
        ax.set_title("Espectro de Frecuencia Normalizado (Hz vs dBFS)")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Amplitud Relativa (dB)")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        st.pyplot(fig)
