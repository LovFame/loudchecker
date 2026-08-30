import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt

st.set_page_config(page_title="Roblox Audio Moderation Checker", page_icon="🎵")

st.title("🎵 Detector de Moderación de Audio para Roblox")
st.write("Sube tu archivo de audio para analizar picos, saturación y subgraves.")

uploaded_file = st.file_uploader("Selecciona tu archivo de audio (MP3 o WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    with st.spinner("Analizando audio completo..."):
        # Cargar audio
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # 1. Pico Máximo
        max_amplitude = np.max(np.abs(y))
        peak_db = 20 * np.log10(max_amplitude) if max_amplitude > 0 else -100.0
        
        # 2. Análisis Espectral (FFT)
        n_fft = 2048
        S = np.abs(librosa.stft(y, n_fft=n_fft))
        S_norm = S / (np.max(S) + 1e-6)
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        mean_spectrum = np.mean(S_norm, axis=1)
        mean_spectrum_db = 20 * np.log10(mean_spectrum + 1e-6)
        
        # Frecuencias de riesgo
        sub_bass_mask = (freqs >= 20) & (freqs <= 60)
        sub_bass_mean = np.mean(mean_spectrum_db[sub_bass_mask]) if np.any(sub_bass_mask) else -100
        
        ultra_high_mask = freqs >= 15000
        ultra_high_mean = np.mean(mean_spectrum_db[ultra_high_mask]) if np.any(ultra_high_mask) else -100

        # --- REVISIÓN INDEPENDIENTE DE TODOS LOS DETALLES ---
        reasons = []
        is_no_apto = False
        is_riesgo = False

        # Evaluador 1: Picos de volumen
        if peak_db > 8.9:
            is_no_apto = True
            reasons.append(f"🔴 **Saturación Crítica:** El pico alcanza **{peak_db:.2f} dB** (Mayor a 8.9 dB).")
        elif peak_db >= 0.0 and peak_db <= 8.9:
            is_riesgo = True
            reasons.append(f"🟡 **Pico en Zona de Riesgo:** El pico alcanza **{peak_db:.2f} dB** (Entre 0.0 y 8.9 dB).")

        # Evaluador 2: Subgraves continuos
        if sub_bass_mean > -18.0:
            is_no_apto = True
            reasons.append(f"🔴 **Subgraves Excesivos (<60 Hz):** La presencia de frecuencias ultrabajas es muy alta ({sub_bass_mean:.1f} dB medio). Riesgo de Disruptive Audio.")
        elif sub_bass_mean > -25.0:
            is_riesgo = True
            reasons.append(f"🟡 **Subgraves Elevados:** Nivel de subgrave medio en {sub_bass_mean:.1f} dB.")

        # Evaluador 3: Agudos estridentes
        if ultra_high_mean > -20.0:
            is_riesgo = True
            reasons.append(f"🟡 **Agudos Altos (>15 kHz):** Acumulación alta en frecuencias ultrasónicas ({ultra_high_mean:.1f} dB medio).")

        # --- MOSTRAR RESULTADO FINAL Y DETALLES ---
        st.subheader("📊 Resultado de Evaluación")

        if is_no_apto:
            st.error(f"❌ **ESTADO: NO APTO**\n\nEste archivo tiene altas probabilidades de ser rechazado o marcado como Disruptive Audio.")
        elif is_riesgo:
            st.warning(f"⚠️ **ESTADO: PUEDE SER NO APTO!**\n\nEl audio está en zona límite. Podría pasar o ser rechazado dependiendo de la moderación.")
        else:
            st.success(f"✅ **ESTADO: APTO**\n\nEl pico está por debajo de 0.0 dB ({peak_db:.2f} dB) y las frecuencias están equilibradas.")

        if reasons:
            st.write("**Detalles del análisis encontrados:**")
            for r in reasons:
                st.markdown(r)

        # --- GRÁFICA DE ESPECTRO ---
        st.subheader("📈 Análisis de Espectro (Frecuencias)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs, mean_spectrum_db, color='#8a2be2')
        ax.set_xscale('log')
        ax.set_xlim(20, sr // 2)
        ax.set_ylim(-80, 5)
        ax.axvspan(20, 60, color='red', alpha=0.15, label='Zona de riesgo por Subgraves')
        ax.set_title("Espectro de Frecuencia Normalizado (Hz vs dBFS)")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Amplitud Relativa (dB)")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        st.pyplot(fig)
