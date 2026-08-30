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
    
    with st.spinner("Analizando frecuencias y densidad de audio..."):
        # Cargar audio
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # 1. Pico Máximo (dBFS)
        max_amplitude = np.max(np.abs(y))
        peak_db = 20 * np.log10(max_amplitude) if max_amplitude > 0 else -100.0
        
        # 2. Análisis del Espectro
        n_fft = 2048
        S = np.abs(librosa.stft(y, n_fft=n_fft))
        S_norm = S / (np.max(S) + 1e-6)
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        mean_spectrum = np.mean(S_norm, axis=1)
        mean_spectrum_db = 20 * np.log10(mean_spectrum + 1e-6)
        
        # Medir energía en subgraves (< 60 Hz) y agudos (> 15 kHz)
        sub_bass_mask = (freqs >= 20) & (freqs <= 60)
        sub_bass_mean = np.mean(mean_spectrum_db[sub_bass_mask]) if np.any(sub_bass_mask) else -100
        
        ultra_high_mask = freqs >= 15000
        ultra_high_mean = np.mean(mean_spectrum_db[ultra_high_mask]) if np.any(ultra_high_mask) else -100

        # --- REGLAS DE EVALUACIÓN MEJORADAS ---
        st.subheader("📊 Resultado de Evaluación")
        
        reasons = []
        is_disruptive_bass = sub_bass_mean > -18.0  # Subgraves demasiado potentes y continuos

        # Evaluación principal
        if peak_db > 8.9:
            st.error(f"❌ **ESTADO: NO APTO**\n\nEl pico alcanza **{peak_db:.2f} dB**, superando 8.9 dB (Saturación crítica).")
            reasons.append(f"🔴 **Saturación excesiva:** El pico supera los 8.9 dB.")

        elif is_disruptive_bass:
            st.error(f"❌ **ESTADO: NO APTO (DETECTADO RIESGO DE DISRUPTIVE AUDIO)**\n\nEl pico de volumen es seguro ({peak_db:.2f} dB), pero la canción tiene **demasiados subgraves continuos por debajo de 60 Hz** ({sub_bass_mean:.1f} dB medio). Roblox batea esto por 'Disruptive Audio'.")
            reasons.append(f"🔴 **Exceso de Subgraves (<60 Hz):** La presencia de frecuencias ultrabajas es muy alta y plana.")

        elif (peak_db >= 0.0 and peak_db <= 8.9) or sub_bass_mean > -25.0:
            st.warning(f"⚠️ **ESTADO: PUEDE SER NO APTO!**\n\nNivel de pico: **{peak_db:.2f} dB**. Tiene graves marcados o volumen al límite.")
            if sub_bass_mean > -25.0:
                reasons.append(f"🟡 **Subgraves elevados:** Nivel de subgrave medio en {sub_bass_mean:.1f} dB.")
        else:
            st.success(f"✅ **ESTADO: APTO**\n\nEl pico está en **{peak_db:.2f} dB** y las frecuencias están balanceadas.")

        if reasons:
            st.write("**Detalles del análisis:**")
            for r in reasons:
                st.markdown(r)

        # --- GRÁFICA ---
        st.subheader("📈 Análisis de Espectro (Frecuencias)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs, mean_spectrum_db, color='#8a2be2')
        ax.set_xscale('log')
        ax.set_xlim(20, sr // 2)
        ax.set_ylim(-80, 5)
        # Resaltar la zona de peligro por subgraves
        ax.axvspan(20, 60, color='red', alpha=0.15, label='Zona de riesgo por Subgraves')
        ax.set_title("Espectro de Frecuencia Normalizado (Hz vs dBFS)")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Amplitud Relativa (dB)")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        st.pyplot(fig)
