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
        
        # 1. Pico Máximo (dBFS real)
        max_amplitude = np.max(np.abs(y))
        peak_db = 20 * np.log10(max_amplitude) if max_amplitude > 0 else -100.0
        
        # 2. Análisis Espectral Normalizado
        n_fft = 2048
        S = np.abs(librosa.stft(y, n_fft=n_fft))
        S_norm = S / (np.max(S) + 1e-6)
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        mean_spectrum_db = 20 * np.log10(np.mean(S_norm, axis=1) + 1e-6)
        
        # Medición de bandas relativas
        sub_bass_mask = (freqs >= 20) & (freqs <= 60)
        mids_mask = (freqs >= 500) & (freqs <= 2000)
        highs_mask = freqs >= 10000

        sub_bass_val = np.mean(mean_spectrum_db[sub_bass_mask]) if np.any(sub_bass_mask) else -100
        mids_val = np.mean(mean_spectrum_db[mids_mask]) if np.any(mids_mask) else -100
        highs_val = np.mean(mean_spectrum_db[highs_mask]) if np.any(highs_mask) else -100

        # Dominancia de graves respecto a los medios
        bass_dominance = sub_bass_val - mids_val

        # --- EVALUACIÓN DE ESTADO ---
        is_no_apto = False
        is_riesgo = False

        # Regla 1: Picos
        if peak_db > 8.9:
            is_no_apto = True
        elif 0.0 <= peak_db <= 8.9:
            is_riesgo = True

        # Regla 2: Dominancia de Subgraves (Umbrales más estrictos)
        if bass_dominance > 14.0:
            is_no_apto = True
        elif bass_dominance > 9.5:
            is_riesgo = True

        # Regla 3: Agudos ahogados / Audio opaco (< -65 dB)
        if highs_val < -65.0:
            is_riesgo = True

        st.subheader("📊 Resultado de Evaluación")

        if is_no_apto:
            st.error("❌ **ESTADO: NO APTO**\n\nEste archivo excede los límites o tiene un desbalance de graves que Roblox marcará como Disruptive.")
        elif is_riesgo:
            st.warning("⚠️ **ESTADO: PUEDE SER NO APTO!**\n\nEl audio está en zona de riesgo. Los graves dominan sobre los medios o faltan agudos.")
        else:
            st.success("✅ **ESTADO: APTO**\n\nEl archivo cumple con los márgenes de volumen y espectro seguro.")

        # --- DESGLOSE DE PARÁMETROS ---
        st.write("**Desglose detallado del análisis:**")

        # Pico
        if peak_db > 8.9:
            st.markdown(f"🔴 **Pico Máximo:** `{peak_db:.2f} dB` — **Saturación Crítica** (Mayor a 8.9 dB).")
        elif peak_db >= 0.0:
            st.markdown(f"🟡 **Pico Máximo:** `{peak_db:.2f} dB` — **En zona de riesgo** (Entre 0.0 y 8.9 dB).")
        else:
            st.markdown(f"🟢 **Pico Máximo:** `{peak_db:.2f} dB` — **Nivel Seguro** (Menor a 0.0 dB).")

        # Subgraves
        if bass_dominance > 14.0:
            st.markdown(f"🔴 **Balance de Subgraves (<60 Hz):** Dominancia de `{bass_dominance:.1f} dB` sobre medios — **Excesivo** (Riesgo alto de Disruptive Audio).")
        elif bass_dominance > 9.5:
            st.markdown(f"🟡 **Balance de Subgraves (<60 Hz):** Dominancia de `{bass_dominance:.1f} dB` sobre medios — **Elevado** (Riesgo moderado).")
        else:
            st.markdown(f"🟢 **Balance de Subgraves (<60 Hz):** Dominancia de `{bass_dominance:.1f} dB` sobre medios — **Equilibrado**.")

        # Agudos
        if highs_val < -65.0:
            st.markdown(f"🟡 **Agudos (>10 kHz):** Nivel relativo en `{highs_val:.1f} dB` — **Sin agudos (Audio ahogado)**.")
        elif highs_val > -20.0:
            st.markdown(f"🟡 **Agudos (>10 kHz):** Nivel relativo en `{highs_val:.1f} dB` — **Muy Altos**.")
        else:
            st.markdown(f"🟢 **Agudos (>10 kHz):** Nivel relativo en `{highs_val:.1f} dB` — **Equilibrado**.")

        # --- GRÁFICA DE ESPECTRO ---
        st.subheader("📈 Análisis de Espectro (Frecuencias)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs, mean_spectrum_db, color='#8a2be2')
        ax.set_xscale('log')
        ax.set_xlim(20, sr // 2)
        ax.set_ylim(-90, 5)
        ax.axvspan(20, 60, color='red', alpha=0.15, label='Zona de Subgraves')
        ax.set_title("Espectro de Frecuencia Relativo (dBFS)")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Amplitud Relativa (dB)")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        st.pyplot(fig)
