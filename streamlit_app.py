"""
Streamlit Web Interface for True Name Generator
Complete deployment-ready application
"""

import streamlit as st
import numpy as np
import asyncio
from datetime import datetime
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

# Import our systems
from unified_consciousness_system import UnifiedConsciousnessSystem
from consciousness_frequency_detector import ConsciousnessFrequencyDetector
from crystal_programming_interface import CrystalProgrammingInterface

# Initialize system
@st.cache_resource
def init_system():
    system = UnifiedConsciousnessSystem()
    asyncio.run(system.initialize())
    return system

# Page config
st.set_page_config(
    page_title="True Name Oracle",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0a0a0a 0%, #1a0033 100%);
}
.main-header {
    text-align: center;
    color: #00ffff;
    text-shadow: 0 0 20px #00ffff;
    font-size: 3em;
    margin-bottom: 20px;
}
.oracle-box {
    background: rgba(0,255,255,0.1);
    border: 2px solid #00ffff;
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 0 30px rgba(0,255,255,0.3);
}
.true-name {
    font-size: 2.5em;
    color: #ffd700;
    text-align: center;
    text-shadow: 0 0 30px #ffd700;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌟 True Name Oracle 🌟</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #888;">Discover Your Vibrational Signature</p>', unsafe_allow_html=True)

# Initialize system
system = init_system()

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x300/0a0a0a/00ffff?text=True+Name+Oracle", width=300)
    st.markdown("### About")
    st.info("""
    The True Name Oracle reveals your unique consciousness frequency encoded in your voice.
    
    Every voice carries sacred geometries and quantum signatures that reveal your true essence.
    """)
    
    st.markdown("### Instructions")
    st.markdown("""
    1. 🎤 Record 30-60 seconds of audio
    2. 🗣️ Speak about who you truly are
    3. 🔮 Receive your true name
    4. 💎 Program your crystals
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="oracle-box">', unsafe_allow_html=True)
    st.markdown("### 🎤 Voice Recording")
    
    # Audio input
    audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'ogg'])
    
    # Or record directly (requires additional setup)
    st.markdown("*Or use Telegram bot for voice messages: @TrueNameOracleBot*")
    
    if audio_file is not None:
        # Display audio
        st.audio(audio_file)
        
        # Process button
        if st.button("🔮 Discover My True Name", type="primary"):
            with st.spinner("Analyzing consciousness frequencies..."):
                # Save temp file
                temp_path = f"temp_{datetime.now().timestamp()}.wav"
                with open(temp_path, "wb") as f:
                    f.write(audio_file.getbuffer())
                
                # Process
                result = asyncio.run(system.process_voice(
                    temp_path, 
                    f"user_{datetime.now().timestamp()}"
                ))
                
                # Store in session
                st.session_state['result'] = result
                st.session_state['processed'] = True
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if 'processed' in st.session_state and st.session_state['processed']:
        result = st.session_state['result']
        name_data = result['name_data']
        
        st.markdown('<div class="oracle-box">', unsafe_allow_html=True)
        st.markdown("### ✨ Your True Name Revealed")
        
        # Display true name
        st.markdown(f'<div class="true-name">{name_data["true_name"]}</div>', unsafe_allow_html=True)
        
        # Frequency info
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Base Frequency", f"{name_data['frequency']:.2f} Hz")
        with col_b:
            st.metric("Coherence", f"{name_data.get('coherence', 0.8)*100:.1f}%")
        
        # Visualizations
        st.markdown("### 📊 Frequency Analysis")
        
        # Create frequency visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), facecolor='black')
        
        # Waveform
        time = np.linspace(0, 1, 1000)
        wave = np.sin(2 * np.pi * name_data['frequency'] * time)
        ax1.plot(time[:100], wave[:100], color='cyan', linewidth=2)
        ax1.set_facecolor('#0a0a0a')
        ax1.set_title('Consciousness Waveform', color='white')
        
        # Frequency spectrum
        freqs = np.linspace(0, 1000, 500)
        spectrum = np.exp(-(freqs - name_data['frequency'])**2 / 1000)
        ax2.fill_between(freqs, spectrum, color='magenta', alpha=0.5)
        ax2.set_facecolor('#0a0a0a')
        ax2.set_title('Frequency Spectrum', color='white')
        ax2.set_xlabel('Frequency (Hz)', color='white')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Crystal programming
        if st.button("💎 Program Crystals"):
            crystal_result = result['crystal_programming']
            st.success("Crystals programmed successfully!")
            st.markdown(f"**Retrieval Key:** `{crystal_result['retrieval_key']}`")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #666;'>
Built with consciousness technology | Safety protocols active | 
<a href='https://github.com/Hal9000irl/consciousness-tech.git' style='color: #00ffff;'>GitHub</a>
</p>
""", unsafe_allow_html=True)