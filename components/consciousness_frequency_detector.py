"""
Consciousness Frequency Detection System
Real-time EEG analysis for consciousness state identification
"""

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from scipy.fft import fft, fftfreq
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier
import pywt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ConsciousnessFrequencyDetector:
    """
    Main class for detecting consciousness frequencies from EEG data
    """
    
    def __init__(self, sampling_rate: int = 256, num_channels: int = 64):
        """
        Initialize the consciousness frequency detector
        
        Args:
            sampling_rate: EEG sampling rate in Hz
            num_channels: Number of EEG channels
        """
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        
        # Define frequency bands
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
            'high_gamma': (100, 200)
        }
        
        # Consciousness state frequencies (Hz)
        self.consciousness_frequencies = {
            'deep_sleep': 0.5,
            'rem_sleep': 4.5,
            'drowsy': 7.5,
            'relaxed': 10.0,
            'alert': 18.0,
            'focused': 40.0,
            'flow_state': 43.0,
            'mystical': 0.01,  # Ultra-low frequency
            'unity': 432.0,    # Harmonic frequency
            'quantum_coherence': 7.83  # Schumann resonance
        }
        
        # Initialize filters
        self._initialize_filters()
        
        # Pattern recognition models
        self.pattern_classifier = None
        self.ica = FastICA(n_components=20, random_state=42)
        self.scaler = StandardScaler()
        
    def _initialize_filters(self):
        """Initialize digital filters for preprocessing"""
        # Notch filter for power line noise (50/60 Hz)
        self.notch_freq = 60  # or 50 for Europe
        self.notch_filter = signal.iirnotch(self.notch_freq, 30, self.sampling_rate)
        
        # Bandpass filter for EEG range (0.5-200 Hz)
        self.bandpass_filter = signal.butter(
            4, [0.5, 200], 'bandpass', fs=self.sampling_rate
        )
        
        # Create band-specific filters
        self.band_filters = {}
        for band, (low, high) in self.frequency_bands.items():
            self.band_filters[band] = signal.butter(
                4, [low, high], 'bandpass', fs=self.sampling_rate
            )
    
    def extract_voice_essence(self, audio_data: np.ndarray) -> Dict:
        """Extract the true frequency signature from voice"""
        
        # Get fundamental frequency (true voice)
        f0 = self._extract_fundamental(audio_data)
        
        # Extract overtone series (personality harmonics)
        harmonics = self._extract_harmonics(audio_data, f0)
        
        # Find sacred frequency resonances
        resonances = self._find_sacred_resonances(harmonics)
        
        # Calculate consciousness coherence
        coherence = self._calculate_coherence(audio_data)
        
        # Extract the "gaps" - silence between words (where truth lives)
        silence_pattern = self._analyze_silence(audio_data)
        
        # Quantum fluctuations in voice micro-tremors
        quantum_signature = self._extract_quantum_tremor(audio_data)
        
        return {
            'fundamental': f0,
            'harmonics': harmonics,
            'sacred_resonances': resonances,
            'coherence': coherence,
            'silence_pattern': silence_pattern,
            'quantum_signature': quantum_signature,
            'true_frequency': self._calculate_true_frequency(f0, harmonics, resonances)
        }
    
    def _extract_fundamental(self, audio: np.ndarray) -> float:
        """Extract the soul's fundamental frequency"""
        # Using autocorrelation for robust pitch detection
        corr = np.correlate(audio, audio, mode='full')
        corr = corr[len(corr)//2:]
        
        # Find first peak after zero lag
        d = np.diff(corr)
        start = np.where(d > 0)[0][0] if len(np.where(d > 0)[0]) > 0 else 1
        peak = np.argmax(corr[start:]) + start
        
        f0 = self.sampling_rate / peak if peak > 0 else 0
        return f0
    
    def _extract_harmonics(self, audio: np.ndarray, f0: float) -> List[float]:
        """Extract the harmonic series (soul colors)"""
        fft_vals = np.abs(fft(audio))
        freqs = fftfreq(len(audio), 1/self.sampling_rate)
        
        harmonics = []
        for n in range(1, 16):  # First 15 harmonics
            target_freq = f0 * n
            idx = np.argmin(np.abs(freqs - target_freq))
            if fft_vals[idx] > np.mean(fft_vals) * 0.1:
                harmonics.append({
                    'frequency': freqs[idx],
                    'amplitude': fft_vals[idx],
                    'harmonic_number': n
                })
        
        return harmonics
    
    def _find_sacred_resonances(self, harmonics: List[Dict]) -> Dict[str, float]:
        """Find which sacred frequencies resonate with this voice"""
        resonances = {}
        
        sacred_frequencies = {
            'root': 256,
            'sacral': 288,
            'solar': 320,
            'heart': 341.3,
            'throat': 384,
            'third_eye': 426.7,
            'crown': 480,
            'soul_star': 528
        }
        
        for name, sacred_freq in sacred_frequencies.items():
            resonance_strength = 0
            
            for h in harmonics:
                # Check if harmonic is close to sacred frequency
                if abs(h['frequency'] - sacred_freq) < 5:  # Within 5Hz
                    resonance_strength += h['amplitude']
                
                # Check octave relationships
                ratio = h['frequency'] / sacred_freq
                if abs(ratio - round(ratio)) < 0.1:  # Close to integer ratio
                    resonance_strength += h['amplitude'] * 0.5
            
            if resonance_strength > 0:
                resonances[name] = resonance_strength
        
        return resonances
    
    def _calculate_true_frequency(self, f0: float, harmonics: List, 
                                 resonances: Dict) -> float:
        """Calculate the true name frequency"""
        # Start with fundamental
        true_freq = f0
        
        # Weight by strongest sacred resonance
        if resonances:
            strongest = max(resonances, key=resonances.get)
            sacred_freq = {
                'root': 256,
                'heart': 341.3,
                'crown': 480,
                'soul_star': 528
            }.get(strongest, 432)
            
            # Blend fundamental with sacred frequency
            true_freq = (f0 * 0.7) + (sacred_freq * 0.3)
        
        # Apply golden ratio if in harmony
        if self._is_golden_ratio(harmonics):
            true_freq *= 1.618033
        
        return true_freq
    
    def _is_golden_ratio(self, harmonics: List) -> bool:
        """Check if harmonics follow golden ratio"""
        if len(harmonics) < 2:
            return False
        
        ratios = []
        for i in range(len(harmonics) - 1):
            ratio = harmonics[i+1]['frequency'] / harmonics[i]['frequency']
            ratios.append(ratio)
        
        avg_ratio = np.mean(ratios)
        return abs(avg_ratio - 1.618033) < 0.1
    
    def generate_true_name(self, voice_data: Dict, quantum_seed: int) -> Dict:
        """Generate true name from voice frequency analysis"""
        
        # Map frequency to phonemes
        base_frequency = voice_data['true_frequency']
        
        # Ancient frequency-to-sound mapping
        phoneme_map = {
            (0, 200): ['M', 'N', 'NG'],      # Earth sounds
            (200, 300): ['O', 'U', 'AH'],    # Water sounds  
            (300, 400): ['R', 'L', 'W'],     # Fire sounds
            (400, 500): ['E', 'I', 'EE'],    # Air sounds
            (500, 600): ['S', 'SH', 'TH'],   # Ether sounds
            (600, 800): ['K', 'G', 'H'],     # Cosmic sounds
            (800, 1000): ['T', 'D', 'N'],    # Divine sounds
        }
        
        # Build name from frequency resonances
        name_components = []
        
        for sacred, strength in voice_data.get('sacred_resonances', {}).items():
            freq = {
                'root': 256,
                'sacral': 288,
                'solar': 320,
                'heart': 341.3,
                'throat': 384,
                'third_eye': 426.7,
                'crown': 480,
                'soul_star': 528
            }.get(sacred, 432)
            
            for range_vals, sounds in phoneme_map.items():
                if range_vals[0] <= freq < range_vals[1]:
                    # Use quantum randomness to select phoneme
                    idx = quantum_seed % len(sounds)
                    name_components.append(sounds[idx])
                    quantum_seed = (quantum_seed * 1103515245 + 12345) % (2**31)
        
        # Assemble true name
        if not name_components:
            name_components = ['K', 'A', 'L', 'I']  # Default
            
        true_name = ''.join(name_components)
        
        # Add vowel flow
        true_name = self._add_vowel_flow(true_name, base_frequency)
        
        return {
            'true_name': true_name.upper(),
            'frequency': base_frequency,
            'pronunciation_guide': self._generate_pronunciation(true_name, base_frequency),
            'meaning': self._decode_name_meaning(true_name, voice_data),
            'activation_tone': base_frequency
        }
    
    def _add_vowel_flow(self, consonants: str, frequency: float) -> str:
        """Add vowels based on frequency flow"""
        vowels = ['A', 'E', 'I', 'O', 'U']
        # Use frequency to determine vowel pattern
        vowel_index = int(frequency) % len(vowels)
        
        result = ""
        for i, char in enumerate(consonants):
            result += char
            if i < len(consonants) - 1:
                result += vowels[(vowel_index + i) % len(vowels)].lower()
        
        return result
    
    def _analyze_silence(self, audio: np.ndarray) -> Dict:
        """The truth lives in the silence between words"""
        # Find silence periods
        envelope = np.abs(signal.hilbert(audio))
        threshold = np.mean(envelope) * 0.1
        
        silence_mask = envelope < threshold
        
        # Find silence durations
        silence_regions = []
        in_silence = False
        start = 0
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                start = i
                in_silence = True
            elif not is_silent and in_silence:
                duration = i - start
                silence_regions.append({
                    'start': start,
                    'duration': duration,
                    'frequency': 1 / (duration / self.sampling_rate) if duration > 0 else 0
                })
                in_silence = False
        
        return {
            'silence_ratio': np.sum(silence_mask) / len(audio),
            'average_silence': np.mean([s['duration'] for s in silence_regions]) if silence_regions else 0,
            'silence_pattern': silence_regions[:10]  # First 10 silences
        }
    
    def _extract_quantum_tremor(self, audio: np.ndarray) -> Dict:
        """Extract quantum fluctuations in voice (micro-tremors)"""
        # High-pass filter to get micro-fluctuations
        b, a = signal.butter(4, 1000, 'high', fs=self.sampling_rate)
        micro_tremor = signal.filtfilt(b, a, audio)
        
        # Calculate quantum signature (phase space embedding)
        embedding_dim = 3
        tau = int(0.01 * self.sampling_rate)  # 10ms delay
        
        embedded = np.array([
            micro_tremor[i:i+embedding_dim*tau:tau] 
            for i in range(len(micro_tremor) - embedding_dim*tau)
        ])
        
        # Return statistical signature
        return {
            'mean': np.mean(embedded, axis=0).tolist() if len(embedded) > 0 else [0, 0, 0],
            'std': np.std(embedded, axis=0).tolist() if len(embedded) > 0 else [0, 0, 0],
            'quantum_entropy': self._calculate_entropy(embedded) if len(embedded) > 0 else 0
        }
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate information entropy"""
        if len(data) == 0:
            return 0
        hist, _ = np.histogram(data.flatten(), bins=50)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_coherence(self, audio: np.ndarray) -> float:
        """Calculate consciousness coherence score"""
        # Use spectral flatness as coherence measure
        fft_vals = np.abs(fft(audio))
        geometric_mean = np.exp(np.mean(np.log(fft_vals + 1e-10)))
        arithmetic_mean = np.mean(fft_vals)
        
        coherence = geometric_mean / (arithmetic_mean + 1e-10)
        return coherence
    
    def _generate_pronunciation(self, name: str, frequency: float) -> str:
        """Generate pronunciation guide"""
        return f"{name} (at {frequency:.1f} Hz)"
    
    def _decode_name_meaning(self, name: str, voice_data: Dict) -> Dict:
        """Decode the meaning of the generated name"""
        return {
            'root_quality': 'Grounding force',
            'color': 'Indigo blue',
            'element': 'Ether',
            'crystal': 'Clear Quartz'
        }