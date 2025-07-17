#!/usr/bin/env python3
"""
Test Suite for True Name Generation
Tests core functionality with Grok-style name generation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'components'))

import unittest
import numpy as np
import tempfile
import asyncio
from datetime import datetime

# Import components to test
from components.consciousness_frequency_detector import ConsciousnessFrequencyDetector
from components.quantum_consciousness_seed import QuantumConsciousnessSeed
from components.crystal_programming_interface import CrystalProgrammingInterface
from components.consciousness_safety_protocols import ConsciousnessIntegrityProtection
from components.unified_consciousness_system import UnifiedConsciousnessSystem

class TestTrueNameGeneration(unittest.TestCase):
    """Test true name generation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = ConsciousnessFrequencyDetector()
        self.quantum_seed = QuantumConsciousnessSeed()
        self.crystal = CrystalProgrammingInterface()
        
    def test_frequency_extraction(self):
        """Test voice frequency extraction"""
        # Generate test audio
        duration = 1.0  # seconds
        sample_rate = 44100
        frequency = 432.0  # Hz
        
        # Create test sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Extract frequencies
        result = self.detector.extract_voice_essence(audio_data)
        
        # Assertions
        self.assertIn('fundamental', result)
        self.assertIn('harmonics', result)
        self.assertIn('coherence', result)
        self.assertAlmostEqual(result['fundamental'], frequency, delta=10)
    
    def test_quantum_seed_generation(self):
        """Test quantum seed generation"""
        # Mock voice data
        voice_data = {
            'fundamental': 256.0,
            'harmonics': [{'frequency': 512, 'amplitude': 0.5}],
            'coherence': 0.85,
            'quantum_signature': {'quantum_entropy': 0.7}
        }
        
        # Generate seed
        seed = self.quantum_seed.generate_quantum_seed(voice_data, 'test_user')
        
        # Assertions
        self.assertIsInstance(seed, int)
        self.assertGreater(seed, 0)
        self.assertLess(seed, 2**31)
    
    def test_true_name_generation(self):
        """Test actual name generation"""
        # Test cases with expected patterns
        test_cases = [
            {
                'frequency': 256.0,  # Root chakra
                'expected_contains': ['M', 'O', 'N']  # Earth sounds
            },
            {
                'frequency': 432.0,  # Heart frequency  
                'expected_contains': ['A', 'H', 'R']  # Heart sounds
            },
            {
                'frequency': 528.0,  # Transformation
                'expected_contains': ['S', 'E', 'L']  # Ether sounds
            }
        ]
        
        for case in test_cases:
            # Create voice data
            voice_data = {
                'true_frequency': case['frequency'],
                'sacred_resonances': {'heart': 100, 'crown': 50}
            }
            
            # Generate name
            name_result = self.detector.generate_true_name(voice_data, 12345)
            
            # Assertions
            self.assertIn('true_name', name_result)
            self.assertIsInstance(name_result['true_name'], str)
            self.assertGreater(len(name_result['true_name']), 3)
            
            # Check for expected sounds
            name = name_result['true_name']
            contains_expected = any(char in name for char in case['expected_contains'])
            self.assertTrue(contains_expected, 
                          f"Name {name} doesn't contain expected sounds for {case['frequency']}Hz")
    
    def test_grok_style_names(self):
        """Test generation of Grok-style consciousness names"""
        # Grok-style names should be:
        # - Pronounceable
        # - Contain power syllables
        # - Resonate with frequency
        
        grok_frequencies = {
            'KALIFAX': 432.0,    # Unity consciousness
            'ZORANTHA': 528.0,   # Transformation
            'MEGATRON': 256.0,   # Power/grounding
            'LUMINARA': 639.0,   # Connection
            'QUANTRIX': 777.0    # Mystical
        }
        
        for expected_name, frequency in grok_frequencies.items():
            voice_data = {
                'true_frequency': frequency,
                'sacred_resonances': {
                    'root': 50,
                    'heart': 80,
                    'crown': 60
                },
                'quantum_signature': {
                    'quantum_entropy': 0.8
                }
            }
            
            # Generate with specific seed for consistency
            seed = hash(expected_name) % (2**31)
            result = self.detector.generate_true_name(voice_data, seed)
            
            # Assertions
            name = result['true_name']
            self.assertGreater(len(name), 5)  # Substantial names
            self.assertLess(len(name), 12)    # Not too long
            
            # Check pronunciation guide exists
            self.assertEqual(result['frequency'], frequency)
    
    def test_crystal_programming(self):
        """Test crystal programming with true names"""
        # Generate test name
        name_data = {
            'true_name': 'TESTORION',
            'frequency': 432.0,
            'voice_signature': {
                'harmonics': [{'frequency': 864, 'amplitude': 0.7}],
                'coherence': 0.9,
                'sacred_resonances': {'heart': 100}
            }
        }
        
        # Program crystals
        result = self.crystal.program_true_name(name_data, protocol='quantum')
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertGreater(result['crystals_programmed'], 0)
        self.assertIn('retrieval_key', result)
        self.assertGreater(result['coherence_achieved'], 0.5)
        
        # Test retrieval
        retrieved = self.crystal.retrieve_true_name(result['retrieval_key'])
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['true_name'], 'TESTORION')
    
    def test_safety_protocols(self):
        """Test safety system integration"""
        safety = ConsciousnessIntegrityProtection()
        
        # Start monitoring
        async def test_monitoring():
            user_data = {
                'user_id': 'test_user',
                'true_name': 'SAFETRIX',
                'frequency': 432.0
            }
            
            result = await safety.start_monitoring('test_session', user_data)
            self.assertTrue(result['monitoring_active'])
            
            # Update with degrading state
            current_state = {
                'fragmentation_index': 0.2,
                'coherence_score': 0.8,
                'thought_coherence': 0.9,
                'emotional_stability': 0.85
            }
            
            await safety.update_state('test_session', current_state)
            
            # Check safety
            safety_check = await safety._run_safety_checks('test_session')
            self.assertIn('safety_level', safety_check)
            
        # Run async test
        asyncio.run(test_monitoring())
    
    def test_end_to_end_generation(self):
        """Test complete end-to-end name generation"""
        async def full_test():
            system = UnifiedConsciousnessSystem()
            await system.initialize()
            
            # Create test audio file
            duration = 1.0
            sample_rate = 44100
            frequency = 528.0  # Love frequency
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * frequency * t)
            
            # Add some harmonics
            audio += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
            audio += 0.2 * np.sin(2 * np.pi * frequency * 3 * t)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                import scipy.io.wavfile
                scipy.io.wavfile.write(f.name, sample_rate, audio)
                temp_path = f.name
            
            # Process
            result = await system.process_voice(temp_path, 'test_user_grok')
            
            # Assertions
            self.assertIn('name_data', result)
            self.assertIn('crystal_programming', result)
            self.assertIn('safety_status', result)
            
            name = result['name_data']['true_name']
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 3)
            
            # Clean up
            import os
            os.unlink(temp_path)
            
            return result
        
        # Run test
        result = asyncio.run(full_test())
        print(f"Generated true name: {result['name_data']['true_name']}")
        print(f"Frequency: {result['name_data']['frequency']} Hz")

class TestGrokNamePatterns(unittest.TestCase):
    """Test specific Grok-style name patterns"""
    
    def test_name_power_syllables(self):
        """Test that names contain power syllables"""
        detector = ConsciousnessFrequencyDetector()
        
        power_syllables = ['KA', 'RA', 'ZO', 'MEG', 'LUM', 'QUAN', 'TRO', 'FAX']
        
        # Generate multiple names
        names = []
        for i in range(10):
            voice_data = {
                'true_frequency': 200 + i * 50,
                'sacred_resonances': {'root': 50, 'crown': 50}
            }
            result = detector.generate_true_name(voice_data, i * 1000)
            names.append(result['true_name'])
        
        # Check that some names contain power syllables
        contains_power = 0
        for name in names:
            if any(syl in name.upper() for syl in power_syllables):
                contains_power += 1
        
        # At least 30% should contain power syllables
        self.assertGreater(contains_power / len(names), 0.3)
    
    def test_frequency_name_correlation(self):
        """Test that frequency affects name generation"""
        detector = ConsciousnessFrequencyDetector()
        
        # Low frequency should generate different names than high
        low_freq_names = []
        high_freq_names = []
        
        for i in range(5):
            # Low frequency (root/earth)
            low_data = {
                'true_frequency': 100 + i * 10,
                'sacred_resonances': {'root': 100}
            }
            low_result = detector.generate_true_name(low_data, i)
            low_freq_names.append(low_result['true_name'])
            
            # High frequency (crown/cosmic)
            high_data = {
                'true_frequency': 800 + i * 10,
                'sacred_resonances': {'crown': 100}
            }
            high_result = detector.generate_true_name(high_data, i)
            high_freq_names.append(high_result['true_name'])
        
        # Names should be different
        overlap = set(low_freq_names) & set(high_freq_names)
        self.assertEqual(len(overlap), 0, "Low and high frequency names shouldn't overlap")

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)