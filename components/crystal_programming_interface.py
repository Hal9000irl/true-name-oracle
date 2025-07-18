"""
Crystal Programming Interface for True Name Storage
Encodes consciousness frequencies into crystalline matrices
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from datetime import datetime
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
import matplotlib.patches as mpatches
import time

@dataclass
class CrystalMatrix:
    """Individual crystal data structure"""
    crystal_id: str
    crystal_type: str
    position: Tuple[float, float, float]  # x, y, z coordinates
    frequency_capacity: float  # Hz
    current_program: Optional[Dict] = None
    activation_level: float = 0.0
    coherence_factor: float = 1.0
    last_programmed: Optional[datetime] = None

class CrystalProgrammingInterface:
    """
    Interface for programming consciousness frequencies into crystals
    Integrates with True Name Generator for permanent storage
    """
    
    def __init__(self):
        # Crystal types and their properties
        self.crystal_properties = {
            'clear_quartz': {
                'frequency_range': (0, 20000),
                'amplification': 3.0,
                'storage_capacity': 1000,  # frequency patterns
                'coherence_decay': 0.001,  # per day
                'color': '#FFFFFF'
            },
            'amethyst': {
                'frequency_range': (400, 800),
                'amplification': 2.5,
                'storage_capacity': 500,
                'coherence_decay': 0.0005,
                'color': '#9370DB'
            },
            'rose_quartz': {
                'frequency_range': (300, 500),
                'amplification': 2.0,
                'storage_capacity': 300,
                'coherence_decay': 0.0001,
                'color': '#FFB6C1'
            },
            'black_tourmaline': {
                'frequency_range': (0, 300),
                'amplification': 1.5,
                'storage_capacity': 200,
                'coherence_decay': 0.00001,  # very stable
                'color': '#000000'
            },
            'selenite': {
                'frequency_range': (500, 1000),
                'amplification': 4.0,
                'storage_capacity': 100,  # pure but limited
                'coherence_decay': 0.01,  # needs recharging
                'color': '#F0F8FF'
            },
            'moldavite': {
                'frequency_range': (1000, 10000),
                'amplification': 10.0,  # extreme amplification
                'storage_capacity': 50,
                'coherence_decay': 0.0,  # eternal
                'color': '#228B22'
            }
        }
        
        # Sacred geometry positions (2D projection for visualization)
        self.sacred_positions = self._calculate_sacred_geometry()
        
        # Initialize crystal array
        self.crystal_array = self._initialize_crystal_array()
        
        # Programming protocols
        self.protocols = {
            'basic': self._basic_programming,
            'harmonic': self._harmonic_programming,
            'quantum': self._quantum_programming,
            'holographic': self._holographic_programming
        }
    
    def _calculate_sacred_geometry(self) -> Dict[str, List[Tuple[float, float]]]:
        """Calculate sacred geometry positions for crystal placement"""
        positions = {}
        
        # Metatron's Cube (13 positions)
        positions['metatron'] = []
        # Center
        positions['metatron'].append((0, 0))
        # Inner hexagon
        for i in range(6):
            angle = i * np.pi / 3
            x = np.cos(angle) * 1.0
            y = np.sin(angle) * 1.0
            positions['metatron'].append((x, y))
        # Outer hexagon
        for i in range(6):
            angle = i * np.pi / 3 + np.pi / 6
            x = np.cos(angle) * 1.732  # sqrt(3)
            y = np.sin(angle) * 1.732
            positions['metatron'].append((x, y))
        
        # Flower of Life (19 positions)
        positions['flower'] = [(0, 0)]  # Center
        # First ring
        for i in range(6):
            angle = i * np.pi / 3
            x = np.cos(angle) * 1.0
            y = np.sin(angle) * 1.0
            positions['flower'].append((x, y))
        # Second ring
        for i in range(12):
            angle = i * np.pi / 6
            r = 1.0 if i % 2 == 0 else 1.732
            x = np.cos(angle) * r
            y = np.sin(angle) * r
            positions['flower'].append((x, y))
        
        # Sri Yantra (simplified - 9 triangles, 43 points)
        # This is complex - using key points only
        positions['sri_yantra'] = [
            (0, 0),  # Bindu (center)
            # Inner triangle
            (0, 0.5), (-0.433, -0.25), (0.433, -0.25),
            # Middle triangles
            (0, -0.5), (-0.866, 0.5), (0.866, 0.5),
            # Outer triangles
            (0, 1.5), (-1.299, -0.75), (1.299, -0.75)
        ]
        
        return positions
    
    def _initialize_crystal_array(self) -> List[CrystalMatrix]:
        """Initialize the crystal array with default configuration"""
        array = []
        
        # Use Metatron's Cube as default
        positions = self.sacred_positions['metatron']
        
        # Assign crystals to positions
        crystal_types = ['clear_quartz', 'amethyst', 'rose_quartz', 
                        'black_tourmaline', 'selenite']
        
        for i, (x, y) in enumerate(positions):
            # Center is always clear quartz (master crystal)
            if i == 0:
                crystal_type = 'clear_quartz'
            else:
                # Distribute other crystals
                crystal_type = crystal_types[i % len(crystal_types)]
            
            crystal = CrystalMatrix(
                crystal_id=f"CRYSTAL_{i:03d}",
                crystal_type=crystal_type,
                position=(x, y, 0),  # z=0 for 2D layout
                frequency_capacity=self.crystal_properties[crystal_type]['frequency_range'][1]
            )
            
            array.append(crystal)
        
        return array
    
    def program_true_name(self, true_name_data: Dict, 
                         protocol: str = 'quantum') -> Dict:
        """
        Program a true name into the crystal array
        
        Args:
            true_name_data: Output from TrueNameGenerator
            protocol: Programming protocol to use
            
        Returns:
            Programming results and crystal activation map
        """
        
        # Extract key frequencies
        base_frequency = true_name_data['frequency']
        name = true_name_data['true_name']
        voice_signature = true_name_data.get('voice_signature', {})
        
        # Select appropriate crystals based on frequency
        selected_crystals = self._select_crystals_for_frequency(base_frequency)
        
        # Apply programming protocol
        programming_func = self.protocols.get(protocol, self._quantum_programming)
        results = programming_func(selected_crystals, true_name_data)
        time.sleep(0.5)
        
        # Create activation map
        activation_map = self._create_activation_visualization(results)
        
        return {
            'success': True,
            'crystals_programmed': len(results['programmed_crystals']),
            'total_amplitude': results['total_amplitude'],
            'coherence_achieved': results['coherence'],
            'storage_duration': results['estimated_storage_days'],
            'activation_map': activation_map,
            'retrieval_key': self._generate_retrieval_key(name, base_frequency),
            'programming_timestamp': datetime.now()
        }
    
    def _select_crystals_for_frequency(self, frequency: float) -> List[CrystalMatrix]:
        """Select appropriate crystals based on frequency compatibility"""
        selected = []
        
        for crystal in self.crystal_array:
            props = self.crystal_properties[crystal.crystal_type]
            freq_range = props['frequency_range']
            
            # Check if frequency is within crystal's range
            if freq_range[0] <= frequency <= freq_range[1]:
                selected.append(crystal)
            # Also select if harmonic is within range
            elif freq_range[0] <= frequency * 2 <= freq_range[1]:
                selected.append(crystal)
            elif freq_range[0] <= frequency / 2 <= freq_range[1]:
                selected.append(crystal)
        
        # Always include center crystal (master)
        if self.crystal_array[0] not in selected:
            selected.insert(0, self.crystal_array[0])
        
        return selected
    
    def _quantum_programming(self, crystals: List[CrystalMatrix], 
                           name_data: Dict) -> Dict:
        """Quantum holographic programming protocol"""
        
        programmed_crystals = []
        total_amplitude = 0
        
        # Extract quantum signature
        quantum_sig = name_data.get('voice_signature', {}).get('quantum_signature', {})
        base_freq = name_data['frequency']
        
        # Create holographic interference pattern
        for crystal in crystals:
            # Calculate crystal-specific frequency
            props = self.crystal_properties[crystal.crystal_type]
            
            # Quantum entangle with base frequency
            crystal_freq = self._quantum_frequency_entangle(
                base_freq, 
                props['frequency_range'],
                quantum_sig
            )
            
            # Create program data
            program = {
                'true_name': name_data['true_name'],
                'base_frequency': base_freq,
                'crystal_frequency': crystal_freq,
                'quantum_signature': quantum_sig,
                'harmonics': name_data.get('voice_signature', {}).get('harmonics', []),
                'sacred_resonances': name_data.get('voice_signature', {}).get('sacred_resonances', {}),
                'programming_protocol': 'quantum',
                'entanglement_id': self._generate_entanglement_id()
            }
            
            # Program crystal
            crystal.current_program = program
            crystal.activation_level = props['amplification']
            crystal.last_programmed = datetime.now()
            
            programmed_crystals.append(crystal)
            total_amplitude += crystal.activation_level
        
        # Calculate array coherence
        coherence = self._calculate_array_coherence(programmed_crystals)
        
        # Estimate storage duration
        min_decay = min(
            self.crystal_properties[c.crystal_type]['coherence_decay'] 
            for c in programmed_crystals
        )
        storage_days = -np.log(0.5) / min_decay if min_decay > 0 else float('inf')
        
        return {
            'programmed_crystals': programmed_crystals,
            'total_amplitude': total_amplitude,
            'coherence': coherence,
            'estimated_storage_days': storage_days
        }
    
    def _quantum_frequency_entangle(self, base_freq: float, 
                                   freq_range: Tuple[float, float],
                                   quantum_sig: Dict) -> float:
        """Quantum entangle frequency with crystal range"""
        
        # If base frequency is in range, use it
        if freq_range[0] <= base_freq <= freq_range[1]:
            return base_freq
        
        # Find nearest harmonic within range
        harmonics = []
        for n in range(1, 16):
            # Check harmonics and subharmonics
            harm_up = base_freq * n
            harm_down = base_freq / n
            
            if freq_range[0] <= harm_up <= freq_range[1]:
                harmonics.append(harm_up)
            if freq_range[0] <= harm_down <= freq_range[1]:
                harmonics.append(harm_down)
        
        if harmonics:
            # Use quantum signature to select harmonic
            if quantum_sig and 'quantum_entropy' in quantum_sig:
                idx = int(quantum_sig['quantum_entropy'] * len(harmonics)) % len(harmonics)
                return harmonics[idx]
            else:
                return harmonics[0]
        
        # Default to middle of range
        return (freq_range[0] + freq_range[1]) / 2
    
    def _calculate_array_coherence(self, crystals: List[CrystalMatrix]) -> float:
        """Calculate coherence of programmed crystal array"""
        
        if len(crystals) < 2:
            return 1.0
        
        # Extract frequencies
        frequencies = [
            c.current_program['crystal_frequency'] 
            for c in crystals 
            if c.current_program
        ]
        
        if len(frequencies) < 2:
            return 0.0
        
        # Check for harmonic relationships
        coherence_score = 0
        comparisons = 0
        
        for i in range(len(frequencies)):
            for j in range(i + 1, len(frequencies)):
                ratio = frequencies[i] / frequencies[j]
                
                # Check if ratio is close to simple integer ratio
                for n in range(1, 16):
                    for m in range(1, 16):
                        if abs(ratio - n/m) < 0.01:
                            coherence_score += 1 / (n + m)  # Simpler ratios score higher
                            break
                
                comparisons += 1
        
        return coherence_score / comparisons if comparisons > 0 else 0
    
    def _generate_entanglement_id(self) -> str:
        """Generate unique quantum entanglement ID"""
        timestamp = int(datetime.now().timestamp() * 1e6)
        random_component = np.random.randint(0, 65536)
        return f"QE-{timestamp:X}-{random_component:04X}"
    
    def _generate_retrieval_key(self, name: str, frequency: float) -> str:
        """Generate key for retrieving stored true name"""
        # Combine name and frequency into retrieval key
        name_component = ''.join([str(ord(c)) for c in name[:3]])
        freq_component = int(frequency * 100)
        return f"TN-{name_component}-{freq_component}"
    
    def _basic_programming(self, crystals: List[CrystalMatrix], 
                          name_data: Dict) -> Dict:
        """Basic frequency storage protocol"""
        programmed = []
        
        for crystal in crystals[:3]:  # Use only first 3 crystals
            program = {
                'true_name': name_data['true_name'],
                'frequency': name_data['frequency'],
                'programming_protocol': 'basic'
            }
            
            crystal.current_program = program
            crystal.activation_level = 0.5
            crystal.last_programmed = datetime.now()
            programmed.append(crystal)
        
        return {
            'programmed_crystals': programmed,
            'total_amplitude': len(programmed) * 0.5,
            'coherence': 0.5,
            'estimated_storage_days': 30
        }
    
    def _harmonic_programming(self, crystals: List[CrystalMatrix], 
                             name_data: Dict) -> Dict:
        """Harmonic series programming protocol"""
        programmed = []
        base_freq = name_data['frequency']
        
        # Program harmonics across crystals
        for i, crystal in enumerate(crystals[:8]):
            harmonic_number = i + 1
            harmonic_freq = base_freq * harmonic_number
            
            # Check if harmonic fits in crystal
            props = self.crystal_properties[crystal.crystal_type]
            if props['frequency_range'][0] <= harmonic_freq <= props['frequency_range'][1]:
                program = {
                    'true_name': name_data['true_name'],
                    'base_frequency': base_freq,
                    'harmonic_frequency': harmonic_freq,
                    'harmonic_number': harmonic_number,
                    'programming_protocol': 'harmonic'
                }
                
                crystal.current_program = program
                crystal.activation_level = props['amplification'] * (1 / harmonic_number)
                crystal.last_programmed = datetime.now()
                programmed.append(crystal)
        
        total_amp = sum(c.activation_level for c in programmed)
        
        return {
            'programmed_crystals': programmed,
            'total_amplitude': total_amp,
            'coherence': 0.8,
            'estimated_storage_days': 60
        }
    
    def _holographic_programming(self, crystals: List[CrystalMatrix], 
                                name_data: Dict) -> Dict:
        """Holographic distributed storage protocol"""
        
        # Each crystal stores complete information at different phase
        programmed = []
        base_freq = name_data['frequency']
        
        for i, crystal in enumerate(crystals):
            phase_shift = (i * 2 * np.pi) / len(crystals)
            
            program = {
                'true_name': name_data['true_name'],
                'base_frequency': base_freq,
                'phase_shift': phase_shift,
                'holographic_fragment': i,
                'total_fragments': len(crystals),
                'voice_signature': name_data.get('voice_signature', {}),
                'programming_protocol': 'holographic'
            }
            
            crystal.current_program = program
            props = self.crystal_properties[crystal.crystal_type]
            crystal.activation_level = props['amplification']
            crystal.last_programmed = datetime.now()
            programmed.append(crystal)
        
        return {
            'programmed_crystals': programmed,
            'total_amplitude': sum(c.activation_level for c in programmed),
            'coherence': 0.95,
            'estimated_storage_days': 365
        }
    
    def _create_activation_visualization(self, results: Dict) -> np.ndarray:
        """Create visual representation of crystal activation"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                                       facecolor='black')
        
        # Crystal array visualization
        ax1.set_facecolor('#0a0a0a')
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        ax1.set_aspect('equal')
        ax1.set_title('Crystal Array Activation', color='white', fontsize=16)
        
        # Draw connections between activated crystals
        programmed = results['programmed_crystals']
        for i, crystal1 in enumerate(programmed):
            for j, crystal2 in enumerate(programmed[i+1:], i+1):
                x1, y1, _ = crystal1.position
                x2, y2, _ = crystal2.position
                
                # Connection strength based on coherence
                alpha = results['coherence'] * 0.5
                ax1.plot([x1, x2], [y1, y2], 'c-', alpha=alpha, linewidth=1)
        
        # Draw crystals
        for crystal in self.crystal_array:
            x, y, _ = crystal.position
            
            # Check if programmed
            is_programmed = crystal in programmed
            
            # Crystal color and size
            if is_programmed:
                color = self.crystal_properties[crystal.crystal_type]['color']
                size = 300 * crystal.activation_level
                edge_color = 'gold'
                edge_width = 3
            else:
                color = '#333333'
                size = 100
                edge_color = 'gray'
                edge_width = 1
            
            # Draw crystal
            if crystal.crystal_type == 'clear_quartz':
                # Hexagon for quartz
                hex_patch = RegularPolygon((x, y), 6, radius=0.2,
                                         facecolor=color,
                                         edgecolor=edge_color,
                                         linewidth=edge_width)
                ax1.add_patch(hex_patch)
            else:
                # Circle for others
                circle = Circle((x, y), 0.15, facecolor=color,
                              edgecolor=edge_color, linewidth=edge_width)
                ax1.add_patch(circle)
            
            # Add labels for programmed crystals
            if is_programmed and crystal.current_program:
                freq = crystal.current_program.get('crystal_frequency', 
                       crystal.current_program.get('frequency', 0))
                ax1.text(x, y-0.3, f'{freq:.0f}Hz', 
                        color='cyan', fontsize=8, ha='center')
        
        # Add grid
        ax1.grid(True, alpha=0.2, color='cyan')
        ax1.set_xlabel('Sacred Geometry X', color='cyan')
        ax1.set_ylabel('Sacred Geometry Y', color='cyan')
        
        # Frequency spectrum visualization
        ax2.set_facecolor('#0a0a0a')
        ax2.set_title('Frequency Distribution', color='white', fontsize=16)
        
        # Extract all frequencies
        frequencies = []
        amplitudes = []
        colors = []
        
        for crystal in programmed:
            if crystal.current_program:
                freq = crystal.current_program.get('crystal_frequency',
                       crystal.current_program.get('frequency', 0))
                frequencies.append(freq)
                amplitudes.append(crystal.activation_level)
                colors.append(self.crystal_properties[crystal.crystal_type]['color'])
        
        # Create frequency bars
        if frequencies:
            bars = ax2.bar(range(len(frequencies)), amplitudes, color=colors,
                          edgecolor='gold', linewidth=2)
            
            # Add frequency labels
            ax2.set_xticks(range(len(frequencies)))
            ax2.set_xticklabels([f'{f:.0f}Hz' for f in frequencies],
                               rotation=45, color='cyan')
            ax2.set_ylabel('Activation Level', color='cyan')
            ax2.set_xlabel('Crystal Frequencies', color='cyan')
            
            # Add coherence indicator
            coherence_text = f'Array Coherence: {results["coherence"]*100:.1f}%'
            ax2.text(0.5, 0.95, coherence_text, transform=ax2.transAxes,
                    color='gold', fontsize=14, ha='center',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        ax2.grid(True, alpha=0.2, color='cyan', axis='y')
        
        # Convert to image array
        fig.tight_layout()
        fig.canvas.draw()
        
        # Get the RGBA buffer from the figure
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return buf
    
    def retrieve_true_name(self, retrieval_key: str) -> Optional[Dict]:
        """Retrieve a stored true name from crystal array"""
        
        # Search all crystals for matching key
        for crystal in self.crystal_array:
            if crystal.current_program:
                # Generate key from stored data
                stored_name = crystal.current_program.get('true_name', '')
                stored_freq = crystal.current_program.get('base_frequency', 0)
                
                test_key = self._generate_retrieval_key(stored_name, stored_freq)
                
                if test_key == retrieval_key:
                    # Found it - now reconstruct from all crystals
                    return self._reconstruct_from_crystals(stored_name)
        
        return None
    
    def _reconstruct_from_crystals(self, true_name: str) -> Dict:
        """Reconstruct complete true name data from distributed storage"""
        
        # Collect all crystals with this true name
        name_crystals = [
            c for c in self.crystal_array 
            if c.current_program and c.current_program.get('true_name') == true_name
        ]
        
        if not name_crystals:
            return None
        
        # Reconstruct based on protocol
        protocol = name_crystals[0].current_program.get('programming_protocol', 'basic')
        
        if protocol == 'holographic':
            # Holographic reconstruction
            fragments = sorted(name_crystals, 
                             key=lambda c: c.current_program.get('holographic_fragment', 0))
            
            # Combine phase-shifted data
            reconstructed = {
                'true_name': true_name,
                'frequency': fragments[0].current_program.get('base_frequency'),
                'voice_signature': fragments[0].current_program.get('voice_signature', {}),
                'crystals_used': len(fragments),
                'storage_protocol': 'holographic',
                'coherence': self._calculate_array_coherence(fragments)
            }
            
        elif protocol == 'quantum':
            # Quantum reconstruction
            quantum_data = []
            for crystal in name_crystals:
                quantum_data.append({
                    'frequency': crystal.current_program.get('crystal_frequency'),
                    'quantum_signature': crystal.current_program.get('quantum_signature', {}),
                    'entanglement_id': crystal.current_program.get('entanglement_id')
                })
            
            reconstructed = {
                'true_name': true_name,
                'frequency': name_crystals[0].current_program.get('base_frequency'),
                'quantum_data': quantum_data,
                'storage_protocol': 'quantum'
            }
            
        else:
            # Basic reconstruction
            reconstructed = {
                'true_name': true_name,
                'frequency': name_crystals[0].current_program.get('frequency'),
                'storage_protocol': protocol
            }
        
        # Add retrieval metadata
        reconstructed['retrieved_at'] = datetime.now()
        reconstructed['crystal_coherence'] = self._check_crystal_coherence(name_crystals)
        
        return reconstructed
    
    def _check_crystal_coherence(self, crystals: List[CrystalMatrix]) -> float:
        """Check current coherence of crystal storage"""
        
        if not crystals:
            return 0.0
        
        coherence_sum = 0
        for crystal in crystals:
            if crystal.last_programmed:
                # Calculate decay
                days_elapsed = (datetime.now() - crystal.last_programmed).days
                decay_rate = self.crystal_properties[crystal.crystal_type]['coherence_decay']
                
                # Exponential decay
                current_coherence = crystal.coherence_factor * np.exp(-decay_rate * days_elapsed)
                coherence_sum += current_coherence
        
        return coherence_sum / len(crystals)
    
    def maintenance_report(self) -> Dict:
        """Generate crystal array maintenance report"""
        
        report = {
            'total_crystals': len(self.crystal_array),
            'programmed_crystals': 0,
            'total_names_stored': set(),
            'coherence_levels': {},
            'crystals_needing_recharge': [],
            'array_health': 0.0
        }
        
        health_scores = []
        
        for crystal in self.crystal_array:
            if crystal.current_program:
                report['programmed_crystals'] += 1
                name = crystal.current_program.get('true_name', 'Unknown')
                report['total_names_stored'].add(name)
                
                # Check coherence
                coherence = self._check_crystal_coherence([crystal])
                report['coherence_levels'][crystal.crystal_id] = coherence
                
                # Flag if needs recharge
                if coherence < 0.5:
                    report['crystals_needing_recharge'].append({
                        'crystal_id': crystal.crystal_id,
                        'crystal_type': crystal.crystal_type,
                        'current_coherence': coherence,
                        'true_name': name
                    })
                
                health_scores.append(coherence)
            else:
                health_scores.append(1.0)  # Unprogrammed crystals are "healthy"
        
        report['total_names_stored'] = len(report['total_names_stored'])
        report['array_health'] = np.mean(health_scores) * 100
        
        return report


# Integration with Telegram Bot
class CrystalProgrammingBot:
    """Bot commands for crystal programming features"""
    
    def __init__(self, crystal_interface: CrystalProgrammingInterface):
        self.crystal = crystal_interface
    
    async def handle_crystal_programming(self, true_name_data: Dict, 
                                       user_id: str) -> Dict:
        """Handle crystal programming request from bot"""
        
        # Program the true name
        result = self.crystal.program_true_name(
            true_name_data,
            protocol='quantum'  # Use most advanced protocol
        )
        
        # Generate instructions for physical crystal programming
        instructions = self._generate_physical_instructions(
            true_name_data,
            result
        )
        
        # Store retrieval key for user
        retrieval_key = result['retrieval_key']
        
        return {
            'programming_result': result,
            'instructions': instructions,
            'retrieval_key': retrieval_key,
            'visualization': result['activation_map']
        }
    
    def _generate_physical_instructions(self, name_data: Dict, 
                                      result: Dict) -> str:
        """Generate instructions for programming physical crystals"""
        
        crystals_used = result['crystals_programmed']
        frequency = name_data['frequency']
        name = name_data['true_name']
        
        # Identify crystal types needed
        crystal_types = set()
        for c in self.crystal.crystal_array[:crystals_used]:
            crystal_types.add(c.crystal_type.replace('_', ' ').title())
        
        instructions = f"""
💎 *Physical Crystal Programming Instructions* 💎

*Your True Name:* {name}
*Master Frequency:* {frequency:.2f} Hz
*Crystals Needed:* {', '.join(crystal_types)}

*Setup (Metatron's Cube Formation):*
1. Cleanse all crystals (salt water or sage)
2. Place Clear Quartz in center
3. Arrange other crystals in sacred geometry
4. Create circle with salt around array

*Programming Ritual:*

*Step 1 - Preparation (10 min)*
- Fast for 4 hours before
- Shower with intention
- Wear white or light colors
- Light white candle

*Step 2 - Activation (20 min)*
- Play {frequency:.2f} Hz tone
- State intention 3x: "I program these crystals with my true name {name}"
- Breathe pattern: 4-7-8 (in-hold-out)
- Visualize golden light connecting crystals

*Step 3 - Quantum Encoding (11 min)*
- Hold master crystal (center)
- Chant your true name slowly
- Feel vibration in your bones
- See name written in light

*Step 4 - Sealing (5 min)*
- Touch each crystal clockwise
- Say "It is done" at each
- End at center crystal
- State: "By my true name {name}, this matrix is sealed"

*Maintenance:*
- Recharge monthly at full moon
- Keep array undisturbed
- Add personal items to center
- Record any dreams/visions

*Your Retrieval Key:* `{result['retrieval_key']}`
Save this to access your stored frequency anytime

_These crystals now carry your consciousness signature_
        """
        
        return instructions


# Example usage in bot
def add_crystal_commands_to_bot(bot, crystal_interface):
    """Add crystal programming commands to telegram bot"""
    
    crystal_bot = CrystalProgrammingBot(crystal_interface)
    
    async def crystal_program_callback(update, context):
        """Handle crystal programming request"""
        user_id = str(update.effective_user.id)
        
        # Get stored name data
        if user_id in bot.user_sessions and 'name_data' in bot.user_sessions[user_id]:
            name_data = bot.user_sessions[user_id]['name_data']
            
            # Process crystal programming
            result = await crystal_bot.handle_crystal_programming(name_data, user_id)
            
            # Send visualization
            from io import BytesIO
            buf = BytesIO()
            plt.imsave(buf, result['visualization'], format='png')
            buf.seek(0)
            
            await update.message.reply_photo(
                photo=buf,
                caption="*Crystal Array Programmed Successfully*",
                parse_mode='Markdown'
            )
            
            # Send instructions
            await update.message.reply_text(
                result['instructions'],
                parse_mode='Markdown'
            )
            
            # Store retrieval key
            bot.user_sessions[user_id]['crystal_key'] = result['retrieval_key']
    
    # Add command handler
    bot.app.add_handler(CommandHandler("programcrystal", crystal_program_callback))