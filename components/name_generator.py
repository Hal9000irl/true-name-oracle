import os
import requests
import json
from typing import Dict
from dotenv import load_dotenv
import sys
sys.path.append('..')  # To import from parent directory
from prompts.alan_watts_personality import AlanWattsPrompts
import time

load_dotenv()

class NameGenerator:
    def __init__(self):
        # X.AI Configuration
        self.api_url = "https://api.x.ai/v1/chat/completions"
        self.api_key = os.getenv("XAI_API_KEY")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # Initialize Alan Watts prompts
        self.watts = AlanWattsPrompts()
    
    def generate_names(self, frequency_data: Dict) -> Dict:
        """Generate true names using Grok-3 with Alan Watts personality"""
        
        try:
            # First attempt with Alan Watts personality
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "messages": [
                        {
                            "role": "system",
                            "content": self.watts.get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": self.watts.get_name_generation_prompt(frequency_data)
                        }
                    ],
                    "model": "grok-3-latest",
                    "stream": False,
                    "temperature": 0.8  # Higher for Watts' playful creativity
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse JSON from Grok's response
                try:
                    # Clean up response if needed (Grok might add extra text)
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end != 0:
                        json_content = content[json_start:json_end]
                        names_data = json.loads(json_content)
                    else:
                        names_data = json.loads(content)
                    
                    time.sleep(0.5)
                    return {
                        'shadow': names_data['shadow_name'],
                        'shadow_meaning': names_data['shadow_meaning'],
                        'bridge': names_data['bridge_name'],
                        'bridge_meaning': names_data['bridge_meaning'],
                        'star': names_data['star_name'],
                        'star_meaning': names_data['star_meaning']
                    }
                    
                except json.JSONDecodeError:
                    # Try refinement prompt if first attempt failed
                    return self._refine_with_watts(frequency_data, content)
            else:
                print(f"API Error: {response.status_code}")
                return self._generate_watts_fallback(frequency_data)
                
        except Exception as e:
            print(f"Connection error: {str(e)}")
            return self._generate_watts_fallback(frequency_data)
    
    def _refine_with_watts(self, frequency_data: Dict, previous_attempt: str) -> Dict:
        """Try again with refinement prompt"""
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "messages": [
                        {
                            "role": "system",
                            "content": self.watts.get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": self.watts.get_refinement_prompt(frequency_data, previous_attempt)
                        }
                    ],
                    "model": "grok-3-latest",
                    "stream": False,
                    "temperature": 0.9  # Even more creative
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                # Try parsing again
                # [Same parsing logic as above]
                
        except:
            return self._generate_watts_fallback(frequency_data)
    
    def _generate_watts_fallback(self, frequency_data: Dict) -> Dict:
        """Generate names even if API fails, in Alan Watts style"""
        # Use frequency to create poetic names
        base_freq = int(frequency_data['fundamental_freq'])
        soul_ratio = frequency_data['soul_ratio']
        
        # Watts-inspired name components
        if soul_ratio > 1.5:  # Golden ratio vicinity = evolved soul
            shadow_components = ['Void', 'Echo', 'Mist', 'Shadow']
            bridge_components = ['Dance', 'Flow', 'Dream', 'Wave']
            star_components = ['Light', 'Song', 'Joy', 'One']
        else:
            shadow_components = ['Stone', 'Thorn', 'Ash', 'Dust']
            bridge_components = ['River', 'Path', 'Bridge', 'Door']
            star_components = ['Star', 'Sky', 'Sun', 'Home']
        
        # Create names using frequency as seed
        idx = base_freq % len(shadow_components)
        
        return {
            'shadow': f"{shadow_components[idx]}walker",
            'shadow_meaning': "The one who befriends their own darkness",
            'bridge': f"{bridge_components[idx]}dancer",
            'bridge_meaning': "The eternal student of transformation",
            'star': f"{star_components[idx]}singer",
            'star_meaning': "The voice that calls all beings home"
        }