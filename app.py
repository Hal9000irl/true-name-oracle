#!/usr/bin/env python3
"""
True Name Oracle - Main Application Entry Point
Integrates all components into unified web service
"""

import os
import sys
sys.path.append('components')
from pathlib import Path

# Add components to path
sys.path.insert(0, str(Path(__file__).parent / 'components'))

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
from datetime import datetime
import asyncio
import json
import logging
from werkzeug.utils import secure_filename
import tempfile
import streamlit as st

# Import our components
from components.unified_consciousness_system import UnifiedConsciousnessSystem
from components.consciousness_frequency_detector import ConsciousnessFrequencyDetector
from components.crystal_programming_interface import CrystalProgrammingInterface
from components.consciousness_safety_protocols import ConsciousnessIntegrityProtection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'consciousness-key-2024')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enable CORS
CORS(app)

# Initialize SocketIO for real-time features
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize consciousness system
consciousness_system = UnifiedConsciousnessSystem()
asyncio.run(consciousness_system.initialize())

# File upload settings
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'webm'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system_state': consciousness_system.state.value
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_voice():
    """Analyze voice and return true name"""
    try:
        # Check if file was uploaded
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save temporary file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            temp_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{filename}")
            file.save(temp_path)
            
            # Process with consciousness system
            user_id = request.form.get('user_id', f'user_{timestamp}')
            
            # Run async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                consciousness_system.process_voice(temp_path, user_id)
            )
            
            # Clean up temp file
            os.remove(temp_path)
            
            # Return results
            return jsonify({
                'success': True,
                'true_name': result['name_data']['true_name'],
                'frequency': result['name_data']['frequency'],
                'pronunciation': result['name_data'].get('pronunciation_guide', ''),
                'crystal_key': result['crystal_programming']['retrieval_key'],
                'safety_status': result['safety_status']['safety_level'].value,
                'timestamp': datetime.now().isoformat()
            })
            
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrieve/<retrieval_key>')
def retrieve_name(retrieval_key):
    """Retrieve stored true name by key"""
    try:
        # Retrieve from crystal storage
        stored_data = consciousness_system.crystal.retrieve_true_name(retrieval_key)
        
        if stored_data:
            return jsonify({
                'success': True,
                'data': stored_data,
                'retrieved_at': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Name not found'}), 404
            
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/crystal/status')
def crystal_status():
    """Get crystal array status"""
    try:
        report = consciousness_system.crystal.maintenance_report()
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket events for real-time monitoring
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'data': 'Connected to consciousness system'})

@socketio.on('start_monitoring')
def handle_monitoring(data):
    """Start real-time consciousness monitoring"""
    session_id = data.get('session_id')
    
    # Start monitoring in background
    socketio.start_background_task(
        monitor_consciousness,
        session_id
    )
    
    emit('monitoring_started', {'session_id': session_id})

def monitor_consciousness(session_id):
    """Background task for consciousness monitoring"""
    while True:
        try:
            # Get current safety metrics
            safety_data = consciousness_system.safety.monitor.get_dashboard_data()
            
            # Emit update to client
            socketio.emit('safety_update', {
                'session_id': session_id,
                'metrics': safety_data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Check for alerts
            if safety_data.get('total_alerts', 0) > 0:
                socketio.emit('safety_alert', {
                    'session_id': session_id,
                    'alerts': safety_data.get('alerts', [])
                })
            
            socketio.sleep(1)  # Update every second
            
        except Exception as e:
            logger.error(f"Monitoring error: {str(e)}")
            break

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Template routes (if using server-side rendering)
@app.route('/crystal')
def crystal_interface():
    """Crystal programming interface"""
    return render_template('crystal.html')

@app.route('/safety')
def safety_dashboard():
    """Safety monitoring dashboard"""
    return render_template('safety.html')

if __name__ == '__main__':
    # Development server
    socketio.run(
        app,
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )

st.title("Simple Streamlit Form")

with st.form(key="my_form"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    submit_button = st.form_submit_button("Submit")

    if submit_button:
        st.success(f"Hello {name}! Your email ({email}) was submitted!")