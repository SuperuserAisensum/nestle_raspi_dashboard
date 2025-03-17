from flask import (
    Flask, render_template, request, jsonify, send_from_directory,
    Response, abort, redirect, url_for, session, flash
)
import os
import json
import sqlite3
from datetime import datetime, timedelta
import logging
import traceback
import shutil
import tempfile
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
from functools import wraps
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from roboflow import Roboflow
from dds_cloudapi_sdk import Config, Client, TextPrompt
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget
import cv2
import requests
import numpy as np
import math
from sklearn.cluster import KMeans
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# -----------------------------
# Constants
# -----------------------------
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB
DEFAULT_PAGE_SIZE = 20
DATABASE_TIMEOUT = 30.0  # SQLite timeout in seconds

# Initialize geocoder
geolocator = Nominatim(user_agent="nestle_pi_app")

def get_address_from_coordinates(latitude: float, longitude: float) -> str:
    """Convert coordinates to address using geopy"""
    try:
        location = geolocator.reverse((latitude, longitude), language='en')
        if location:
            return location.address
        return "Address not found"
    except GeocoderTimedOut:
        return "Geocoding service timeout"
    except Exception as e:
        logger.error(f"Error getting address: {str(e)}")
        return "Error getting address"

# -----------------------------
# Model Configuration
# -----------------------------
# Roboflow setup
rf_api_key = os.getenv("ROBOFLOW_API_KEY", "Otg64Ra6wNOgDyjuhMYU")
workspace = os.getenv("ROBOFLOW_WORKSPACE", "alat-pelindung-diri")
project_name = os.getenv("ROBOFLOW_PROJECT", "nescafe-4base")
model_version = int(os.getenv("ROBOFLOW_MODEL_VERSION", "89"))

rf = Roboflow(api_key=rf_api_key)
project = rf.workspace(workspace).project(project_name)
yolo_model = project.version(model_version).model

# OWLv2 setup
OWLV2_API_KEY = "bjJkZXZrb2Y1cDMzMXh3OHdzbGl6OlFQOHVmS2JkZjBmQUs2bnF2OVJVdXFoNnc0ZW5kN1hH"
OWLV2_PROMPTS = ["bottle", "tetra pak", "cans", "carton drink"]

# -----------------------------
# Logging Configuration
# -----------------------------
def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("server.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

# -----------------------------
# Flask Application Setup
# -----------------------------
def create_app() -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv('FLASK_SECRET_KEY', 'nestle-iot-monitoring-secret-key'),
        UPLOAD_FOLDER=os.path.join(os.getcwd(), 'uploads'),
        DATABASE=os.path.join(os.getcwd(), 'nestle_iot.db'),
        MAX_CONTENT_LENGTH=MAX_FILE_SIZE
    )
    
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Ensure frames folder exists
    os.makedirs(os.path.join(os.getcwd(), 'frames'), exist_ok=True)
    
    return app

app = create_app()
socketio = SocketIO(app, cors_allowed_origins="*")

# -----------------------------
# Authentication Helpers
# -----------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# -----------------------------
# Authentication Routes
# -----------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # Replace these with your secure credentials or integrate with a user database
        if username != 'nestle' or password != 'jHb5WB.)7M+^sq#wx.VoULhq9*y':
            error = 'Invalid credentials. Please try again.'
        else:
            session['logged_in'] = True
            return redirect(url_for('index'))
    return render_template('login.html', error=error)

@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# -----------------------------
# Database Setup
# -----------------------------
def get_db_connection() -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(
            app.config['DATABASE'],
            timeout=DATABASE_TIMEOUT,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise

def init_db() -> None:
    try:
        with get_db_connection() as conn:
            # Create table if not exists with all columns
            conn.execute('''
                CREATE TABLE IF NOT EXISTS detection_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    roboflow_outputs TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    iqi_score REAL,
                    location TEXT
                )
            ''')
            
            # Add columns if they don't exist
            try:
                # First try to add location column without NOT NULL constraint
                conn.execute('ALTER TABLE detection_events ADD COLUMN location TEXT')
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                # Then update existing NULL values to 'Unknown'
                conn.execute("UPDATE detection_events SET location = 'Unknown' WHERE location IS NULL")
                # Finally, add NOT NULL constraint
                conn.execute("ALTER TABLE detection_events RENAME TO detection_events_old")
                conn.execute('''
                    CREATE TABLE detection_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        device_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        roboflow_outputs TEXT NOT NULL,
                        image_path TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        iqi_score REAL,
                        location TEXT NOT NULL DEFAULT 'Unknown'
                    )
                ''')
                conn.execute("INSERT INTO detection_events SELECT * FROM detection_events_old")
                conn.execute("DROP TABLE detection_events_old")
            except sqlite3.OperationalError:
                pass  # Constraint might already exist
                
            conn.commit()
            
        logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise

# -----------------------------
# Helper Functions
# -----------------------------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file, device_id: str) -> str:
    if not file or not allowed_file(file.filename):
        raise ValueError("Invalid file type")
    
    filename = secure_filename(file.filename)
    device_folder = os.path.join(app.config['UPLOAD_FOLDER'], device_id)
    os.makedirs(device_folder, exist_ok=True)
    
    file_path = os.path.join(device_folder, filename)
    file.save(file_path)
    return file_path

def parse_detection_data(roboflow_data: Dict) -> Tuple[int, int]:
    nestle_count = 0
    unclassified_count = 0
    
    try:
        if isinstance(roboflow_data, dict) and 'predictions' in roboflow_data:
            for pred in roboflow_data['predictions']:
                if pred.get('class', '').lower().startswith('nestle'):
                    nestle_count += 1
                else:
                    unclassified_count += 1
        elif isinstance(roboflow_data, list):
            for item in roboflow_data:
                if 'roboflow_predictions' in item:
                    for pred in item['roboflow_predictions']:
                        if pred.get('class', '').lower().startswith('nestle'):
                            nestle_count += 1
                        else:
                            unclassified_count += 1
                if 'dinox_predictions' in item:
                    unclassified_count += len(item['dinox_predictions'])
    except Exception as e:
        logger.error(f"Error parsing detection data: {e}")
        raise
        
    return nestle_count, unclassified_count

def get_default_sku_data() -> Dict:
    current_date = datetime.now()
    dates = [(current_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
    
    return {
        "daily_data": {
            "dates": dates,
            "values": [0] * 7,
            "nestle_values": [0] * 7,
            "competitor_values": [0] * 7
        },
        "nestle": {
            "max": {"count": 0, "date": dates[-1]},
            "min": {"count": 0, "date": dates[0]},
            "avg": {"count": 0, "period": "Last 7 days"}
        },
        "competitor": {
            "max": {"count": 0, "date": dates[-1]},
            "min": {"count": 0, "date": dates[0]},
            "avg": {"count": 0, "period": "Last 7 days"}
        },
        "market_share": {
            "labels": ["Nestlé", "Competitor"],
            "values": [50, 50]
        },
        "top_products": [
            {"name": "Product A", "count": 0},
            {"name": "Product B", "count": 0},
            {"name": "Product C", "count": 0}
        ],
        "daily_count": {
            "product": "All Products",
            "dates": dates,
            "counts": [0] * 7
        }
    }

def copy_file_to_frames(source_path, target_filename):
    frames_dir = os.path.join(os.getcwd(), 'frames')
    target_path = os.path.join(frames_dir, target_filename)
    
    try:
        shutil.copy2(source_path, target_path)
        logger.info(f"Copied {source_path} to {target_path}")
        return target_path
    except Exception as e:
        logger.error(f"Error copying file {source_path} to {target_path}: {e}")
        return None

def is_overlap(box1, boxes2, threshold=0.3):
    x1_min, y1_min, x1_max, y1_max = box1
    for b2 in boxes2:
        x2, y2, w2, h2 = b2
        x2_min = x2 - w2/2
        x2_max = x2 + w2/2
        y2_min = y2 - h2/2
        y2_max = y2 + h2/2

        dx = min(x1_max, x2_max) - max(x1_min, x2_min)
        dy = min(y1_max, y2_max) - max(y1_min, y2_min)
        if (dx >= 0) and (dy >= 0):
            area_overlap = dx * dy
            area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
            if area_box1 > 0 and (area_overlap / area_box1) > threshold:
                return True
    return False

# IQI Calculation Functions
def compute_darkness_index(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_intensity = np.mean(gray_image)
    darkness_index = (avg_intensity / 255) * 100
    if darkness_index <= 35:
        score = (darkness_index / 35) * 100
    elif 35 < darkness_index <= 55:
        score = ((darkness_index - 35) / 20) * 100
        score = 50 + (score / 2)
    elif 55 < darkness_index <= 75:
        score = ((75 - darkness_index) / 20) * 100
        score = 50 + (score / 2)
    else:
        score = ((100 - darkness_index) / 25) * 100
    return round(max(0, min(100, score)), 2)

def compute_intersection(line1, line2, image_shape):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    h, w = image_shape

    x1, y1 = x1 / w, y1 / h
    x2, y2 = x2 / w, y2 / h
    x3, y3 = x3 / w, y3 / h
    x4, y4 = x4 / w, y4 / h

    A1, B1, C1 = y2 - y1, x1 - x2, (y2 - y1) * x1 + (x1 - x2) * y1
    A2, B2, C2 = y4 - y3, x3 - x4, (y4 - y3) * x3 + (x3 - x4) * y3
    determinant = A1 * B2 - A2 * B1

    if abs(determinant) < 1e-10:
        return None

    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant

    x, y = x * w, y * h
    return (x, y)

def find_vanishing_point(image, image_path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 150, apertureSize=3)  # Menurunkan threshold bawah

    # Menurunkan threshold dan panjang minimum garis
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, 
                           minLineLength=100, maxLineGap=15)
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    vp = (cx, cy)
    filtered_lines = []
    intersections = []

    if lines is not None and len(lines) >= 2:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            # Memperlebar range sudut yang diterima
            if (10 < angle < 80 or 100 < angle < 170) and length > 100:
                filtered_lines.append(line[0])

        if len(filtered_lines) >= 2:
            # Memperbesar margin dan threshold
            margin = w * 0.75
            center_threshold = w * 0.4
            for i in range(len(filtered_lines)):
                for j in range(i + 1, len(filtered_lines)):
                    pt = compute_intersection(filtered_lines[i], filtered_lines[j], (h, w))
                    if pt and -margin <= pt[0] <= w + margin and -margin <= pt[1] <= h + margin:
                        distance_to_center = math.sqrt((pt[0] - cx)**2 + (pt[1] - cy)**2)
                        if distance_to_center < center_threshold:
                            intersections.append(pt)

            if intersections:
                intersections = np.array(intersections)
                if len(intersections) > 3:
                    try:
                        kmeans = KMeans(n_clusters=min(3, len(intersections)), n_init=10).fit(intersections)
                        centers = kmeans.cluster_centers_
                        distances = [math.sqrt((x - cx)**2 + (y - cy)**2) for x, y in centers]
                        best_cluster_idx = np.argmin(distances)
                        vp = tuple(centers[best_cluster_idx])
                    except Exception as e:
                        weights = [1 / (math.sqrt((x - cx)**2 + (y - cy)**2) + 1) for x, y in intersections]
                        vp = tuple(np.average(intersections, axis=0, weights=weights))
                else:
                    vp = tuple(np.median(intersections, axis=0))

    return vp

def compute_angle_from_vanishing_point(vp, image, focal_length=1200):
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    dx = vp[0] - cx
    dy = vp[1] - cy
    distance = math.sqrt(dx**2 + dy**2)
    angle_rad = math.atan2(distance, focal_length)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def compute_angle_score(angle):
    if angle is None:
        return 75  # Nilai default lebih tinggi
    # Menggunakan fungsi exponential yang lebih toleran
    score = 100 * math.exp(-angle / 25)  # Mengubah dari 15 ke 25 untuk lebih toleran
    return round(max(0, min(100, score)), 2)

def compute_IQI(image_path, focal_length=1200):
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Error: Image {image_path} not found or cannot be loaded.")
        return None

    darkness_index = compute_darkness_index(image)
    vp = find_vanishing_point(image, image_path)
    
    h, w = image.shape[:2]
    default_vp = (w / 2.0, h / 2.0)
    
    if vp == default_vp:
        angle = None
        angle_index = compute_angle_score(None)
    else:
        angle = compute_angle_from_vanishing_point(vp, image, focal_length)
        angle_index = compute_angle_score(angle)

    # Mengubah bobot: 70% darkness, 30% angle
    if angle is None:
        IQI = round((darkness_index * 0.7) + (angle_index * 0.3), 2)
    else:
        IQI = round((darkness_index * 0.7) + (angle_index * 0.3), 2)

    return IQI

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
@login_required
def index():
    try:
        sku_data = get_default_sku_data()
        return render_template('index.html', sku_data=sku_data)
    except Exception as e:
        logger.error(f"Error rendering index page: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/receive_data', methods=['POST'])
def receive_data():
    try:
        if 'image0' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image0']
        if not image_file.filename:
            return jsonify({'error': 'No selected image file'}), 400

        filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

        # Get IQI score from request if available, otherwise calculate it
        iqi_score = None
        if request.form.get('iqi_score'):
            try:
                iqi_score = float(request.form.get('iqi_score'))
                logger.info(f"Using IQI score from request: {iqi_score}")
            except (ValueError, TypeError):
                logger.warning("Invalid IQI score from request, will calculate.")
                
        # Calculate IQI score only if not provided in request
        if iqi_score is None:
            iqi_score = compute_IQI(image_path)
            logger.info(f"Calculated IQI score: {iqi_score}")

        device_id = request.form.get('device_id')
        timestamp = request.form.get('timestamp')
        roboflow_data = json.loads(request.form.get('roboflow_outputs'))
        
        # Get location data and convert to address
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        address = "Unknown"
        
        if latitude and longitude and latitude != 'N/A' and longitude != 'N/A':
            try:
                latitude = float(latitude)
                longitude = float(longitude)
                address = get_address_from_coordinates(latitude, longitude)
            except ValueError:
                logger.error(f"Invalid coordinates: lat={latitude}, long={longitude}")
        
        # Use the detection data from the client instead of performing new detection
        total_nestle = roboflow_data.get('total_nestle', 0)
        total_unclassified = roboflow_data.get('total_unclassified', 0)
        
        # Store the data in the database
        with get_db_connection() as conn:
            current_time = datetime.now().isoformat()
            current_date = current_time.split('T')[0]
            
            detection_data = {
                'roboflow_predictions': roboflow_data.get('roboflow_predictions', {}),
                'dinox_predictions': {'unclassified': total_unclassified},
                'counts': {'nestle': total_nestle, 'competitor': total_unclassified, 'date': current_date},
                'location': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'address': address
                }
            }
            
            cursor = conn.execute(
                '''INSERT INTO detection_events 
                   (device_id, timestamp, roboflow_outputs, image_path, created_at, iqi_score, location) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (device_id, timestamp, json.dumps(detection_data), 
                 image_path, current_time, iqi_score, address)
            )
            event_id = cursor.lastrowid
            conn.commit()

        socketio.emit('new_detection', {
            'id': event_id,
            'device_id': device_id,
            'timestamp': timestamp,
            'date': current_date,
            'nestle_count': total_nestle,
            'competitor_count': total_unclassified,
            'iqi_score': iqi_score,
            'location': address,
            'type': 'new_detection'
        })

        return jsonify({
            'success': True,
            'message': 'Data received and processed successfully',
            'event_id': event_id
        })

    except Exception as e:
        logger.error(f"Error in receive_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/images/<path:filename>')
def get_image(filename):
    try:
        if '..' in filename or filename.startswith('/'):
            abort(404)
        if filename.startswith('frames/'):
            frames_dir = os.path.join(os.getcwd(), 'frames')
            file_path = filename.replace('frames/', '')
            return send_from_directory(frames_dir, file_path)
        elif os.path.exists(os.path.join(os.getcwd(), 'frames', filename)):
            frames_dir = os.path.join(os.getcwd(), 'frames')
            return send_from_directory(frames_dir, filename)
        else:
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        abort(404)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/events', methods=['GET'])
def get_events():
    try:
        device_id = request.args.get('device_id')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        page = max(1, request.args.get('page', 1, type=int))
        limit = min(50, request.args.get('limit', DEFAULT_PAGE_SIZE, type=int))
        sort_order = request.args.get('sort', 'desc').upper()

        query = 'SELECT * FROM detection_events WHERE 1=1'
        params = []

        if device_id:
            query += ' AND device_id = ?'
            params.append(device_id)
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)

        # Allow changing the sort order with the sort parameter
        if sort_order == 'ASC':
            query += ' ORDER BY id ASC LIMIT ? OFFSET ?'
        else:
            query += ' ORDER BY id DESC LIMIT ? OFFSET ?'
        
        params.extend([limit, (page - 1) * limit])

        with get_db_connection() as conn:
            events = conn.execute(query, params).fetchall()
            count_query = query.split('ORDER BY')[0].replace('SELECT *', 'SELECT COUNT(*)')
            total = conn.execute(count_query, params[:-2]).fetchone()[0]

        results = []
        for event in events:
            try:
                roboflow_outputs = event['roboflow_outputs']
                if not roboflow_outputs:
                    continue

                try:
                    roboflow_data = json.loads(roboflow_outputs)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in event {event['id']}: {roboflow_outputs}")
                    continue

                nestle_count = 0
                competitor_count = 0

                if isinstance(roboflow_data, dict):
                    if 'roboflow_predictions' in roboflow_data:
                        nestle_count = len(roboflow_data['roboflow_predictions'])
                    if 'dinox_predictions' in roboflow_data:
                        competitor_count = len(roboflow_data['dinox_predictions'])
                elif isinstance(roboflow_data, dict) and 'predictions' in roboflow_data:
                    nestle_count = len(roboflow_data['predictions'])
                
                results.append({
                    'id': event['id'],
                    'device_id': event['device_id'],
                    'timestamp': event['timestamp'],
                    'nestle_detections': nestle_count,
                    'unclassified_detections': competitor_count,
                    'created_at': event['created_at'],
                    'image_path': event['image_path'],
                    'iqi_score': event['iqi_score'],
                    'location': event['location']
                })
            except Exception as e:
                logger.error(f"Error processing event {event['id']}: {str(e)}")
                continue

        return jsonify({
            'data': results,
            'pagination': {
                'total': total,
                'page': page,
                'limit': limit,
                'pages': (total + limit - 1) // limit
            }
        })

    except Exception as e:
        logger.error(f"Error in get_events: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/events/<int:event_id>')
def get_event_details(event_id):
    try:
        with get_db_connection() as conn:
            event = conn.execute('SELECT * FROM detection_events WHERE id = ?', (event_id,)).fetchone()
            
            if not event:
                return jsonify({'error': 'Event not found'}), 404

            roboflow_data = json.loads(event['roboflow_outputs']) if event['roboflow_outputs'] else {}
            
            nestle_products = {}
            competitor_products = {}
            
            if isinstance(roboflow_data, dict):
                if 'roboflow_predictions' in roboflow_data:
                    nestle_products = roboflow_data['roboflow_predictions']
                if 'dinox_predictions' in roboflow_data:
                    competitor_products = roboflow_data['dinox_predictions']

            nestle_count = sum(nestle_products.values() if isinstance(nestle_products, dict) else [1 for _ in nestle_products])
            comp_count = sum(competitor_products.values() if isinstance(competitor_products, dict) else [1 for _ in competitor_products])

            image_path = event['image_path']
            if image_path:
                image_path = os.path.basename(image_path)
                if not image_path.startswith('uploads/'):
                    image_path = f'uploads/{image_path}'

            # Get location data from roboflow_outputs if available
            location_data = roboflow_data.get('location', {})
            location_info = {
                'address': event['location'],
                'coordinates': {
                    'latitude': location_data.get('latitude', 'N/A'),
                    'longitude': location_data.get('longitude', 'N/A')
                }
            }

            response = {
                'id': event['id'],
                'device_id': event['device_id'],
                'timestamp': event['timestamp'],
                'nestleCount': nestle_count,
                'compCount': comp_count,
                'products': {
                    'nestle_products': nestle_products,
                    'competitor_products': competitor_products
                },
                'image_path': image_path,
                'created_at': event['created_at'],
                'iqi_score': event['iqi_score'],
                'location': location_info
            }
            
            return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error getting event details: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/dashboard_data')
def get_dashboard_data():
    try:
        current_date = datetime.now()
        dates = [(current_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
        
        dashboard_data = {
            "daily_data": {
                "dates": dates,
                "nestle_values": [0] * 7,
                "competitor_values": [0] * 7
            },
            "nestle": {
                "max": {"count": 0, "date": dates[-1]},
                "min": {"count": float('inf'), "date": dates[0]},
                "avg": {"count": 0, "period": "Last 7 days"}
            },
            "competitor": {
                "max": {"count": 0, "date": dates[-1]},
                "min": {"count": float('inf'), "date": dates[0]},
                "avg": {"count": 0, "period": "Last 7 days"}
            },
            "top_products": [
                {'name': 'No Product', 'count': 0},
                {'name': 'No Product', 'count': 0},
                {'name': 'No Product', 'count': 0}
            ]
        }

        product_counts = {}
        
        with get_db_connection() as conn:
            events = conn.execute(
                """SELECT * FROM detection_events 
                   WHERE timestamp >= ? 
                   ORDER BY timestamp""",
                (dates[0],)
            ).fetchall()
            
            for event in events:
                try:
                    if not event['roboflow_outputs']:
                        continue

                    roboflow_data = json.loads(event['roboflow_outputs'])
                    
                    if isinstance(roboflow_data, dict) and 'roboflow_predictions' in roboflow_data:
                        nestle_products = roboflow_data['roboflow_predictions']
                        
                        if isinstance(nestle_products, dict):
                            for product, count in nestle_products.items():
                                if product not in product_counts:
                                    product_counts[product] = 0
                                product_counts[product] += count

                except Exception as e:
                    logger.error(f"Error processing event for top products: {str(e)}")
                continue
                    
        top_products = []
        if product_counts:
            sorted_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
            top_3 = sorted_products[:3]
            
            while len(top_3) < 3:
                top_3.append(('No Product', 0))
            
            top_products = [
                {'name': product, 'count': count}
                for product, count in top_3
            ]
        else:
            top_products = [
                {'name': 'No Product', 'count': 0},
                {'name': 'No Product', 'count': 0},
                {'name': 'No Product', 'count': 0}
            ]

        dashboard_data["top_products"] = top_products

        with get_db_connection() as conn:
            events = conn.execute(
                """SELECT * FROM detection_events 
                   WHERE timestamp >= ? 
                   ORDER BY timestamp""",
                (dates[0],)
            ).fetchall()

            nestle_total = 0
            competitor_total = 0
            nestle_count = 0
            competitor_count = 0
            
            nestle_counts = []
            competitor_counts = []

            for event in events:
                try:
                    if not event['roboflow_outputs']:
                        continue
            
                    roboflow_data = json.loads(event['roboflow_outputs'])
                    event_date = datetime.fromisoformat(event['timestamp'].split('T')[0]).strftime("%Y-%m-%d")
                    
                    current_nestle_count = 0
                    current_competitor_count = 0

                    if isinstance(roboflow_data, dict):
                        if 'roboflow_predictions' in roboflow_data:
                            nestle_products = roboflow_data['roboflow_predictions']
                            current_nestle_count = sum(nestle_products.values() if isinstance(nestle_products, dict) 
                                                     else [1 for _ in nestle_products])
                            
                            if current_nestle_count > 0:
                                nestle_counts.append({'count': current_nestle_count, 'date': event_date})
                            
                            if event_date in dates:
                                date_index = dates.index(event_date)
                                dashboard_data['daily_data']['nestle_values'][date_index] += current_nestle_count

                        if 'dinox_predictions' in roboflow_data:
                            competitor_products = roboflow_data['dinox_predictions']
                            current_competitor_count = sum(competitor_products.values() if isinstance(competitor_products, dict) 
                                                        else [1 for _ in competitor_products])
                            
                            if current_competitor_count > 0:
                                competitor_counts.append({'count': current_competitor_count, 'date': event_date})
                            
                            if event_date in dates:
                                date_index = dates.index(event_date)
                                dashboard_data['daily_data']['competitor_values'][date_index] += current_competitor_count

                    if current_nestle_count > dashboard_data['nestle']['max']['count']:
                        dashboard_data['nestle']['max']['count'] = current_nestle_count
                        dashboard_data['nestle']['max']['date'] = event_date

                    if current_competitor_count > dashboard_data['competitor']['max']['count']:
                        dashboard_data['competitor']['max']['count'] = current_competitor_count
                        dashboard_data['competitor']['max']['date'] = event_date

                    if current_nestle_count > 0:
                        nestle_total += current_nestle_count
                        nestle_count += 1
                    if current_competitor_count > 0:
                        competitor_total += current_competitor_count
                        competitor_count += 1

                except Exception as e:
                    logger.error(f"Error processing event {event['id']} for dashboard: {str(e)}")
                    continue

            if nestle_counts:
                min_nestle = min(nestle_counts, key=lambda x: x['count'])
                dashboard_data['nestle']['min']['count'] = min_nestle['count']
                dashboard_data['nestle']['min']['date'] = min_nestle['date']
            else:
                dashboard_data['nestle']['min']['count'] = 0
                
            if competitor_counts:
                min_competitor = min(competitor_counts, key=lambda x: x['count'])
                dashboard_data['competitor']['min']['count'] = min_competitor['count']
                dashboard_data['competitor']['min']['date'] = min_competitor['date']
            else:
                dashboard_data['competitor']['min']['count'] = 0

            dashboard_data['nestle']['avg']['count'] = round(nestle_total / nestle_count if nestle_count > 0 else 0, 1)
            dashboard_data['competitor']['avg']['count'] = round(competitor_total / competitor_count if competitor_count > 0 else 0, 1)
        
        return jsonify(dashboard_data)
    
    except Exception as e:
        logger.error(f"Error generating dashboard data: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/export/csv')
def export_csv():
    try:
        with get_db_connection() as conn:
            events = conn.execute('SELECT * FROM detection_events ORDER BY timestamp DESC').fetchall()
        
        if not events:
            return Response("No data available for export", mimetype='text/plain')
        
        csv_data = "id,device_id,timestamp,nestle_detections,unclassified_detections,created_at\n"
        
        for event in events:
            try:
                roboflow_data = json.loads(event['roboflow_outputs'])
                nestle_count, unclassified_count = parse_detection_data(roboflow_data)
                
                csv_data += f"{event['id']},{event['device_id']},{event['timestamp']},{nestle_count},{unclassified_count},{event['created_at']}\n"
            except Exception as e:
                logger.error(f"Error processing event {event['id']} for CSV: {e}")
                continue
        
        response = Response(
            csv_data,
            mimetype='text/csv',
            headers={"Content-Disposition": "attachment;filename=nestle_detection_data.csv"}
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error exporting CSV: {str(e)}")
        return jsonify({'error': 'Error generating CSV export'}), 500

@app.route('/api/devices')
def get_devices():
    try:
        with get_db_connection() as conn:
            devices = conn.execute("SELECT DISTINCT device_id FROM detection_events").fetchall()
            
        device_list = [{"id": row["device_id"], "name": row["device_id"]} for row in devices]
        
        return jsonify(device_list)
        
    except Exception as e:
        logger.error(f"Error getting devices: {str(e)}")
        return jsonify({'error': 'Error retrieving device list'}), 500

@app.route('/check_image', methods=['POST'])
def check_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name

            # Calculate IQI score
            iqi_score = compute_IQI(temp_path)

            try:
                yolo_pred = yolo_model.predict(temp_path, confidence=50, overlap=80).json()
                
                nestle_products = {}
                nestle_boxes = []
                for pred in yolo_pred['predictions']:
                    class_name = pred['class']
                    nestle_products[class_name] = nestle_products.get(class_name, 0) + 1
                    nestle_boxes.append({
                        'x': pred['x'],
                        'y': pred['y'],
                        'width': pred['width'],
                        'height': pred['height'],
                        'class': class_name,
                        'confidence': pred['confidence']
                    })
                total_nestle = sum(nestle_products.values())

                headers = {"Authorization": "Basic " + OWLV2_API_KEY}
                data = {"prompts": OWLV2_PROMPTS, "model": "owlv2"}
                with open(temp_path, "rb") as f:
                    files = {"image": f}
                    response = requests.post(
                        "https://api.landing.ai/v1/tools/text-to-object-detection",
                        files=files,
                        data=data,
                        headers=headers
                    )
                owlv2_result = response.json()

                competitor_products = {}
                competitor_boxes = []
                total_competitor = 0

                if 'data' in owlv2_result and owlv2_result['data']:
                    for obj in owlv2_result['data'][0]:
                        if 'bounding_box' in obj:
                            bbox = obj['bounding_box']
                            if not is_overlap(bbox, [(box['x'], box['y'], box['width'], box['height']) for box in nestle_boxes]):
                                category = obj.get('class', 'unclassified')
                                competitor_products[category] = competitor_products.get(category, 0) + 1
                                competitor_boxes.append({
                                    'box': bbox,
                                    'class': category,
                                    'confidence': obj.get('score', 0)
                                })
                                total_competitor += 1

                img = cv2.imread(temp_path)
                
                for box in nestle_boxes:
                    x, y, w, h = box['x'], box['y'], box['width'], box['height']
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, f"Nestle: {box['class']}", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.55, (255, 0, 0), 2)

                for box in competitor_boxes:
                    x1, y1, x2, y2 = box['box']
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 0, 255), 2)
                    cv2.putText(img, f"Competitor: {box['class']}", 
                               (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.55, (0, 0, 255), 2)

                labeled_filename = 'labeled_' + secure_filename(file.filename)
                labeled_path = os.path.join(app.config['UPLOAD_FOLDER'], labeled_filename)
                cv2.imwrite(labeled_path, img)

                with get_db_connection() as conn:
                    current_time = datetime.now().isoformat()
                    current_date = current_time.split('T')[0]
                    
                    detection_data = {
                        'roboflow_predictions': nestle_products,
                        'dinox_predictions': {'unclassified': total_competitor},
                        'counts': {'nestle': total_nestle, 'competitor': total_competitor, 'date': current_date}
                    }
                    
                    cursor = conn.execute(
                        '''INSERT INTO detection_events 
                           (device_id, timestamp, roboflow_outputs, image_path, created_at, iqi_score) 
                           VALUES (?, ?, ?, ?, ?, ?)''',
                        ('web_upload', current_time, json.dumps(detection_data), 
                         labeled_path, current_time, iqi_score)
                    )
                    event_id = cursor.lastrowid
                    conn.commit()

                socketio.emit('new_detection', {
                    'id': event_id,
                    'device_id': 'web_upload',
                    'timestamp': current_time,
                    'date': current_date,
                    'nestle_count': total_nestle,
                    'competitor_count': total_competitor,
                    'iqi_score': iqi_score,
                    'type': 'new_detection'
                })

                result = {
                    'nestle_products': nestle_products,
                    'competitor_products': {'unclassified': total_competitor},
                    'total_nestle': total_nestle,
                    'total_competitor': total_competitor,
                    'labeled_image': f'uploads/{labeled_filename}',
                    'timestamp': current_time,
                    'date': current_date,
                    'iqi_score': iqi_score
                }

                return jsonify(result)

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        logger.error(f"Error processing uploaded image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/all_products')
def get_all_products():
    try:
        current_date = datetime.now()
        start_date = (current_date - timedelta(days=6)).strftime("%Y-%m-%d")
        
        product_counts = {}
        
        with get_db_connection() as conn:
            events = conn.execute(
                """SELECT * FROM detection_events 
                   WHERE timestamp >= ? 
                   ORDER BY timestamp""",
                (start_date,)
            ).fetchall()

            for event in events:
                try:
                    if not event['roboflow_outputs']:
                        continue

                    roboflow_data = json.loads(event['roboflow_outputs'])
                    
                    if isinstance(roboflow_data, dict) and 'roboflow_predictions' in roboflow_data:
                        nestle_products = roboflow_data['roboflow_predictions']
                        
                        if isinstance(nestle_products, dict):
                            for product, count in nestle_products.items():
                                if product not in product_counts:
                                    product_counts[product] = 0
                                product_counts[product] += count

                except Exception as e:
                    logger.error(f"Error processing event for all products: {str(e)}")
                    continue

        sorted_products = [
            {'name': name, 'count': count}
            for name, count in sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        return jsonify(sorted_products)

    except Exception as e:
        logger.error(f"Error getting all products: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        filename = secure_filename(os.path.basename(filename))
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
            return send_from_directory(
                app.config['UPLOAD_FOLDER'], 
                filename,
                as_attachment=True
            )
        else:
            abort(404)
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {e}")
        abort(500)

def update_existing_iqi_scores():
    try:
        with get_db_connection() as conn:
            # Get all events without IQI scores
            events = conn.execute('''
                SELECT id, image_path 
                FROM detection_events 
                WHERE iqi_score IS NULL
            ''').fetchall()
            
            for event in events:
                try:
                    # Convert relative path to absolute path
                    image_path = event['image_path']
                    if image_path.startswith('uploads/'):
                        image_path = os.path.join(os.getcwd(), image_path)
                    
                    # Calculate IQI score
                    iqi_score = compute_IQI(image_path)
                    if iqi_score is not None:
                        conn.execute('''
                            UPDATE detection_events 
                            SET iqi_score = ? 
                            WHERE id = ?
                        ''', (iqi_score, event['id']))
                        conn.commit()
                        logger.info(f"Updated IQI score for event {event['id']}")
                except Exception as e:
                    logger.error(f"Error updating IQI score for event {event['id']}: {str(e)}")
                    continue
                    
        logger.info("Finished updating IQI scores for existing images")
    except Exception as e:
        logger.error(f"Error in update_existing_iqi_scores: {str(e)}")

# -----------------------------
# Main Application Entry
# -----------------------------
if __name__ == '__main__':
    try:
        init_db()
        update_existing_iqi_scores()  # Update IQI scores for existing images
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logger.error(f"Server startup error: {e}")
