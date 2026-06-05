from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_wtf.csrf import CSRFProtect, generate_csrf
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pickle
import os
import json
import requests
from datetime import datetime
from flask import flash
from crop_symptoms_data import CROP_SYMPTOMS, CROP_DISEASES, CROP_DISEASES

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'development-secret-key')

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Add CSRF token to all templates
@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
CROP_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'crop_data.csv')
PINCODE_SOIL_MAP_PATH = os.path.join(os.path.dirname(__file__), 'data', 'pincode_soil_map.csv')

# Check if required files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler file not found. Please ensure model.pkl and scaler.pkl exist.")

# Load model and scaler
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Load crop data
try:
    crop_data = pd.read_csv(CROP_DATA_PATH)
    pincode_soil_df = pd.read_csv(PINCODE_SOIL_MAP_PATH)
    pincode_soil_df['pincode'] = pincode_soil_df['pincode'].astype(str)
except Exception as e:
    print(f"Error loading data files: {str(e)}")
    crop_data = pd.DataFrame()
    pincode_soil_df = pd.DataFrame()

# OpenWeather API Configuration
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '').strip()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-form', methods=['GET', 'POST'])
def disease_form():
    crop = request.args.get('crop', '')
    # Note: Form submission is handled via AJAX to /predict endpoint
    # This route only renders the form template
    
    # Get unique pincodes for dropdown
    pincodes = []
    if not pincode_soil_df.empty:
        pincodes = pincode_soil_df[['pincode', 'district']].drop_duplicates().to_dict('records')
    
    return render_template('form.html', 
                         crop=crop, 
                         pincodes=pincodes,
                         pincode_soil_df=pincode_soil_df.to_dict('records'))

@app.route('/get-location')
def get_location():
    pincode = request.args.get('pincode')
    season = request.args.get('season', '').lower()
    if not pincode:
        return jsonify({'success': False, 'error': 'No pincode provided'})
    
    try:
        # Convert pincode to string and strip any whitespace
        pincode = str(pincode).strip()
        
        # Find matching pincode in the dataframe
        pincode_soil_df['pincode'] = pincode_soil_df['pincode'].astype(str).str.strip()
        match = pincode_soil_df[pincode_soil_df['pincode'] == pincode]
        
        if match.empty:
            # Special handling for known pincodes
            if pincode == '533352':
                match = pd.DataFrame([{
                    'pincode': '533352',
                    'district': 'East Godavari',
                    'soil_type': 'Alluvial Soil'
                }])
            elif pincode == '534201':
                match = pd.DataFrame([{
                    'pincode': '534201',
                    'district': 'West Godavari',
                    'soil_type': 'Alluvial / Deltaic Soil'
                }])
            else:
                return jsonify({
                    'success': False,
                    'error': 'pincode_not_found',
                    'message': f'Pincode {pincode} not found in our database.'
                })
        
        location_data = match.iloc[0].to_dict()
        district = location_data.get('district', '').strip()
        soil_type = location_data.get('soil_type', 'Unknown')
        
        # Get weather data with fallback values based on season
        weather_data = get_weather_data(district, season)
        
        # Prepare response
        response = {
            'success': True,
            'district': district,
            'soil_type': soil_type,
            'temperature': weather_data.get('temperature', 28.0),
            'humidity': weather_data.get('humidity', 65.0),
            'rainfall': weather_data.get('rainfall', 0)
        }
        
        if not weather_data.get('success'):
            response['warning'] = 'Using default weather data: ' + weather_data.get('error', 'Weather data not available')
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print(f"[ERROR] in get_location: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': 'server_error',
            'message': f'Error processing request: {str(e)}'
        })


@app.route('/get-crop-symptoms/<crop_name>', methods=['GET'])
def get_crop_symptoms(crop_name):
    """
    Return reference symptoms list for a given crop.
    This uses CROP_SYMPTOMS for suggestions only and does not affect prediction logic.
    """
    if not crop_name:
        return jsonify({"symptoms": []})

    # Normalize crop name: handle hyphens, spaces, and case
    crop_name_normalized = crop_name.strip().lower().replace('-', ' ')
    symptoms = []

    try:
        # Try exact match first
        for name, vals in CROP_SYMPTOMS.items():
            name_normalized = name.lower().replace('-', ' ')
            if name_normalized == crop_name_normalized:
                symptoms = vals or []
                break
        
        # If no exact match, try partial matching (e.g., "corn" -> "maize")
        if not symptoms:
            # Handle common aliases
            alias_map = {
                'corn': 'maize',
                'bitter gourd': 'bitter gourd',
                'ridge gourd': 'ridge gourd'
            }
            alias_crop = alias_map.get(crop_name_normalized, crop_name_normalized)
            for name, vals in CROP_SYMPTOMS.items():
                name_normalized = name.lower().replace('-', ' ')
                if name_normalized == alias_crop:
                    symptoms = vals or []
                    break
    except Exception as e:
        print(f"[ERROR] in get_crop_symptoms: {str(e)}")
        symptoms = []

    return jsonify({"symptoms": symptoms})


@app.route('/get-crop-diseases/<crop_name>', methods=['GET'])
def get_crop_diseases(crop_name):
    """
    Return reference disease list for a given crop.
    This uses CROP_DISEASES for suggestions only and does not affect prediction logic.
    """
    if not crop_name:
        return jsonify({"diseases": []})

    # Normalize crop name: handle hyphens, spaces, and case
    crop_name_normalized = crop_name.strip().lower().replace('-', ' ')
    diseases = []

    try:
        # Try exact match first
        for name, vals in CROP_DISEASES.items():
            name_normalized = name.lower().replace('-', ' ')
            if name_normalized == crop_name_normalized:
                diseases = vals or []
                break
        
        # If no exact match, try partial matching (e.g., "corn" -> "maize")
        if not diseases:
            # Handle common aliases
            alias_map = {
                'corn': 'maize',
                'bitter gourd': 'bitter gourd',
                'ridge gourd': 'ridge gourd'
            }
            alias_crop = alias_map.get(crop_name_normalized, crop_name_normalized)
            for name, vals in CROP_DISEASES.items():
                name_normalized = name.lower().replace('-', ' ')
                if name_normalized == alias_crop:
                    diseases = vals or []
                    break
    except Exception as e:
        print(f"[ERROR] in get_crop_diseases: {str(e)}")
        diseases = []

    return jsonify({"diseases": diseases})


@app.route('/get-location-data/<pincode>', methods=['GET'])
def get_location_data(pincode):
    """
    Simple helper API to get district, soil type and default weather values
    for a given pincode using only local project data (no external APIs).
    """
    # Ensure dataframe is loaded
    if pincode_soil_df.empty:
        return jsonify({
            "district": "",
            "soil_type": "",
            "temperature": 28.0,
            "humidity": 65.0,
            "rainfall": 0.0
        })

    # Normalize pincode as string
    pincode = str(pincode).strip()
    try:
        # Ensure consistent string type in dataframe
        df = pincode_soil_df.copy()
        df["pincode"] = df["pincode"].astype(str).str.strip()

        match = df[df["pincode"] == pincode]

        if match.empty:
            # If pincode not found, return defaults with empty district/soil
            return jsonify({
                "district": "",
                "soil_type": "",
                "temperature": 28.0,
                "humidity": 65.0,
                "rainfall": 0.0
            })

        row = match.iloc[0].to_dict()
        district = row.get("district", "") or ""
        soil_type = row.get("soil_type", "") or ""

        # Default realistic values for now (no external weather API)
        temperature = 28.0   # average warm-season temperature (°C)
        humidity = 65.0      # typical relative humidity (%)
        rainfall = 10.0      # moderate daily rainfall (mm)

        return jsonify({
            "district": district,
            "soil_type": soil_type,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall
        })

    except Exception:
        # On any unexpected error, still return default structure
        return jsonify({
            "district": "",
            "soil_type": "",
            "temperature": 28.0,
            "humidity": 65.0,
            "rainfall": 0.0
        })

def normalize_district_name(district):
    """Normalize district name by removing common suffixes and extra characters"""
    if not district:
        return None
    
    # Remove common suffixes and extra words
    suffixes = ['district', 'dist', 'dist.', 'dists', 'ds', 'd']
    words = district.strip().split(',')
    
    # Take only the first part before any comma
    name = words[0].strip()
    
    # Remove any suffix that matches our list (case insensitive)
    name_parts = name.split()
    if len(name_parts) > 1 and name_parts[-1].lower() in suffixes:
        name = ' '.join(name_parts[:-1])
    
    return name.strip()

def get_weather_data(district, season=''):
    """Helper function to fetch weather data from OpenWeather API with geocoding and fallback values"""
    if not district:
        return {'success': False, 'error': 'No district provided'}
    
    # Default fallback values based on season
    fallback_values = {
        'temperature': 28.0,  # Default average temperature
        'humidity': 65.0,     # Default average humidity
        'rainfall': 0         # Default rainfall (will be set based on season)
    }
    
    # Set rainfall based on season if provided
    if 'kharif' in season or 'monsoon' in season:
        fallback_values['rainfall'] = 45.0  # 20-70mm average for Kharif/Monsoon
    elif 'rabi' in season:
        fallback_values['rainfall'] = 5.0   # 0-10mm average for Rabi
    elif 'zaid' in season:
        fallback_values['rainfall'] = 12.5  # 5-20mm average for Zaid

    if not OPENWEATHER_API_KEY:
        return {
            'success': False,
            'error': 'OpenWeather API key is not configured',
            **fallback_values
        }
    
    try:
        # Normalize district name
        normalized_district = normalize_district_name(district)
        if not normalized_district:
            return {
                'success': False,
                'error': 'Invalid district name',
                **fallback_values
            }
        
        # Try to get weather data using OpenWeather API
        # First try with district name, then with coordinates if that fails
        
        # Try direct geocoding first
        geocode_url = (
            f'http://api.openweathermap.org/geo/1.0/direct?'
            f'q={requests.utils.quote(normalized_district)},IN&'
            f'limit=1&appid={OPENWEATHER_API_KEY}'
        )
        
        try:
            geocode_response = requests.get(geocode_url, timeout=5)
            if geocode_response.status_code == 200:
                location_data = geocode_response.json()
                if location_data:
                    # Extract coordinates
                    lat = location_data[0].get('lat')
                    lon = location_data[0].get('lon')
                    
                    if lat and lon:
                        # Get weather data using coordinates
                        weather_url = (
                            f'https://api.openweathermap.org/data/2.5/weather?'
                            f'lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric'
                        )
                        
                        weather_response = requests.get(weather_url, timeout=5)
                        if weather_response.status_code == 200:
                            weather_data = weather_response.json()
                            
                            # Extract relevant weather information
                            main_data = weather_data.get('main', {})
                            temperature = main_data.get('temp', fallback_values['temperature'])
                            humidity = main_data.get('humidity', fallback_values['humidity'])
                            
                            # Get rainfall (1h) if available, otherwise use fallback
                            rainfall = fallback_values['rainfall']
                            if 'rain' in weather_data and '1h' in weather_data['rain']:
                                rainfall = float(weather_data['rain']['1h'])
                            
                            return {
                                'success': True,
                                'temperature': temperature,
                                'humidity': humidity,
                                'rainfall': rainfall
                            }
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"[WARNING] Weather API request failed: {str(e)}")
        
        # If we get here, return fallback values with a warning
        return {
            'success': False,
            'error': 'Using fallback weather data',
            **fallback_values
        }
        
    except Exception as e:
        print(f"[ERROR] in get_weather_data: {str(e)}")
        return {
            'success': False,
            'error': f'Error getting weather data: {str(e)}',
            **fallback_values
        }

# This endpoint is deprecated - using /get-location instead
@app.route('/get-weather')
def get_weather():
    return jsonify({
        'success': False,
        'error': 'This endpoint is deprecated. Use /get-location instead.'
    }), 410  # 410 Gone

# Predict disease based on form input
@app.route('/predict', methods=['POST'])
@csrf.exempt  # Temporarily disable CSRF for testing
def predict_disease():
    try:
        # Support both JSON (AJAX) and traditional form submissions
        data = {}
        if request.is_json:
            data = request.get_json() or {}
        else:
            # Map HTML form field names to the expected keys
            form = request.form
            data = {
                'crop_name': form.get('crop') or form.get('crop_name'),
                'pincode': form.get('pincode'),
                'season': form.get('season'),
                'temperature': form.get('temperature'),
                'humidity': form.get('humidity'),
                'rainfall': form.get('rainfall'),
                'soil_type': form.get('soil_type') or form.get('soilType'),
                'district': form.get('district'),
                'symptoms': form.get('symptoms'),
            }

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        # Get required fields with proper validation
        crop_name = data.get('crop_name')
        pincode = data.get('pincode')
        season = data.get('season')
        
        # Convert numeric fields with proper error handling
        try:
            temperature = float(data.get('temperature', 0))
            rainfall = float(data.get('rainfall', 0))
            humidity = float(data.get('humidity', 0))
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'Invalid numeric value for temperature, rainfall, or humidity'
            }), 400

        # Validate required fields
        if not all([crop_name, pincode, season]):
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            }), 400

        # Prefer district/soil_type from incoming data; otherwise derive from pincode
        soil_type = data.get('soil_type') or 'Unknown'
        district = data.get('district') or 'Unknown'
        if (soil_type == 'Unknown' or district == 'Unknown') and pincode:
            try:
                soil_data = pincode_soil_df[pincode_soil_df['pincode'] == str(pincode)].iloc[0]
                soil_type = soil_data.get('soil_type', soil_type)
                district = soil_data.get('district', district)
            except Exception:
                pass

        # Observed symptoms (for debugging/logic we keep variable named observed_symptoms)
        observed_symptoms = data.get('symptoms', '') or ''

        # Debug print of all key input fields
        print("[DEBUG] /predict input:", {
            "crop": crop_name,
            "season": season,
            "district": district,
            "soil_type": soil_type,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "observed_symptoms": observed_symptoms,
        })

        # Make prediction
        disease, confidence, is_healthy = predict_disease_rules(
            crop_name=crop_name,
            season=season,
            soil_type=soil_type,
            temperature=temperature,
            rainfall=rainfall,
            symptoms=observed_symptoms,  # Symptoms are optional
            humidity=humidity  # Add humidity for prediction
        )

        # Prepare and return response
        return jsonify({
            'success': True,
            'crop': crop_name,
            'district': district,
            'soil_type': soil_type,
            'season': season,
            'temperature': temperature,
            'rainfall': rainfall,
            'humidity': humidity,
            'disease': disease,
            'confidence': f"{confidence}%",
            'is_healthy': is_healthy,
            'advice': get_disease_advice(disease, crop_name)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing request: {str(e)}'
        }), 500
        symptoms = request.form.get('symptoms', '').lower()
        
        # Get soil type from pincode
        soil_type = 'Unknown'
        district = 'Unknown'
        if pincode:
            try:
                soil_data = pincode_soil_df[pincode_soil_df['pincode'] == str(pincode)].iloc[0]
                soil_type = soil_data.get('soil_type', 'Unknown')
                district = soil_data.get('district', 'Unknown')
            except:
                pass
        
        # Simple rule-based prediction (replace with actual model prediction)
        disease, confidence, is_healthy = predict_disease_rules(
            crop_name=crop_name,
            season=season,
            soil_type=soil_type,
            temperature=temperature,
            rainfall=rainfall,
            symptoms=symptoms,
            humidity=None  # Humidity not available in this code path
        )
        
        # Prepare response
        result = {
            'success': True,
            'crop': crop_name,
            'district': district,
            'pincode': pincode,
            'soil_type': soil_type,
            'season': season,
            'disease': disease,
            'confidence': f"{confidence}%",
            'is_healthy': is_healthy,
            'advice': get_disease_advice(disease, crop_name)
        }
        
        return render_template('form.html', 
                             form_data=request.form,
                             result=result,
                             pincodes=pincode_soil_df[['pincode', 'district']].drop_duplicates().to_dict('records'),
                             pincode_soil_df=pincode_soil_df.to_dict('records'))
        
    except Exception as e:
        print(f"Error in predict_disease: {str(e)}")
        flash(f"An error occurred: {str(e)}", 'error')
        return redirect(url_for('disease_form'))

def predict_disease_rules(crop_name, season, soil_type, temperature, rainfall, symptoms, humidity=None):
    """Rule-based disease prediction based on symptoms and crop-specific diseases"""
    
    # Normalize inputs
    symptoms_lower = (symptoms or '').lower().strip()
    crop_name_lower = (crop_name or '').lower().strip()
    
    # If no symptoms provided, check if environmental conditions suggest disease risk
    if not symptoms_lower:
        if temperature > 35 and rainfall > 100:
            return 'High Risk: Environmental Stress', 65, False
        elif temperature < 10 or temperature > 40:
            return 'High Risk: Temperature Stress', 60, False
        else:
            return 'Healthy', 90, True
    
    # Get crop-specific diseases for matching
    crop_diseases = []
    try:
        # Normalize crop name for lookup (handle hyphens, spaces)
        crop_normalized = crop_name_lower.replace('-', ' ')
        for name, diseases in CROP_DISEASES.items():
            if name.lower().replace('-', ' ') == crop_normalized:
                crop_diseases = diseases
                break
    except Exception:
        pass
    
    # Comprehensive disease keyword detection
    disease_keywords = {
        # Blight diseases
        'blight': ['blight', 'ascochyta', 'alternaria', 'early blight', 'late blight', 'gummy stem blight'],
        # Rot diseases
        'rot': ['rot', 'seed rot', 'seedling rot', 'collar rot', 'boll rot', 'root rot', 'stem rot', 'charcoal rot'],
        # Mildew diseases
        'mildew': ['mildew', 'powdery mildew', 'downy mildew'],
        # Wilt diseases
        'wilt': ['wilt', 'fusarium wilt', 'bacterial wilt'],
        # Rust diseases
        'rust': ['rust', 'leaf rust', 'stem rust', 'white rust'],
        # Spot diseases
        'spot': ['spot', 'leaf spot', 'angular leaf spot', 'bacterial spot', 'target spot', 'brown leaf spots'],
        # Yellowing/Chlorosis
        'yellowing': ['yellow', 'yellowing', 'yellow patches', 'chlorosis', 'yellowing leaves'],
        # Mold/Fungal
        'mold': ['mold', 'grey mold', 'gray mold', 'grain mold', 'botrytis gray mold', 'botrytis grey mold'],
        # Virus
        'virus': ['virus', 'mosaic virus', 'viral'],
        # Bacterial
        'bacterial': ['bacterial', 'bacterial blight', 'bacterial spot'],
        # Seed/Seedling issues
        'seedling': ['seedling', 'seed rot', 'seedling blight', 'seedling blights', 'seedling disease'],
        # Anthracnose
        'anthracnose': ['anthracnose'],
        # Stripe/Blotch
        'stripe_blotch': ['stripe', 'blotch', 'net blotch', 'barley stripe'],
        # Sclerotinia
        'sclerotinia': ['sclerotinia', 'sclerotinia stem rot'],
        # Wilted/Wilting
        'wilting': ['wilted', 'wilting', 'wilted leaves'],
    }
    
    # Check symptoms against disease keywords (priority order)
    detected_diseases = []
    confidence_scores = []
    
    for disease_type, keywords in disease_keywords.items():
        for keyword in keywords:
            if keyword in symptoms_lower:
                # Try to match with crop-specific disease names
                matched_disease = None
                for crop_disease in crop_diseases:
                    if keyword in crop_disease.lower():
                        matched_disease = crop_disease
                        break
                
                if matched_disease:
                    detected_diseases.append(matched_disease)
                    confidence_scores.append(85)
                else:
                    # Generic disease name based on keyword
                    disease_name_map = {
                        'blight': 'Blight Disease',
                        'rot': 'Rot Disease',
                        'mildew': 'Mildew Disease',
                        'wilt': 'Wilt Disease',
                        'rust': 'Rust Disease',
                        'spot': 'Leaf Spot Disease',
                        'yellowing': 'Yellowing Disease / Nutrient Deficiency',
                        'mold': 'Mold / Fungal Infection',
                        'virus': 'Viral Disease',
                        'bacterial': 'Bacterial Disease',
                        'seedling': 'Seedling Disease',
                        'anthracnose': 'Anthracnose',
                        'stripe_blotch': 'Stripe/Blotch Disease',
                        'sclerotinia': 'Sclerotinia Stem Rot',
                        'wilting': 'Wilting Disease',
                    }
                    detected_diseases.append(disease_name_map.get(disease_type, 'Disease Detected'))
                    confidence_scores.append(75)
                break  # Found a match, move to next check
    
    # If diseases detected, return the first one with highest confidence
    if detected_diseases:
        # Get the disease with highest confidence
        max_idx = confidence_scores.index(max(confidence_scores))
        return detected_diseases[max_idx], confidence_scores[max_idx], False
    
    # Check environmental conditions as secondary indicator
    if temperature > 30 and rainfall > 100:
        return 'Fungal Infection (Environmental Risk)', 70, False
    elif temperature < 10 or temperature > 40:
        return 'Temperature Stress', 65, False
    elif humidity and humidity > 85:
        return 'High Humidity Risk - Fungal Disease Possible', 60, False
    
    # If no disease indicators found, return healthy
    return 'Healthy', 90, True

def get_disease_advice(disease, crop_name):
    """Get treatment advice based on disease"""
    advice = {
        # Blight diseases
        'Blight Disease': f'Apply fungicides containing chlorothalonil or mancozeb. Remove and destroy infected {crop_name} plants. Improve air circulation and reduce humidity.',
        'Late Blight': f'Apply fungicides containing chlorothalonil or mancozeb. Remove and destroy infected {crop_name} plants. Improve air circulation.',
        'Ascochyta blight': f'Apply fungicides like azoxystrobin or tebuconazole. Remove infected plant debris. Practice crop rotation for {crop_name}.',
        'Alternaria blight': f'Apply fungicides containing mancozeb or chlorothalonil. Remove infected leaves. Ensure proper spacing for {crop_name} plants.',
        'Gummy stem blight': f'Apply fungicides like thiophanate-methyl. Remove and destroy infected plant parts. Improve ventilation for {crop_name}.',
        
        # Rot diseases
        'Rot Disease': f'Apply appropriate fungicides. Remove infected plant parts immediately. Improve drainage and avoid overwatering {crop_name}.',
        'Seed rot': f'Use treated seeds with fungicides. Ensure proper seed storage conditions. Practice seed treatment before planting {crop_name}.',
        'Seedling rot': f'Apply seed treatment fungicides. Ensure well-drained soil. Avoid overwatering {crop_name} seedlings.',
        'Collar rot': f'Apply fungicides like carbendazim. Remove infected plants. Improve soil drainage for {crop_name}.',
        'Boll rot': f'Apply fungicides and remove infected bolls. Improve air circulation. Harvest {crop_name} promptly when mature.',
        'Charcoal rot': f'Apply fungicides and practice crop rotation. Ensure proper irrigation. Remove infected {crop_name} plants.',
        
        # Mildew diseases
        'Mildew Disease': f'Apply fungicides containing sulfur, myclobutanil, or tebuconazole. Improve air circulation. Water at base, not leaves for {crop_name}.',
        'Powdery mildew': f'Apply fungicides like sulfur or myclobutanil. Improve air circulation. Remove infected leaves from {crop_name}.',
        'Downy mildew': f'Apply fungicides containing mancozeb or metalaxyl. Improve drainage and reduce humidity for {crop_name}.',
        
        # Wilt diseases
        'Wilt Disease': f'Apply fungicides like benomyl or thiophanate-methyl. Remove infected plants. Practice crop rotation for {crop_name}.',
        'Fusarium wilt': f'Apply fungicides and use resistant varieties. Remove infected plants. Practice crop rotation for {crop_name}.',
        'Bacterial wilt': f'Use copper-based bactericides. Remove infected plants. Practice crop rotation. Avoid overhead watering for {crop_name}.',
        
        # Rust diseases
        'Rust Disease': f'Apply fungicides containing myclobutanil or tebuconazole. Remove and destroy infected leaves from {crop_name}.',
        'Leaf rust': f'Apply fungicides like propiconazole. Remove infected leaves. Ensure proper spacing for {crop_name}.',
        'Stem rust': f'Apply fungicides and remove infected plant parts. Use resistant varieties of {crop_name}.',
        'White rust': f'Apply fungicides containing mancozeb. Remove infected leaves. Improve air circulation for {crop_name}.',
        
        # Spot diseases
        'Leaf Spot Disease': f'Apply fungicides like chlorothalonil or mancozeb. Remove infected leaves. Avoid overhead watering for {crop_name}.',
        'Bacterial Spot': f'Use copper-based bactericides. Practice crop rotation. Avoid overhead watering for {crop_name}.',
        'Brown leaf spots': f'Apply fungicides. Remove infected leaves. Improve air circulation and reduce humidity for {crop_name}.',
        'Angular leaf spot': f'Apply copper-based fungicides. Remove infected leaves. Practice crop rotation for {crop_name}.',
        
        # Yellowing/Chlorosis
        'Yellowing Disease / Nutrient Deficiency': f'Check soil pH and nutrient levels. Apply appropriate fertilizers. May indicate nitrogen deficiency in {crop_name}. Consider soil testing.',
        'Yellowing leaves': f'Check for nutrient deficiencies (especially nitrogen). Apply balanced fertilizer. Check for root diseases in {crop_name}.',
        'Yellow patches on leaves': f'Apply appropriate fertilizers. Check for nutrient deficiencies. May indicate disease in {crop_name} - consult expert.',
        
        # Mold/Fungal
        'Mold / Fungal Infection': f'Apply appropriate fungicide. Ensure proper spacing between {crop_name} plants. Water at the base, not the leaves.',
        'Grain mold': f'Apply fungicides before harvest. Ensure proper drying of {crop_name} grains. Store in dry conditions.',
        'Botrytis gray mold': f'Apply fungicides like iprodione. Remove infected plant parts. Improve air circulation for {crop_name}.',
        
        # Virus
        'Viral Disease': f'Remove and destroy infected plants immediately. Control insect vectors. Use virus-free seeds for {crop_name}.',
        'Mosaic virus': f'Remove infected plants. Control aphids and other vectors. Use virus-free seeds for {crop_name}.',
        
        # Bacterial
        'Bacterial Disease': f'Use copper-based bactericides. Remove infected plant parts. Practice crop rotation for {crop_name}.',
        'Bacterial Blight': f'Apply copper-based bactericides. Remove infected leaves. Avoid overhead watering for {crop_name}.',
        
        # Seedling issues
        'Seedling Disease': f'Use treated seeds with fungicides. Ensure proper seed storage. Practice good seedbed management for {crop_name}.',
        'Seedling blight': f'Apply seed treatment fungicides. Ensure well-drained soil. Avoid overwatering {crop_name} seedlings.',
        
        # Specific diseases
        'Anthracnose': f'Apply fungicides like chlorothalonil or mancozeb. Remove infected plant parts. Improve air circulation for {crop_name}.',
        'Net Blotch': f'Apply fungicides like propiconazole. Remove infected leaves. Practice crop rotation for {crop_name}.',
        'Barley Stripe': f'Use treated seeds with fungicides. Practice crop rotation. Remove infected plants for {crop_name}.',
        'Sclerotinia Stem Rot': f'Apply fungicides like carbendazim. Remove infected plants. Improve air circulation for {crop_name}.',
        'Wilting Disease': f'Check for root diseases. Apply appropriate fungicides. Improve soil drainage for {crop_name}.',
        
        # Environmental
        'Fungal Infection (Environmental Risk)': f'Apply appropriate fungicide. Ensure proper spacing between {crop_name} plants. Water at the base, not the leaves.',
        'High Risk: Environmental Stress': f'Monitor {crop_name} plants closely. Apply preventive fungicides. Ensure proper irrigation and drainage.',
        'Temperature Stress': f'Provide shade or protection from extreme temperatures. Ensure adequate irrigation for {crop_name}. Monitor for secondary infections.',
        'High Humidity Risk - Fungal Disease Possible': f'Improve air circulation. Apply preventive fungicides. Reduce humidity around {crop_name} plants.',
        
        # Healthy
        'Healthy': f'Your {crop_name} plants appear healthy. Continue good agricultural practices and monitor regularly.',
    }
    
    # Try exact match first
    if disease in advice:
        return advice[disease]
    
    # Try partial match for crop-specific diseases
    for key, value in advice.items():
        if disease.lower() in key.lower() or key.lower() in disease.lower():
            return value
    
    # Default advice
    return f'Consult with a local agricultural expert for specific treatment recommendations for {disease} in {crop_name} plants.'

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
