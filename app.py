from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import joblib
import pandas as pd
from ekstraksi_glcm import extract_glcm_features
from segmentasi import segment_leaf
from scipy.spatial.distance import euclidean
import tempfile
import os
import time
import traceback

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, origins=['*'])

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found on this server.',
        'status_code': 404
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An internal server error occurred.',
        'status_code': 500
    }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({ 
        'error': 'Request Entity Too Large',
        'message': 'The uploaded file is too large.',
        'status_code': 413
    }), 413

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad Request',
        'message': 'The request could not be understood by the server.',
        'status_code': 400
    }), 400

# Load model, scaler, and data with error handling
try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

try:
    scaler = joblib.load("scaler.pkl")
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading scaler: {str(e)}")
    scaler = None

try:
    glcm_data = pd.read_csv("glcm_features.csv")
    print("GLCM features data loaded successfully")
except Exception as e:
    print(f"Error loading GLCM features: {str(e)}")
    glcm_data = None

# Serve static files
@app.route('/')
def serve_website():
    return send_from_directory('website', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('website', filename)

@app.route('/api')
def home():
    return jsonify({
        'message': 'Durian Leaf Classification API',
        'endpoints': {
            'POST /predict': 'Upload image for classification',
            'GET /health': 'Check API health status'
        },
        'status': 'running'
    })

@app.route('/health')
def health_check():
    model_status = "loaded" if model is not None else "not loaded"
    scaler_status = "loaded" if scaler is not None else "not loaded"
    data_status = "loaded" if glcm_data is not None else "not loaded"
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'scaler_status': scaler_status,
        'data_status': data_status,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        # Check if model and data are loaded
        if model is None:
            return jsonify({
                'error': 'Model not available',
                'message': 'Machine learning model is not loaded. Please check server configuration.',
                'status_code': 500
            }), 500
        
        if glcm_data is None:
            return jsonify({
                'error': 'Data not available',
                'message': 'GLCM features data is not loaded. Please check server configuration.',
                'status_code': 500
            }), 500
        
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image provided',
                'message': 'Please upload an image file.',
                'status_code': 400
            }), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select an image file to upload.',
                'status_code': 400
            }), 400
        
        # Check file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({
                'error': 'Invalid file type',
                'message': 'Please upload a valid image file (PNG, JPG, JPEG, GIF, BMP).',
                'status_code': 400
            }), 400
        
        # Read and decode image
        try:
            img_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                return jsonify({
                    'error': 'Invalid image',
                    'message': 'Could not decode the uploaded image. Please upload a valid image file.',
                    'status_code': 400
                }), 400
                
        except Exception as e:
            return jsonify({
                'error': 'Image processing error',
                'message': f'Error processing uploaded image: {str(e)}',
                'status_code': 400
            }), 400

        # Create temporary file for segmentation
        temp_file = None
        temp_file_path = None
        segmented_img = None
        
        try:
            # Create temporary file with delete=False to handle manually
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file_path = temp_file.name
            temp_file.close()  # Close the file handle first
            
            # Write image to temporary file
            cv2.imwrite(temp_file_path, img)
            
            # Perform segmentation
            segmented_img = segment_leaf(temp_file_path)
            
            if segmented_img is None:
                return jsonify({
                    'error': 'Segmentation failed',
                    'message': 'Could not segment the leaf from the image.',
                    'status_code': 500
                }), 500
            
        except Exception as e:
            return jsonify({
                'error': 'Segmentation error',
                'message': f'Error during image segmentation: {str(e)}',
                'status_code': 500
            }), 500
        
        finally:
            # Clean up temporary file with retry mechanism
            if temp_file_path is not None:
                try:
                    time.sleep(0.1)  # Small delay to ensure file is released
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                except PermissionError:
                    # If still can't delete, try again after a longer delay
                    try:
                        time.sleep(1)
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                    except:
                        pass  # If still fails, let it be (temp files will be cleaned by system)

        # Extract GLCM features from segmented image
        try:
            input_features = extract_glcm_features(segmented_img)
            
            if input_features is None or len(input_features) != 4:
                return jsonify({
                    'error': 'Ekstraksi fitur gagal',
                    'message': 'Tidak dapat mengekstrak fitur GLCM dari gambar tersegmentasi.',
                    'status_code': 500
                }), 500
            
            # Scale the input features for KNN
            if scaler is not None:
                input_features_scaled = scaler.transform([input_features])[0]
            else:
                return jsonify({
                    'error': 'Scaler tidak tersedia',
                    'message': 'Scaler untuk normalisasi fitur tidak dimuat.',
                    'status_code': 500
                }), 500
                
        except Exception as e:
            return jsonify({
                'error': 'Kesalahan ekstraksi fitur',
                'message': f'Kesalahan mengekstrak fitur GLCM: {str(e)}',
                'status_code': 500
            }), 500

        # Calculate Euclidean distances with all existing features (menggunakan scaled features)
        try:
            distances = []
            # Scale all existing features for comparison
            existing_features_scaled = scaler.transform(glcm_data[['contrast', 'correlation', 'energy', 'homogeneity']])
            
            for index, row in glcm_data.iterrows():
                existing_features = existing_features_scaled[index]
                distance = euclidean(input_features_scaled, existing_features)
                distances.append({
                    'distance': float(distance),
                    'label': row['label'],
                    'index': int(index)
                })
            
            # Sort by distance (closest first)
            distances.sort(key=lambda x: x['distance'])
            
        except Exception as e:
            return jsonify({
                'error': 'Kesalahan perhitungan jarak',
                'message': f'Kesalahan menghitung jarak: {str(e)}',
                'status_code': 500
            }), 500

        # Get prediction from the trained model (menggunakan scaled features)
        try:
            raw_prediction = model.predict([input_features_scaled])[0]
            model_prediction = format_prediction_result(raw_prediction)
            
        except Exception as e:
            return jsonify({
                'error': 'Kesalahan prediksi',
                'message': f'Kesalahan membuat prediksi: {str(e)}',
                'status_code': 500
            }), 500
        
        # Find closest match
        closest_match = distances[0] if distances else None
        
        if closest_match is None:
            return jsonify({
                'error': 'No matches found',
                'message': 'Could not find any matching features in the database.',
                'status_code': 500
            }), 500
        
        # Return successful response
        response_data = {
            'success': True,
            'model_prediction': str(model_prediction),
            'input_features': {
                'contrast': float(input_features[0]),
                'correlation': float(input_features[1]),
                'energy': float(input_features[2]),
                'homogeneity': float(input_features[3])
            },
            'total_comparisons': len(distances)
        }

        # Hanya tampilkan matches jika bukan noclass
        if raw_prediction != "noclass":
            response_data['closest_match'] = {
                'label': closest_match['label'],
                'distance': closest_match['distance'],
                'index': closest_match['index']
            }
            response_data['top_5_matches'] = distances[:5]
        else:
            # Untuk noclass, set matches sebagai null atau tidak include sama sekali
            response_data['closest_match'] = None
            response_data['top_5_matches'] = []

        response = jsonify(response_data)
        
        # Add CORS headers to response
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        # Catch-all for any unexpected errors
        error_trace = traceback.format_exc()
        print(f"Unexpected error: {error_trace}")
        
        response = jsonify({
            'error': 'Unexpected error',
            'message': f'An unexpected error occurred: {str(e)}',
            'status_code': 500
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500

# Tambahkan fungsi untuk menangani prediksi noclass
def format_prediction_result(prediction):
    """Format prediction result for better user experience"""
    if prediction == "noclass":
        return "Bukan Daun Durian, Input gambar yang sesuai"
    else:
        return str(prediction)

if __name__ == '__main__':
    print("Starting Durian Leaf Classification Server...")
    print("Server will be available at: http://localhost:5000")
    print("Website will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
