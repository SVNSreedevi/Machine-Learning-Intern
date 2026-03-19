# app.py - Updated with comprehensive data analytics
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from model import train_and_save_model, load_model, predict_maintenance
import os
import json
import logging
from datetime import datetime
import secrets

app = Flask(__name__)

# Global variables for model
model = None
encoders = None
scaler = None
model_info = "Not loaded"
model_initialized = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_model():
    global model, encoders, scaler, model_info, model_initialized
    
    if model_initialized:
        return
        
    print("Initializing model...")
    
    # Try to load existing model first
    model, encoders, scaler, model_info, _ = load_model()
    
    if model is None:
        print("No pre-trained model found. Training new model...")
        try:
            if os.path.exists('vehicle_maintenance_data.csv'):
                df = pd.read_csv('vehicle_maintenance_data.csv')
                model, encoders, scaler, model_name, accuracy = train_and_save_model(df)
                model_info = f"{model_name} (Accuracy: {accuracy:.2%})"
                print(f"Model trained: {model_info}")
                
                # Log model training
                logger.info(f"Model trained successfully: {model_info}")
            else:
                print("Dataset not found. Model cannot be trained.")
                model_info = "Dataset not available"
                logger.warning("Dataset file not found for model training")
        except Exception as e:
            print(f"Error training model: {e}")
            model_info = f"Training failed: {str(e)}"
            logger.error(f"Model training failed: {str(e)}")
    
    model_initialized = True

def prepare_visualization_data():
    """Prepare data for charts and visualizations"""
    try:
        if not os.path.exists('vehicle_maintenance_data.csv'):
            return {'error': 'Dataset file not found'}
        
        df = pd.read_csv('vehicle_maintenance_data.csv')
        
        # Maintenance distribution
        maintenance_dist = {
            'labels': ['🚗 Maintenance Needed', '✅ No Maintenance Needed'],
            'data': [
                int(df['Need_Maintenance'].sum()),
                int(len(df) - df['Need_Maintenance'].sum())
            ],
            'colors': ['#ff6b6b', '#51cf66']
        }
        
        # Vehicle types distribution
        vehicle_types = df['Vehicle_Model'].value_counts()
        vehicle_dist = {
            'labels': vehicle_types.index.tolist(),
            'data': vehicle_types.values.tolist(),
            'colors': ['#ffd43b', '#228be6', '#fa5252', '#40c057', '#be4bdb', '#fd7e14']
        }
        
        # Fuel types distribution
        fuel_types = df['Fuel_Type'].value_counts()
        fuel_dist = {
            'labels': fuel_types.index.tolist(),
            'data': fuel_types.values.tolist(),
            'colors': ['#ffa94d', '#4dabf7', '#69db7c']
        }
        
        # Maintenance history
        maint_history = df['Maintenance_History'].value_counts()
        history_dist = {
            'labels': maint_history.index.tolist(),
            'data': maint_history.values.tolist(),
            'colors': ['#51cf66', '#ffd43b', '#ff6b6b']
        }
        
        # Component conditions
        tire_condition = df['Tire_Condition'].value_counts()
        brake_condition = df['Brake_Condition'].value_counts()
        battery_status = df['Battery_Status'].value_counts()
        
        components_data = {
            'tire': {
                'labels': tire_condition.index.tolist(),
                'data': tire_condition.values.tolist(),
                'colors': ['#51cf66', '#ffd43b', '#ff6b6b']
            },
            'brake': {
                'labels': brake_condition.index.tolist(),
                'data': brake_condition.values.tolist(),
                'colors': ['#51cf66', '#ffd43b', '#ff6b6b']
            },
            'battery': {
                'labels': battery_status.index.tolist(),
                'data': battery_status.values.tolist(),
                'colors': ['#51cf66', '#ffd43b', '#ff6b6b']
            }
        }
        
        # Transmission types
        transmission_types = df['Transmission_Type'].value_counts()
        transmission_dist = {
            'labels': transmission_types.index.tolist(),
            'data': transmission_types.values.tolist(),
            'colors': ['#74c0fc', '#ffa94d']
        }
        
        # Age vs Maintenance correlation
        age_groups = pd.cut(df['Vehicle_Age'], bins=[0, 2, 5, 10, 20], labels=['0-2', '3-5', '6-10', '10+'])
        age_maintenance = df.groupby(age_groups)['Need_Maintenance'].mean() * 100
        
        age_correlation = {
            'labels': age_maintenance.index.tolist(),
            'data': [round(x, 1) for x in age_maintenance.values.tolist()],
            'colors': ['#a9e34b', '#ffd43b', '#ffa94d', '#ff6b6b']
        }
        
        return {
            'maintenance_dist': maintenance_dist,
            'vehicle_dist': vehicle_dist,
            'fuel_dist': fuel_dist,
            'history_dist': history_dist,
            'components_data': components_data,
            'transmission_dist': transmission_dist,
            'age_correlation': age_correlation,
            'total_vehicles': len(df),
            'maintenance_percentage': round((df['Need_Maintenance'].sum() / len(df)) * 100, 2),
            'avg_mileage': round(df['Mileage'].mean(), 2),
            'avg_age': round(df['Vehicle_Age'].mean(), 2),
            'avg_issues': round(df['Reported_Issues'].mean(), 2),
            'success': True
        }
    
    except Exception as e:
        logger.error(f"Error preparing visualization data: {str(e)}")
        return {'error': f'Error preparing data: {str(e)}'}

def get_dataset_statistics():
    """Get comprehensive dataset statistics"""
    try:
        if not os.path.exists('vehicle_maintenance_data.csv'):
            return {'error': 'Dataset file not found'}
        
        df = pd.read_csv('vehicle_maintenance_data.csv')
        
        # Basic statistics
        stats = {
            'total_records': len(df),
            'maintenance_count': int(df['Need_Maintenance'].sum()),
            'no_maintenance_count': int(len(df) - df['Need_Maintenance'].sum()),
            'maintenance_rate': round((df['Need_Maintenance'].sum() / len(df)) * 100, 2),
            'columns': list(df.columns),
            'data_types': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # Numerical columns statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        stats['numerical_stats'] = df[numerical_cols].describe().to_dict()
        
        # Categorical columns value counts
        categorical_cols = df.select_dtypes(include=['object']).columns
        stats['categorical_stats'] = {}
        for col in categorical_cols:
            stats['categorical_stats'][col] = df[col].value_counts().to_dict()
        
        # Correlation matrix
        stats['correlation_matrix'] = df[numerical_cols].corr().round(3).to_dict()
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting dataset statistics: {str(e)}")
        return {'error': f'Error analyzing dataset: {str(e)}'}

def get_sample_data(rows=10):
    """Get sample data from dataset"""
    try:
        if not os.path.exists('vehicle_maintenance_data.csv'):
            return {'error': 'Dataset file not found'}
        
        df = pd.read_csv('vehicle_maintenance_data.csv')
        sample_data = df.head(rows).to_dict('records')
        
        return {
            'sample_data': sample_data,
            'columns': list(df.columns),
            'total_rows': len(df)
        }
    
    except Exception as e:
        logger.error(f"Error getting sample data: {str(e)}")
        return {'error': f'Error getting sample data: {str(e)}'}

def get_feature_analysis():
    """Analyze feature importance and relationships"""
    try:
        if not os.path.exists('vehicle_maintenance_data.csv'):
            return {'error': 'Dataset file not found'}
        
        df = pd.read_csv('vehicle_maintenance_data.csv')
        
        # Feature correlations with target
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numerical_cols].corr()['Need_Maintenance'].drop('Need_Maintenance').sort_values(ascending=False)
        
        feature_analysis = {
            'feature_correlations': correlations.round(3).to_dict(),
            'top_correlated_features': correlations.head(5).to_dict()
        }
        
        return feature_analysis
    
    except Exception as e:
        logger.error(f"Error in feature analysis: {str(e)}")
        return {'error': f'Error in feature analysis: {str(e)}'}

def log_prediction(input_data, prediction_result, confidence):
    """Log prediction details for analytics"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': secrets.token_hex(8),
            'input_data': input_data,
            'prediction': prediction_result,
            'confidence': confidence,
            'model_used': model_info
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Append to prediction log file
        log_file = 'logs/predictions.log'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
        logger.info(f"Prediction logged: {prediction_result} with {confidence}% confidence")
        
    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}")

def get_prediction_analytics():
    """Get analytics from prediction logs"""
    try:
        log_file = 'logs/predictions.log'
        if not os.path.exists(log_file):
            return {'total_predictions': 0, 'maintenance_rate': 0, 'avg_confidence': 0}
        
        predictions = []
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    predictions.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        if not predictions:
            return {'total_predictions': 0, 'maintenance_rate': 0, 'avg_confidence': 0}
        
        total_predictions = len(predictions)
        maintenance_count = sum(1 for p in predictions if p.get('prediction') == 'Maintenance Needed')
        avg_confidence = sum(p.get('confidence', 0) for p in predictions) / total_predictions
        
        return {
            'total_predictions': total_predictions,
            'maintenance_rate': round((maintenance_count / total_predictions) * 100, 2),
            'avg_confidence': round(avg_confidence, 2)
        }
        
    except Exception as e:
        logger.error(f"Error getting prediction analytics: {str(e)}")
        return {'total_predictions': 0, 'maintenance_rate': 0, 'avg_confidence': 0}

@app.route('/')
def index():
    initialize_model()
    
    # Load visualization data
    visualization_data = prepare_visualization_data()
    
    # Get prediction analytics
    prediction_analytics = get_prediction_analytics()
    
    # Get dataset statistics and analytics
    dataset_stats = get_dataset_statistics()
    sample_data = get_sample_data(5)  # Get 5 sample rows
    feature_analysis = get_feature_analysis()
    
    # Combine all data
    combined_data = {
        **visualization_data,
        'prediction_analytics': prediction_analytics,
        'dataset_stats': dataset_stats,
        'sample_data': sample_data,
        'feature_analysis': feature_analysis,
        'model_status': model_info,
        'server_time': datetime.now().isoformat()
    }
    
    return render_template('index.html', 
                         model_info=model_info,
                         visualization_data=json.dumps(combined_data))

@app.route('/predict', methods=['POST'])
def predict():
    initialize_model()
    
    if model is None:
        return jsonify({'error': 'Model not available. Please check if the model was trained successfully.'})
    
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Convert to the format expected by our model
        input_data = {
            'Vehicle_Model': form_data['vehicle_model'],
            'Mileage': float(form_data['mileage']),
            'Maintenance_History': form_data['maintenance_history'],
            'Reported_Issues': int(form_data['reported_issues']),
            'Vehicle_Age': int(form_data['vehicle_age']),
            'Fuel_Type': form_data['fuel_type'],
            'Transmission_Type': form_data['transmission_type'],
            'Engine_Size': int(form_data['engine_size']),
            'Odometer_Reading': float(form_data['odometer_reading']),
            'Fuel_Efficiency': float(form_data['fuel_efficiency']),
            'Tire_Condition': form_data['tire_condition'],
            'Brake_Condition': form_data['brake_condition'],
            'Battery_Status': form_data['battery_status']
        }
        
        # Make prediction
        prediction, probabilities = predict_maintenance(input_data, model, encoders, scaler)
        
        prediction_result = 'Maintenance Needed' if prediction == 1 else 'No Maintenance Needed'
        confidence = round(max(probabilities) * 100, 2)
        
        # Log the prediction
        log_prediction(input_data, prediction_result, confidence)
        
        result = {
            'prediction': prediction_result,
            'confidence': confidence,
            'details': f"Probability: {probabilities[1]*100:.2f}% for Maintenance, {probabilities[0]*100:.2f}% for No Maintenance",
            'model_info': model_info,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'})

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """API endpoint to get analytics data"""
    try:
        visualization_data = prepare_visualization_data()
        prediction_analytics = get_prediction_analytics()
        
        return jsonify({
            'success': True,
            'dataset_analytics': visualization_data,
            'prediction_analytics': prediction_analytics,
            'model_status': model_info
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dataset/stats', methods=['GET'])
def get_dataset_stats():
    """API endpoint to get dataset statistics"""
    try:
        stats = get_dataset_statistics()
        sample_data = get_sample_data(10)
        feature_analysis = get_feature_analysis()
        
        return jsonify({
            'success': True,
            'dataset_stats': stats,
            'sample_data': sample_data,
            'feature_analysis': feature_analysis
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dataset/sample', methods=['GET'])
def get_dataset_sample():
    """API endpoint to get sample dataset rows"""
    try:
        rows = request.args.get('rows', 10, type=int)
        sample_data = get_sample_data(rows)
        
        return jsonify({
            'success': True,
            'sample_data': sample_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/model/status', methods=['GET'])
def get_model_status():
    """API endpoint to check model status"""
    return jsonify({
        'model_loaded': model is not None,
        'model_info': model_info,
        'model_initialized': model_initialized
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("Starting Vehicle Maintenance Predictor Server...")
    print(f"Model Status: {model_info}")
    print("Server running on http://0.0.0.0:5000")
    
    initialize_model()
    app.run(debug=True, host='0.0.0.0', port=5000)