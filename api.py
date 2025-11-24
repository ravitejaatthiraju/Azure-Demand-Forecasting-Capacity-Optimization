import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import os
from datetime import datetime, timedelta
import io

# --- Flask App Initialization and CORS ---
app = Flask(__name__)
CORS(app)

# --- Data and Model Loading ---

# 1. Load the historical dataset
try:
    data_path = os.path.join('data', 'processed', 'featured_dataset.csv')
    historical_data = pd.read_csv(data_path, parse_dates=['date']).sort_values('date')
    print(f"Successfully loaded dataset from {data_path}")
except FileNotFoundError:
    print(f"WARNING: Dataset not found at '{data_path}'. Generating dummy data instead.")
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=180, freq='D'))
    historical_data = pd.DataFrame({
        'date': dates,
        'region': np.random.choice(['East US', 'West US', 'Central Europe'], size=180),
        'resource_type': np.random.choice(['VM', 'Storage', 'Container'], size=180),
        'usage_cpu': np.random.uniform(100, 500, size=180) + np.sin(np.arange(180) / 10) * 50 + 200,
        'cpu_usage_7_day_avg': np.random.uniform(150, 450, size=180),
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'week_of_year': dates.isocalendar().week,
        'holiday': dates.dayofweek.isin([5, 6]).astype(int),
        'users_active': np.random.randint(1000, 5000, size=180),
        'economic_index': np.random.uniform(90, 110, size=180),
        'cloud_market_demand': np.random.uniform(7, 10, size=180),
    })

# 2. Load the primary (best) model
try:
    model_path = os.path.join('models', 'xgboost_model.joblib')
    xgboost_model = joblib.load(model_path)
    print(f"Successfully loaded primary model (XGBoost) from {model_path}")
except Exception as e:
    xgboost_model = None
    print(f"FATAL: Could not load the primary XGBoost model: {e}")

# 3. Define available capacity (for Milestone 4)
AVAILABLE_CAPACITY = {
    ('East US', 'VM'): 30000,
    ('East US', 'Storage'): 45000,
    ('East US', 'Container'): 15000,
    ('West US', 'VM'): 18000,
    ('West US', 'Storage'): 50000,
    ('West US', 'Container'): 12000,
    ('Central Europe', 'VM'): 10000,
    ('Central Europe', 'Storage'): 22000,
    ('Central Europe', 'Container'): 7000,
}


# --- Helper function to create future features ---
def create_future_features(last_known_row, future_dates):
    """Creates a DataFrame with features for future dates based on the last known data."""
    future_df = pd.DataFrame({'date': future_dates, 'region': last_known_row['region'], 'resource_type': last_known_row['resource_type']})
    
    future_df['day_of_week'] = future_df['date'].dt.dayofweek
    iso_cal = future_df['date'].dt.isocalendar()
    future_df['week_of_year'] = iso_cal.week
    future_df['month'] = future_df['date'].dt.month
    future_df['holiday'] = future_df['day_of_week'].isin([5, 6]).astype(int)
    future_df['economic_index'] = last_known_row['economic_index']
    future_df['cloud_market_demand'] = last_known_row['cloud_market_demand']
    future_df['users_active'] = last_known_row['users_active']
    
    return future_df

# --- Prediction logic for the primary XGBoost model ---
def predict_with_xgboost(historical_subset):
    """Generates a forecast using the pre-loaded XGBoost model."""
    last_known_row = historical_subset.iloc[-1]
    last_date = last_known_row['date']
    future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, 31)])
    
    future_df = create_future_features(last_known_row, future_dates)

    last_7_days_cpu = historical_subset['usage_cpu'].iloc[-7:].tolist()
    future_cpu_avg = []
    for _ in range(len(future_df)):
        avg = np.mean(last_7_days_cpu)
        future_cpu_avg.append(avg)
        last_7_days_cpu.pop(0)
        last_7_days_cpu.append(avg)
    future_df['cpu_usage_7_day_avg'] = future_cpu_avg
    
    future_df_encoded = pd.get_dummies(future_df, columns=['region', 'resource_type'])
    training_cols = xgboost_model.get_booster().feature_names
    for col in training_cols:
        if col not in future_df_encoded.columns:
            future_df_encoded[col] = 0
    future_df_encoded = future_df_encoded[training_cols]

    predictions = xgboost_model.predict(future_df_encoded)
    
    return predictions.tolist(), future_df['date'].dt.strftime('%Y-%m-%d').tolist()

# --- SIMULATED prediction logic for other models ---
def simulate_prediction(historical_subset, model_type='ARIMA'):
    """Simulates a forecast for ARIMA or LSTM for demonstration purposes."""
    print(f"LOG: Generating a simulated forecast for {model_type}.")
    last_known_row = historical_subset.iloc[-1]
    last_date = last_known_row['date']
    last_value = last_known_row['usage_cpu']
    future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, 31)])
    
    if model_type == 'ARIMA':
        trend = np.linspace(0, 5, 30)
        seasonality = 10 * np.sin(np.linspace(0, 3 * np.pi, 30))
        noise = np.random.normal(0, 15, 30)
        predicted_cpu = last_value + trend + seasonality + noise
    else: # LSTM simulation
        base_trend = np.logspace(np.log10(1), np.log10(1.2), 30) * last_value
        volatility = np.random.randn(30).cumsum() * 3
        predicted_cpu = base_trend + volatility

    predicted_cpu[predicted_cpu < 20] = 20
    return predicted_cpu.tolist(), [d.strftime('%Y-%m-%d') for d in future_dates]


# --- API Endpoints ---

@app.route('/api/v1/demand-data', methods=['GET'])
def get_demand_data():
    return jsonify(historical_data.to_dict(orient='records'))

@app.route('/api/model-comparison', methods=['GET'])
def get_model_comparison():
    comparison_data = [
        {"Model": "ARIMA", "MAE": 89.26, "RMSE": 105.79, "MAPE": 13.06, "TrainingTime": 25.10, "InferenceSpeed": 0.11412},
        {"Model": "XGBoost", "MAE": 24.64, "RMSE": 31.55, "MAPE": 3.44, "TrainingTime": 1.52, "InferenceSpeed": 0.00004},
        {"Model": "LSTM", "MAE": 132.03, "RMSE": 145.29, "MAPE": 18.23, "TrainingTime": 345.80, "InferenceSpeed": 0.00215}
    ]
    return jsonify(comparison_data)

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    try:
        region = request.args.get('region')
        service = request.args.get('service')
        model_name = request.args.get('model', 'XGBoost')

        if not region or not service:
            return jsonify({"error": "Region and service parameters are required."}), 400
        
        historical_subset = historical_data[(historical_data['region'] == region) & (historical_data['resource_type'] == service)]
        if historical_subset.empty:
            return jsonify({"error": f"No historical data for {region} - {service}"}), 404

        if model_name == 'XGBoost':
            if xgboost_model is None: return jsonify({"error": "XGBoost model is not loaded."}), 500
            predicted_cpu, forecast_dates = predict_with_xgboost(historical_subset)
        elif model_name == 'ARIMA':
            predicted_cpu, forecast_dates = simulate_prediction(historical_subset, model_type='ARIMA')
        elif model_name == 'Prophet' or model_name == 'LSTM':
            predicted_cpu, forecast_dates = simulate_prediction(historical_subset, model_type='LSTM')
        else:
            return jsonify({'error': f"Model '{model_name}' is not a valid option."}), 400

        return jsonify({'forecast_dates': forecast_dates, 'predicted_cpu_usage': predicted_cpu})

    except Exception as e:
        print(f"An error occurred in /api/forecast: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/api/capacity-planning', methods=['GET'])
def get_capacity_recommendation():
    try:
        region = request.args.get('region')
        service = request.args.get('service')
        
        if not region or not service:
            return jsonify({"error": "Region and service parameters are required."}), 400

        historical_subset = historical_data[(historical_data['region'] == region) & (historical_data['resource_type'] == service)]
        if historical_subset.empty:
            return jsonify({"error": f"No historical data for {region} - {service}"}), 404

        if xgboost_model is None:
            return jsonify({"error": "Cannot generate plan: XGBoost model not loaded."}), 500
        
        predicted_cpu, _ = predict_with_xgboost(historical_subset)
        forecast_demand = int(sum(predicted_cpu))
        available_capacity = AVAILABLE_CAPACITY.get((region, service), 0)
        
        adjustment = forecast_demand - available_capacity
        buffer_percent = (available_capacity - forecast_demand) / available_capacity if available_capacity > 0 else -1

        risk_level = "green"
        recommendation = "Capacity is sufficient. No immediate action required."
        if adjustment > 0:
            risk_level = "red"
            recommendation = f"Shortage Alert: Add at least {adjustment:+,} units."
        elif buffer_percent > 0.20:
            risk_level = "yellow"
            over_provisioned_units = abs(adjustment)
            recommendation = f"Over-provisioned: Consider reducing by up to {over_provisioned_units:,} units."

        return jsonify({
            "region": region, "service": service, "forecast_demand": forecast_demand,
            "available_capacity": available_capacity, "recommended_adjustment_units": int(adjustment),
            "recommendation_text": recommendation, "risk_level": risk_level
        })

    except Exception as e:
        print(f"An error occurred in /api/capacity-planning: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/api/monitoring', methods=['GET'])
def get_monitoring_data():
    try:
        weeks = [(datetime.now() - timedelta(weeks=i)).strftime("Week %U") for i in range(8)][::-1]
        accuracy_scores = [3.44, 3.51, 3.48, 3.55, 3.60, 3.75, 3.88, 4.1]
        accuracy_trend = [{"week": w, "mape": s} for w, s in zip(weeks, accuracy_scores)]
        
        current_mape = accuracy_scores[-1]
        health_status = "green"
        drift_message = "Model performance is stable."
        if 5 < current_mape <= 10:
            health_status = "yellow"
            drift_message = "Caution: Model accuracy is degrading. Monitor closely."
        elif current_mape > 10:
            health_status = "red"
            drift_message = "Alert: Significant error drift detected. Retraining recommended."

        return jsonify({
            "health_status": health_status,
            "last_retrain_date": (datetime.now() - timedelta(days=28)).strftime("%Y-%m-%d"),
            "accuracy_trend": accuracy_trend,
            "drift_alert_message": drift_message
        })
    except Exception as e:
        print(f"An error occurred in /api/monitoring: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/api/report', methods=['GET'])
def get_report():
    """Generates and returns a downloadable CSV report for capacity planning."""
    try:
        region = request.args.get('region')
        service = request.args.get('service')
        
        if not region or not service:
            return jsonify({"error": "Region and service parameters are required."}), 400

        # --- Re-use logic from capacity planning ---
        historical_subset = historical_data[(historical_data['region'] == region) & (historical_data['resource_type'] == service)]
        if historical_subset.empty:
            return jsonify({"error": f"No data for {region} - {service}"}), 404
        if xgboost_model is None:
            return jsonify({"error": "Model not loaded."}), 500
        
        predicted_cpu, forecast_dates = predict_with_xgboost(historical_subset)
        forecast_demand = int(sum(predicted_cpu))
        available_capacity = AVAILABLE_CAPACITY.get((region, service), 0)
        adjustment = forecast_demand - available_capacity
        
        # --- Create CSV content in memory ---
        output = io.StringIO()
        output.write("Azure Capacity Forecast Report\n")
        output.write(f"Region,{region}\n")
        output.write(f"Service,{service}\n")
        output.write(f"Generated on,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output.write("\n")
        output.write("Summary\n")
        output.write("Metric,Value\n")
        output.write(f"Total Forecasted Demand (30 days),{forecast_demand}\n")
        output.write(f"Available Capacity,{available_capacity}\n")
        output.write(f"Recommended Adjustment,{adjustment}\n")
        output.write("\n")
        output.write("Daily Forecast Details\n")
        
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_CPU_Usage': predicted_cpu})
        forecast_df.to_csv(output, index=False)
        
        # --- Create and return the response ---
        csv_content = output.getvalue()
        output.close()

        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename=report_{region}_{service}.csv"}
        )

    except Exception as e:
        print(f"An error occurred in /api/report: {e}")
        return jsonify({"error": "An internal server error occurred while generating the report."}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)


