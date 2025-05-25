from flask import Flask, request, render_template
import pandas as pd
import mlflow.sklearn
import requests

# Initialize Flask app
app = Flask(__name__)

# Set MLflow tracking
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("Rainfall2")

# Load production model
production_model_name = "rainfall-prediction-production"
prod_model_uri = f"models:/{production_model_name}@champion"
loaded_model = mlflow.sklearn.load_model(prod_model_uri)

# Feature names expected by model
feature_names = ['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

# Replace with your actual OpenWeatherMap API key
API_KEY = "98884ad9e8ae7136a2b75c4722699663"

# Helper to fetch live weather from OpenWeatherMap
def get_live_weather(city_name):
    base_url ="https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_name,
        "appid": API_KEY,
        "units": "metric"
    }
    response = requests.get(base_url, params=params)
    data = response.json()

    if response.status_code != 200:
        raise Exception(data.get("message", "Failed to fetch weather data"))

    # Extract and approximate necessary fields
    weather_data = {
        'pressure': data['main']['pressure'],
        'dewpoint': data['main']['temp'],  # Approximate dewpoint using temperature
        'humidity': data['main']['humidity'],
        'cloud': data['clouds']['all'],
        'sunshine': 5,  # You can adjust this as needed; API doesn’t provide it directly
        'winddirection': data['wind'].get('deg', 0),
        'windspeed': data['wind']['speed']
    }
    return weather_data

# Home route
@app.route('/')
def home():
    return render_template('index.html', prediction_result=None)

# Manual input prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [
            float(request.form['pressure']),
            float(request.form['dewpoint']),
            float(request.form['humidity']),
            float(request.form['cloud']),
            float(request.form['sunshine']),
            float(request.form['winddirection']),
            float(request.form['windspeed']),
        ]
        input_df = pd.DataFrame([input_data], columns=feature_names)
        prediction = loaded_model.predict(input_df)
        result = "🌧️ Rainfall" if prediction[0] == 1 else "☀️ No Rainfall"
        return render_template('index.html', prediction_result=result)
    except Exception as e:
        return render_template('index.html', prediction_result=f"Error: {str(e)}")

# Real-time weather prediction
@app.route('/realtime', methods=['POST'])
def realtime():
    try:
        city = request.form['city']
        weather = get_live_weather(city)
        input_df = pd.DataFrame([list(weather.values())], columns=feature_names)
        prediction = loaded_model.predict(input_df)
        result = f"Live ({city}): 🌧️ Rainfall" if prediction[0] == 1 else f"Live ({city}): ☀️ No Rainfall"
        return render_template('index.html', prediction_result=result)
    except Exception as e:
        return render_template('index.html', prediction_result=f"Error: {str(e)}")

# Run app
if __name__ == '__main__':
    app.run(debug=True, port=8080)
