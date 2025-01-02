from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained ARIMA model once when the app starts
model = joblib.load('arima_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        days_ahead = int(data['days'])
        
        if days_ahead <= 0:
            return jsonify({'error': 'Please enter a positive number of days.'})
        
        forecast = model.forecast(steps=days_ahead)
        forecast_list = forecast.tolist()

        return jsonify({
            'forecast': forecast_list,
            'message': f'Successfully predicted oil prices for {days_ahead} days.'
        })
    except ValueError:
        return jsonify({'error': 'Invalid input. Please enter a valid integer.'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
