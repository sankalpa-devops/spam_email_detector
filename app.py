import os
from flask import Flask, render_template, request, jsonify
import joblib
import csv
from datetime import datetime, timezone
from threading import Lock


app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(base_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

log_path = os.path.join(logs_dir, "predictions.csv")
if not os.path.exists(log_path):
    with open(log_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "message_length", "prediction", "confidence"])

log_lock = Lock()


def log_prediction(message, prediction, confidence):
    with log_lock:
        file_exists = os.path.exists(log_path)

        with open(log_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if not file_exists or os.path.getsize(log_path) == 0:
                writer.writerow([
                    "timestamp",
                    "message_length",
                    "prediction",
                    "confidence"
                ])

            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                len(message),
                prediction,
                confidence
            ])

# Load the trained model from the same folder as this file
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'spam.model.joblib')
model = joblib.load(model_path)


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    message = ""
    confidence = None
    error = None

    if request.method == 'POST':
        message = request.form.get('message', '').strip()
        if not message:
            error = 'Please enter a message.'
        else:
            result = int(model.predict([message])[0])
            prediction = 'Spam' if result == 1 else 'Ham'

            probabilities = model.predict_proba([message])[0]
            confidence = round(max(probabilities) * 100, 2)

    return render_template('index.html', prediction=prediction, message=message,
                           confidence=confidence, error=error)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({'error': 'Send a JSON object with a message.'}), 400

    message = data.get('message')

    if not isinstance(message, str) or not message.strip():
        return jsonify({'error': 'Please provide a non-empty message.'}), 400

    message = message.strip()

    result = int(model.predict([message])[0])
    probabilities = model.predict_proba([message])[0]

    prediction = 'Spam' if result == 1 else 'Ham'
    confidence = round(float(max(probabilities)) * 100, 2)

    # Log the prediction to CSV 
    log_prediction(message, prediction, confidence)

    return jsonify({
        'prediction': prediction,
        'confidence': confidence
    })


@app.route('/about')
def health():
    return {'status': 'healthy'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)