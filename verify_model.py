import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam.model.joblib")

model = joblib.load(MODEL_PATH)

sample_messages = [
    "Congratulations! You've won a $1000 gift card. Click here to claim your prize.",
    "Hey, are we still on for lunch tomorrow?",]

predictions = model.predict(sample_messages)

for message, prediction in zip(sample_messages, predictions):
    label = "spam" if prediction == 1 else "ham"
    print(f"Message: {message}\nPredicted label: {label}\n")    

assert set(model.classes_) == {0, 1}
print("Model verification successful: Classes are as expected.")