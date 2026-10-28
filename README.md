<<<<<<< HEAD
# AI Spam Detection

This project builds and deploys a machine learning spam detector for SMS messages. It uses a TF-IDF vectorizer and a Multinomial Naive Bayes classifier to classify text as `ham` or `spam`. The trained model is saved and then loaded by a Flask web app for live predictions.

## Project overview

The project contains:

- [train_model.py](./train_model.py): loads the SMS dataset, cleans it, trains the classifier, evaluates the model, and saves it to disk.
- [app.py](./app.py): loads the saved model and serves a simple Flask web interface for prediction.
- [templates/index.html](./templates/index.html): the user interface for entering SMS texts and viewing the result.
- [sms+spam+collection/SMSSpamCollection](./sms+spam+collection/SMSSpamCollection): the SMS dataset used to train the model.
- [spam.model.joblib](./spam.model.joblib): the trained model file produced by the training script.

## Features

- SMS spam classification
- TF-IDF feature extraction
- Naive Bayes text classification
- Model evaluation with accuracy, classification report, and confusion matrix
- Web-based prediction interface using Flask

## Tech stack

- Python 3
- pandas
- scikit-learn
- joblib
- Flask

## Project structure

```text
AI-Spam-Detection/
├── app.py
├── train_model.py
├── spam.model.joblib
├── README.md
├── docs/
│   └── PROJECT_DOCUMENTATION.md
├── templates/
│   └── index.html
├── sms+spam+collection/
│   └── SMSSpamCollection
└── venv/
```

## Setup

1. Open a terminal in the project folder.
2. Create a virtual environment if needed:

```bash
python -m venv venv
```

3. Activate the environment:

- Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

- Windows Command Prompt:

```cmd
venv\Scripts\activate.bat
```

4. Install the dependencies:

```bash
pip install pandas scikit-learn joblib flask
```

## Train the model

Run:

```bash
python train_model.py
```

This script:

- loads the SMS dataset
- drops empty or duplicate rows
- filters only `ham` and `spam` labels
- splits the data into train/test sets
- creates a TF-IDF + Naive Bayes pipeline
- trains the model
- prints model performance metrics
- saves the trained model to `spam.model.joblib`

## Run the web app

Start the Flask app:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Enter an SMS-like message and click the analyze button to see whether the app predicts it as `Spam` or `Ham`.

## Model behavior

The app uses the saved model and calculates a confidence score from the model's predicted class probabilities. The score is displayed as a percentage to show how strongly the model believes the prediction.

## Dataset

The model uses the classic SMS Spam Collection dataset. It contains labeled text messages that are used to train a binary classifier. Messages are labeled as:

- `ham`: normal/non-spam messages
- `spam`: unwanted spam messages

## Notes

- This project is a simple demonstration of machine learning for text classification.
- The model is not perfect and should be considered a learning project, not a production-grade spam filter.
- For a production system, you would likely add additional validation, monitoring, retraining, and more comprehensive evaluation.

## License

This project is provided for educational purposes and can be adapted for your own learning or experiments.
=======
# spam_email_detector
>>>>>>> origin/main
