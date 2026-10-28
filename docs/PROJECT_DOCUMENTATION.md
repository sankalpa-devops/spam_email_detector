# Project Documentation

## 1. Overview

This project is an SMS spam detection application built with Python, scikit-learn, and Flask. It trains a text classifier on labeled SMS messages and exposes a web interface that allows users to type a message and get a prediction.

The classifier uses:

- `TfidfVectorizer` for converting text to numerical features
- `MultinomialNB` for classification
- A `Pipeline` to combine preprocessing and model execution

## 2. Purpose

The goal of the project is to:

- demonstrate how machine learning can classify text messages
- teach the workflow from dataset loading to model training and evaluation
- provide a simple interface for predicting whether a message is spam or not

## 3. Main files

### train_model.py

This file handles the training workflow.

Responsibilities:

- locate the dataset file
- read the CSV-style text file with tab-separated columns
- clean missing and duplicate entries
- keep only valid labels (`ham`, `spam`)
- map labels to numeric values
- split data into training and test sets
- train a `Pipeline` with TF-IDF and Naive Bayes
- evaluate the model with accuracy and classification metrics
- save the trained model to `spam.model.joblib`

### app.py

This file loads the saved model and exposes the Flask application.

Responsibilities:

- initialize the Flask app
- load the trained model using `joblib`
- accept a message via POST request
- predict whether it is spam or ham
- compute prediction confidence
- render the result in the HTML template

### templates/index.html

This file contains the frontend markup and CSS used for the prediction form.

It includes:

- a textarea for message input
- a submit button
- a result area showing the prediction
- a confidence percentage
- an error message for empty submissions

## 4. Data pipeline

The training workflow follows this sequence:

1. Read the dataset from `sms+spam+collection/SMSSpamCollection`
2. Remove null values and duplicate messages
3. Filter against valid labels
4. Map `ham` to `0` and `spam` to `1`
5. Split the data using `train_test_split`
6. Build the model pipeline
7. Train the classifier
8. Run evaluation
9. Save the final trained model

## 5. Machine learning details

### Feature extraction

`TfidfVectorizer` converts text into a numerical representation that emphasizes important words while reducing the weight of common words.

### Model

`MultinomialNB` is a probabilistic classifier commonly used for text classification tasks.

### Evaluation

The script reports:

- accuracy
- classification report
- confusion matrix

This helps determine whether the model is performing well and which class is being misclassified more often.

## 6. Running the project

### Train the model

```bash
python train_model.py
```

### Start the web app

```bash
python app.py
```

Then visit:

```text
http://127.0.0.1:5000
```

## 7. Dependency install (recommended)

Use a virtual environment and install the dependencies:

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pandas scikit-learn joblib flask
```

## 8. Example usage

Example input:

```text
Congratulations! You have won a free prize. Click here now.
```

Expected output:

- Prediction: `Spam`
- Confidence: a percentage value

Example input:

```text
Hi, can we meet tomorrow after class?
```

Expected output:

- Prediction: `Ham`
- Confidence: a percentage value

## 9. Limitations

This is a beginner-friendly project and is suitable for learning and demonstration. In a real-world system, you would likely want:

- more datasets and better labeling
- more extensive validation
- hyperparameter tuning
- production-grade deployment setup
- monitoring and security safeguards

## 10. Future improvements

Possible improvements include:

- switching to a more advanced model such as Logistic Regression or Linear SVM
- adding a REST API instead of the Flask page
- saving the model metadata and metrics
- adding unit tests
- making the interface more user-friendly
- allowing batch prediction for multiple messages

## 11. Summary

This project demonstrates a complete pipeline for machine learning-based SMS spam detection, from data cleaning and model training to serving predictions through a simple web interface. It is a strong foundation for learning text classification, Flask integration, and model evaluation.
