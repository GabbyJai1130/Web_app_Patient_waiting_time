# Web_app_Patient_waiting_time

This is a Flask application that allows patients at the emergency department of hospital to get an estimated waiting time before receving treatment. 

The model is based on XGBoost regression algorithm where it takes the age of patient, his queuing number (i.e. number of patients before him) and his triage level; and return an predicted wait time.

## Files
''data_preprocessing.py`` python script that transforms the dataset for model training

``scaler.pkl`` pickle file of the scaler, which normalizes the input values

``xgb_model.py`` XGB model that predicts the estimated waiting time

``app.py`` the deployable flask application

