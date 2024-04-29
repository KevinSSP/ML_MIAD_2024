import pandas as pd
import json
import joblib
import sys
import os
import xgboost

def predict_price(indata):
    modelo = joblib.load('prediction_car_reg.pkl')
    reg = modelo['modelo']
    columnas = modelo['columnas']
                 
    df = pd.DataFrame([indata])
    
    # Convertir variables categóricas en variables numéricas
    dataTesting_encoded = pd.get_dummies(df, columns=['State', 'Make', 'Model'])

    # Asegurarse de que las columnas del conjunto de prueba coincidan con las del conjunto de entrenamiento
    missing_cols = set(columnas) - set(dataTesting_encoded.columns)
    for col in missing_cols:
        dataTesting_encoded[col] = 0
        
    dataTesting_encoded = dataTesting_encoded[columnas]

    # Realizar predicciones en el conjunto de prueba preprocesado
    y_pred = reg.predict(dataTesting_encoded)

    return y_pred

