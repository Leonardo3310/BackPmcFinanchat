import sklearn as sk
import joblib
import pandas as pd
from Dependencies import preprocessing_text, Transformer_Representacion_Seleccion 

#Cargar el modelo
modelo = joblib.load('Aplicacion/Modelos/pipelon.joblib')

#Funcion para clasificar opiniones (End-point #1)
def clasificacion(opiniones):
    tamano = len(opiniones)
    predicciones = []
    for i in range(tamano):
        opinion = opiniones[i]
        opinion_df = pd.DataFrame({'Textos_espanol': [opinion]})
        prediccion = modelo.predict(opinion_df)
        predicciones.append(prediccion)
    return predicciones


#Funcion reentrenar el modelo (End-point #2)