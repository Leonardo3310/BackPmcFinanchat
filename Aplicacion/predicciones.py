import sklearn as sk
import joblib
import pandas as pd
from Dependencies import preprocessing_text, Transformer_Representacion_Seleccion 

#Cargar el modelo
modelo = joblib.load("Aplicacion/Modelos/pipelon.joblib")
#modelo_retrain = joblib.load("Aplicacion/Modelos/pipelon-retrain.joblib")

#Funcion para clasificar opiniones (End-point #1)
def clasificacion(opiniones):

    tamano = len(opiniones)
    predicciones = []

    for i in range(tamano):

        opinion = opiniones[i]
        opinion_df = pd.DataFrame({'Textos_espanol': [opinion]})
        clase_predicha  = int(modelo.predict(opinion_df)[0])
        probabilidades = modelo.predict_proba(opinion_df)

        prob_ods3 = str(round(float(probabilidades[0][0])*100,2))
        prob_ods4 = str(round(float(probabilidades[0][1])*100,2))
        prob_ods5 = str(round(float(probabilidades[0][2])*100,2))

        prediccion = {
            "# Opinion": i+1,
            "clase_predicha": clase_predicha,
            "probabilidades": {
                "ODS 3": prob_ods3+'%',
                "ODS 4": prob_ods4+'%',
                "ODS 5": prob_ods5+'%'
            }
        }
        predicciones.append(prediccion)
    return predicciones


#Funcion reentrenar el modelo (End-point #2)