import sklearn as sk
import joblib
import cloudpickle
from Dependencies import preprocessing_text, Transformer_Representacion_Seleccion 

# with open('Aplicacion\pipelon.joblib', 'rb') as f:
#     modelo = cloudpickle.load(f)

modelo = joblib.load('Aplicacion\pipelon.joblib')

dato ='Mujeres'

# 3. Hacer predicciones
prediccion = modelo.predict(dato)

# 4. Mostrar el resultado
print("Predicción:", prediccion)