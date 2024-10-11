import contractions
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd

def quitarPuntuacion(words):
    new_words = []
    for word in words:
        if word is not None:
            # Adjusted regular expression pattern to exclude colon
            new_word = re.sub(r'[^\w\s:]', '', word)
            if new_word != '':
                new_words.append(new_word)
    return new_words

def aMinuscula(words):
    new_words = []
    for word in words:
        if word is not None:
            new_word = word.lower()
            if new_word != ' ':
                new_words.append(new_word)
    return new_words


def eliminarNumeros(words):
    new_words = []
    for word in words:
        if not contieneNumero(word):
            new_words.append(word)
    return new_words

def contieneNumero(s):
    pattern = re.compile(r'\d')
    return bool(pattern.search(s))


spanish_stopwords = set(stopwords.words('spanish'))
def quitarStopwords(words):
    new_words = []
    for word in words:
        if word is not None:
            if word not in spanish_stopwords:
                new_words.append(word)
    return new_words

def preProcesamiento(words):
    words = aMinuscula(words)
    words = eliminarNumeros(words)
    words = quitarPuntuacion(words)
    words = quitarStopwords(words)
    return words

#Esta version elimina convierte a misnusculas, quita puntuacion, quita stopwords
def preprocessing_text(texto):

    texto['Textos_espanol'] = texto['Textos_espanol'].apply(contractions.fix)
    texto['words'] = texto['Textos_espanol'].apply(word_tokenize)
    texto['words'] = texto['words'].apply(preProcesamiento)
    texto['words'] = texto['words'].apply(lambda x: ' '.join(map(str, x)))
    return texto['words']

class Transformer_Representacion_Seleccion:
    def __init__(self, count_vectorizer):
        self.count_vectorizer = count_vectorizer
        self.palabras = None
        self.palabras_deseadas = None  # Inicialización de las palabras deseadas

    def fit(self, X, y=None):
        # Ajustar el CountVectorizer y obtener las palabras (características)
        X_transformed = self.count_vectorizer.fit_transform(X)
        self.palabras = self.count_vectorizer.get_feature_names_out()

        # Crear un DataFrame temporal para realizar las selecciones de palabras relevantes
        X_todas_palabras = pd.DataFrame(X_transformed.toarray(), columns=self.palabras)
        
        # Seleccionar las palabras relevantes usando un ciclo tradicional
        self.palabras_deseadas = []
        for nombre in X_todas_palabras.columns:
            # Contar cuántas filas tienen valores distintos de 0 para cada palabra
            dato = (X_todas_palabras[nombre] != 0).sum()
            if dato > 1:
                # Si la palabra aparece en más de una fila, la agregamos a la lista
                self.palabras_deseadas.append(nombre)
        
        return self

    def transform(self, X):
        # Transformar los datos usando CountVectorizer
        X_transformed = self.count_vectorizer.transform(X)
        
        # Crear DataFrame con todas las palabras
        X_todas_palabras = pd.DataFrame(X_transformed.toarray(), columns=self.palabras)
        
        # Seleccionar solo las palabras relevantes
        palabras_a_usar = pd.Index(self.palabras_deseadas).intersection(X_todas_palabras.columns)
        
        # Retornar el DataFrame con las palabras relevantes
        return X_todas_palabras[palabras_a_usar].copy()