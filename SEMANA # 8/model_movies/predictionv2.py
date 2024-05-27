import pandas as pd
import json
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model
from utils import split_into_lemmas

# Descargar recursos necesarios para NLTK
nltk.download('wordnet')

# Entrada de texto
input_text = {
    "year": 2003, 
    "title": "Most", 
    "plot": "most is the story of a single father who takes his eight year - old son to work with him at the railroad drawbridge where he is the bridge tender .  a day before ,  the boy meets a woman boarding a train ,  a drug abuser .  at the bridge ,  the father goes into the engine room ,  and tells his son to stay at the edge of the nearby lake .  a ship comes ,  and the bridge is lifted .  though it is supposed to arrive an hour later ,  the train happens to arrive .  the son sees this ,  and tries to warn his father ,  who is not able to see this .  just as the oncoming train approaches ,  his son falls into the drawbridge gear works while attempting to lower the bridge ,  leaving the father with a horrific choice .  the father then lowers the bridge ,  the gears crushing the boy .  the people in the train are completely oblivious to the fact a boy died trying to save them ,  other than the drug addict woman ,  who happened to look out her train window .  the movie ends ,  with the man wandering a new city ,  and meets the woman ,  no longer a drug addict ,  holding a small baby .  other relevant narratives run in parallel ,  namely one of the female drug - addict ,  and they all meet at the climax of this tumultuous film ."
}

input_text2 = {
    "year": 2008, 
    "title": "How to Be a Serial Killer", 
    "plot": "a serial killer decides to teach the secrets of his satisfying career to a video store clerk ."
}

input_text3 = {
    "year": 1941, 
    "title": "A Woman's Face", 
    "plot": "in sweden ,  a female blackmailer with a disfiguring facial scar meets a gentleman who lives beyond his means .  they become accomplices in blackmail ,  and she falls in love with him ,  bitterly resigned to the impossibility of his returning her affection .  her life changes when one of her victims proves to be the wife of a plastic surgeon ,  who catches her in his apartment ,  but believes her to be a jewel thief rather than a blackmailer .  he offers her the chance to look like a normal woman again ,  and she accepts ,  despite the agony of multiple operations .  meanwhile ,  her gentleman accomplice forms an evil scheme to rid himself of the one person who stands in his way to a fortune  -  his four - year - old - nephew ."
}

input_text4 = {
    "year": 2008, 
    "title": "Ghost Town", 
    "plot": "bertram pincus is a man whose people skills leave much to be desired .  when pincus dies unexpectedly ,  but is miraculously revived after seven minutes ,  he wakes up to discover that he now has the annoying ability to see ghosts .  even worse ,  they all want something from him ,  particularly frank herlihy who pesters him into breaking up the impending marriage of his widow gwen .  that puts pincus squarely in the middle of a triangle with spirited result ."
}

def prepared_input(table, vect):
    plot = table.loc[0, 'plot']
    X_test_dtm = vect.transform([plot]).toarray()
    return X_test_dtm

def predict_genres(indata):
    # Cargar el modelo y el vectorizador entrenado
    modelo = load_model('prediction_moviesgen.h5')
    vect = joblib.load('vectorizer.pkl')
                 
    df = pd.DataFrame([indata])

    # Convertir entrada a formato del modelo
    dataTesting_vect = prepared_input(df, vect)

    # Realizar predicciones en el conjunto de prueba preprocesado
    y_pred = modelo.predict(dataTesting_vect)

    # Guardar predicciones en formato exigido en la competencia de Kaggle
    cols = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
            'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
            'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    
    # Filtrar las columnas cuyo valor sea mayor o igual a 0.50
    res = pd.DataFrame({col: [val] for col, val in zip(cols, y_pred[0]) if val >= 0.5})

    # Convertir DataFrame a JSON
    res_json = res.to_json(orient='records')

    return res_json

def main(indata):
    return predict_genres(indata)

