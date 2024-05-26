import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    model = load_model('prediction_moviesgen.h5')
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
