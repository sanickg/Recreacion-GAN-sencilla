"""
config.py

Aqui guardamos todas las configuraciones (hiperparametros) del proyecto
en un solo lugar. Asi si hay que cambiar algo (por ejemplo, cuantas
veces entrena, o el tamano de las imagenes) solo tocar este archivo
y no hay necesidad de buscar en todo el codigo.
"""

import torch

# ---------------------------------------------------------
# DISPOSITIVO: donde se hacen los calculos
# ---------------------------------------------------------
# Esto usara "cpu". Si se quiere conectar a una
# tarjeta grafica con CUDA, esta linea la detecta sola y la usa,
# sin que haya que cambiar nada mas en el resto del codigo.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# DATOS
# ---------------------------------------------------------
IMAGE_SIZE = 28         # Las imagenes de MNIST son de 28x28 pixeles
IMAGE_CHANNELS = 1       # 1 canal = blanco y negro. Una foto a color seria 3 (RGB)
BATCH_SIZE = 64          # Cuantas imagenes le mostramos a la red al mismo tiempo

# ---------------------------------------------------------
# RED NEURONAL
# ---------------------------------------------------------
LATENT_DIM = 64          # Tamano del vector de "ruido" que recibe el Generador (su punto de partida)
HIDDEN_DIM = 128         # Tamano de las capas ocultas (mas grande = mas capacidad, pero mas lento)

# ---------------------------------------------------------
# ENTRENAMIENTO
# ---------------------------------------------------------
NUM_EPOCHS = 30           # Cuantas veces se recorre TODO el dataset. Bajo a proposito por ser CPU.
LEARNING_RATE = 0.0002   # Que tan grandes son los "pasos" al aprender. Valor estandar para GANs.

# ---------------------------------------------------------
# VISUALIZACION
# ---------------------------------------------------------
SAMPLES_TO_SHOW = 16          # Cuantas imagenes falsas generamos para revisar el progreso
OUTPUT_DIR = "resultados"     # Carpeta donde se guardan las imagenes generadas durante el entrenamiento
