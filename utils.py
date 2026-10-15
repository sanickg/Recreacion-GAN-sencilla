"""
utils.py

Funciones auxiliares. Aqui usamos matplotlib para poder "ver" las
imagenes que va generando la red durante el entrenamiento.

matplotlib es una libreria para hacer graficas y dibujar datos. Aqui
no la usamos para graficar numeros en un eje X/Y, sino para dibujar
las imagenes (matrices de pixeles) que produce el generador, y
guardarlas como archivo .png para que las puedas revisar despues sin
tener que correr nada.
"""

import os
import matplotlib.pyplot as plt
import torch

import config


def guardar_imagenes_generadas(generador, epoca, ruido_fijo):
    """
    Genera un pequeno lote de imagenes falsas y las guarda como una
    sola imagen tipo "cuadricula" (grid), para ver como va mejorando
    el generador a lo largo del entrenamiento.

    generador: la red generadora (en el punto de entrenamiento actual)
    epoca: numero de epoca actual, solo para nombrar el archivo
    ruido_fijo: usamos SIEMPRE el mismo ruido de entrada cada vez que
                revisamos el progreso, para poder comparar de forma
                justa como evoluciona la MISMA "semilla" de imagen a
                lo largo del entrenamiento.
    """

    # Creamos la carpeta de resultados si no existe todavia
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # torch.no_grad() le dice a PyTorch "no calcules gradientes aqui".
    # Los gradientes solo hacen falta para APRENDER; aqui solo estamos
    # usando la red para generar imagenes, no entrenandola.
    with torch.no_grad():
        imagenes_falsas = generador(ruido_fijo)

    # Las imagenes estan normalizadas entre -1 y 1 (ver dataset.py).
    # Las regresamos al rango 0-1 para poder dibujarlas correctamente.
    imagenes_falsas = (imagenes_falsas + 1) / 2

    # Armamos una figura de matplotlib con varias imagenes en cuadricula
    filas = 4
    columnas = config.SAMPLES_TO_SHOW // filas

    fig, ejes = plt.subplots(filas, columnas, figsize=(columnas, filas))

    for i, eje in enumerate(ejes.flatten()):
        # .squeeze() quita la dimension del canal (1) para graficar bien
        # .cpu() asegura que la imagen este en memoria "normal"
        # .numpy() la convierte a un arreglo que matplotlib entiende
        imagen = imagenes_falsas[i].squeeze().cpu().numpy()
        eje.imshow(imagen, cmap="gray")  # cmap="gray" = dibujar en escala de grises
        eje.axis("off")  # quitamos los numeritos de los ejes, no aportan nada aqui

    plt.tight_layout()

    ruta_archivo = os.path.join(config.OUTPUT_DIR, f"epoca_{epoca:03d}.png")
    plt.savefig(ruta_archivo)
    plt.close(fig)  # cerramos la figura para liberar memoria

    print(f"  Imagenes de ejemplo guardadas en: {ruta_archivo}")
