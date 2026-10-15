"""
train.py

Este es el archivo principal: aqui es donde se entrena la
GAN. Correr este archivo (python train.py) inicia todo el proceso.

RESUMEN DE LA IDEA (leelo antes de ver el codigo):

En cada paso del entrenamiento pasan 2 cosas, una tras otra:

1) Entrenamos al DISCRIMINADOR:
   - Le mostramos imagenes REALES del dataset -> le decimos "esto es real (1)"
   - Le mostramos imagenes FALSAS recien creadas por el generador -> le
     decimos "esto es falso (0)"
   - Ajustamos al discriminador para que se equivoque menos la
     proxima vez.

2) Entrenamos al GENERADOR:
   - Le pedimos que genere imagenes falsas
   - Se las pasamos al discriminador
   - Si el discriminador las detecta como falsas, "castigamos" al
     generador (tiene que mejorar)
   - Ajustamos al generador para que la proxima vez engane mejor al
     discriminador.

Repetimos esto miles de veces, y poco a poco el generador aprende a
crear imagenes que se parecen cada vez mas a numeros reales escritos
a mano.
"""

import torch
import torch.nn as nn
from tqdm import tqdm  # tqdm dibuja la barrita de progreso en la terminal

import config
from dataset import obtener_dataloader
from models import Generador, Discriminador
from utils import guardar_imagenes_generadas


def entrenar():
    print(f"Usando dispositivo: {config.DEVICE}")

    # 1) Preparamos los datos
    dataloader = obtener_dataloader()

    # 2) Creamos las dos redes y las mandamos al dispositivo (CPU aqui)
    generador = Generador().to(config.DEVICE)
    discriminador = Discriminador().to(config.DEVICE)

    # 3) Funcion de perdida (loss): mide que tan "equivocada" esta la red.
    # BCELoss (Binary Cross Entropy) es la funcion estandar cuando la
    # respuesta es "si o no" / "real o falso" (0 o 1).
    funcion_perdida = nn.BCELoss()

    # 4) Optimizadores: son los que ajustan los "pesos" internos de la
    # red para que aprenda. Adam es uno de los optimizadores mas
    # usados en deep learning.
    optimizador_generador = torch.optim.Adam(
        generador.parameters(), lr=config.LEARNING_RATE
    )
    optimizador_discriminador = torch.optim.Adam(
        discriminador.parameters(), lr=config.LEARNING_RATE
    )

    # Ruido fijo que usamos SIEMPRE para revisar el progreso visualmente
    # (ver utils.py para el porque)
    ruido_fijo = torch.randn(config.SAMPLES_TO_SHOW, config.LATENT_DIM, device=config.DEVICE)

    # ------------------------------------------------------------
    # LOOP DE ENTRENAMIENTO
    # ------------------------------------------------------------
    for epoca in range(config.NUM_EPOCHS):
        print(f"\nEpoca {epoca + 1}/{config.NUM_EPOCHS}")

        # tqdm envuelve el dataloader solo para mostrar la barra de progreso
        for imagenes_reales, _ in tqdm(dataloader):
            # El "_" es porque MNIST tambien trae la etiqueta (que numero
            # es, del 0 al 9), pero a esta GAN NO le importa que numero
            # es, solo si la imagen parece real o falsa. Por eso la
            # ignoramos con "_".

            imagenes_reales = imagenes_reales.to(config.DEVICE)
            tamano_lote = imagenes_reales.size(0)  # cuantas imagenes hay en este lote

            # Etiquetas: 1 = real, 0 = falso
            etiquetas_reales = torch.ones(tamano_lote, 1, device=config.DEVICE)
            etiquetas_falsas = torch.zeros(tamano_lote, 1, device=config.DEVICE)

            # ==========================================================
            # PASO 1: Entrenar al DISCRIMINADOR
            # ==========================================================
            optimizador_discriminador.zero_grad()  # limpiamos gradientes de la vuelta anterior

            # -- Con imagenes reales --
            prediccion_reales = discriminador(imagenes_reales)
            perdida_reales = funcion_perdida(prediccion_reales, etiquetas_reales)

            # -- Con imagenes falsas --
            ruido = torch.randn(tamano_lote, config.LATENT_DIM, device=config.DEVICE)
            imagenes_falsas = generador(ruido)

            # .detach() le dice a PyTorch "no calcules gradientes hacia
            # el generador en este paso", porque aqui SOLO estamos
            # entrenando al discriminador.
            prediccion_falsas = discriminador(imagenes_falsas.detach())
            perdida_falsas = funcion_perdida(prediccion_falsas, etiquetas_falsas)

            # Sumamos ambas perdidas y actualizamos al discriminador
            perdida_discriminador = perdida_reales + perdida_falsas
            perdida_discriminador.backward()  # calcula los gradientes
            optimizador_discriminador.step()  # ajusta los pesos de la red

            # ==========================================================
            # PASO 2: Entrenar al GENERADOR
            # ==========================================================
            optimizador_generador.zero_grad()

            # Le volvemos a pasar las imagenes falsas al discriminador,
            # pero ahora SIN .detach(), porque esta vez si queremos que
            # el gradiente fluya hacia el generador.
            prediccion_falsas_2 = discriminador(imagenes_falsas)

            # Aqui esta el truco central de las GANs: le decimos al
            # generador que "la respuesta correcta" para sus imagenes
            # falsas era "real" (etiquetas_reales). Asi, la perdida es
            # alta cuando el discriminador SI detecto que eran falsas,
            # y el generador aprende a evitar eso la proxima vez.
            perdida_generador = funcion_perdida(prediccion_falsas_2, etiquetas_reales)
            perdida_generador.backward()
            optimizador_generador.step()

        # Al final de cada epoca mostramos las perdidas y guardamos
        # unas imagenes de ejemplo para ver el progreso
        print(f"  Perdida Discriminador: {perdida_discriminador.item():.4f} | "
              f"Perdida Generador: {perdida_generador.item():.4f}")

        guardar_imagenes_generadas(generador, epoca + 1, ruido_fijo)

    print("\nEntrenamiento terminado. Revisa la carpeta 'resultados' para ver como fue mejorando el generador.")


if __name__ == "__main__":
    # Esto solo corre si ejecutas este archivo directamente
    # (python train.py), no si lo importas desde otro archivo.
    entrenar()
