"""
dataset.py

Este archivo descarga y prepara el dataset que le vamos a dar de
"comer" a la GAN.

Usamos MNIST: un dataset clasico y muy pequeno de numeros escritos a
mano (del 0 al 9), en blanco y negro, de 28x28 pixeles. Es el "hola
mundo" del deep learning con imagenes, no necesita GPU.
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import config


def obtener_dataloader():
    """
    Descarga MNIST (si no está instalado) y lo prepara para entrenar.

    Regresa un DataLoader: un objeto de PyTorch que entrega los datos
    en "lotes" (batches) automaticamente, en vez de
    hacerlo a mano.
    """

    # transforms.Compose encadena varias transformaciones que se le
    # aplican a cada imagen antes de usarla:
    transform = transforms.Compose([
        transforms.ToTensor(),                 # convierte la imagen en un tensor de PyTorch
        transforms.Normalize((0.5,), (0.5,))   # deja los pixeles entre -1 y 1 en vez de 0 y 255
        # Normalizar ayuda a que la red entrene mas facil y estable.
    ])

    # Descarga el dataset automaticamente la primera vez que se corre
    # (se guarda en la carpeta "./data"). Las siguientes veces
    # ya no vuelve a descargar, usa la copia local.
    dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # El DataLoader agarra el dataset y lo entrega en paquetes
    # (batches) de tamano BATCH_SIZE, revolviendolos (shuffle=True)
    # para que la red no aprenda el orden de las imagenes.
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    return dataloader
