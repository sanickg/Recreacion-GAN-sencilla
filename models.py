"""
models.py

Aqui viven las dos redes neuronales que compiten entre si: el
GENERADOR y el DISCRIMINADOR. Esa competencia es la idea central de
una GAN (Generative Adversarial Network = Red Generativa Adversaria).

- El GENERADOR es un falsificador: recibe puro ruido aleatorio y
  trata de convertirlo en una imagen que parezca real.
- El DISCRIMINADOR es un detective: recibe una imagen (real o
  falsa) y trata de adivinar si es real o si la hizo el generador.

Ambos entrenan al mismo tiempo, y mientras compiten, los dos van
mejorando: el generador crea falsificaciones cada vez mas
convincentes, y el discriminador se vuelve cada vez mejor detective.
Al final nos quedamos con el generador, que ya aprendio a crear
imagenes nuevas parecidas a las del dataset.
"""

import torch.nn as nn
import config


class Generador(nn.Module):
    """
    Recibe un vector de numeros aleatorios (ruido) y lo transforma
    poco a poco en una imagen de 28x28 pixeles.
    """

    def __init__(self):
        super().__init__()

        # nn.Sequential encadena capas: la salida de una es la entrada
        # de la siguiente, en orden, de arriba hacia abajo.
        self.red = nn.Sequential(
            # Primera capa: de LATENT_DIM numeros a HIDDEN_DIM numeros
            nn.Linear(config.LATENT_DIM, config.HIDDEN_DIM),
            nn.ReLU(),  # activacion: apaga (pone en 0) los valores negativos

            # Segunda capa: agrandamos la capa oculta
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM * 2),
            nn.ReLU(),

            # Capa final: produce exactamente el numero de pixeles de
            # una imagen (28 * 28 = 784, en blanco y negro)
            nn.Linear(config.HIDDEN_DIM * 2, config.IMAGE_SIZE * config.IMAGE_SIZE),

            # Tanh comprime la salida para que quede entre -1 y 1, el
            # MISMO rango en el que normalizamos las imagenes reales
            # en dataset.py. Esto importa: si el generador no produce
            # valores en el mismo rango que los datos reales, el
            # discriminador lo detecta facil.
            nn.Tanh()
        )

    def forward(self, ruido):
        """
        'forward' define que pasa cuando le metes datos a la red.
        ruido: tensor de forma (batch_size, LATENT_DIM)
        """
        imagen_plana = self.red(ruido)  # forma: (batch_size, 784)

        # Reacomodamos los 784 numeros en una imagen de 28x28 con 1 canal
        imagen = imagen_plana.view(-1, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
        return imagen


class Discriminador(nn.Module):
    """
    Recibe una imagen (real o generada) y regresa un solo numero
    entre 0 y 1: que tan "real" cree que es esa imagen.
    (0 = definitivamente falsa, 1 = definitivamente real)
    """

    def __init__(self):
        super().__init__()

        self.red = nn.Sequential(
            # Aplanamos la imagen de 28x28 a un solo vector de 784 numeros
            nn.Flatten(),

            nn.Linear(config.IMAGE_SIZE * config.IMAGE_SIZE, config.HIDDEN_DIM * 2),
            nn.LeakyReLU(0.2),  # como ReLU, pero deja pasar un poco de los valores negativos (funciona mejor en GANs)

            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.LeakyReLU(0.2),

            # Ultima capa: un solo numero de salida
            nn.Linear(config.HIDDEN_DIM, 1),

            # Sigmoid comprime ese numero para que quede entre 0 y 1,
            # como si fuera una probabilidad.
            nn.Sigmoid()
        )

    def forward(self, imagen):
        return self.red(imagen)
