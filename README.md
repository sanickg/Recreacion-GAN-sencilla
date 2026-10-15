# GAN sencilla para aprender (MNIST)

Este es un proyecto de ejemplo, hecho para **aprender**, no para tener
el mejor resultado posible. Genera números escritos a mano (como los
del dataset MNIST) a partir de ruido aleatorio.

No se necesita GPU (de moento): está pensado para correr (lento, pero correr) en CPU.

## ¿Qué es una GAN, en una frase?

Dos redes neuronales compiten: una (**Generador**) intenta crear
imágenes falsas cada vez más convincentes, y la otra
(**Discriminador**) intenta detectar cuáles son falsas. Al competir,
ambas mejoran, y al final el Generador queda "entrenado" para crear
imágenes nuevas parecidas a las reales.

## Orden recomendado para leer el código

1. **`config.py`** — Todos los números configurables del proyecto,
   en un solo lugar. Léelo primero para tener contexto de qué
   tamaños/valores se van a usar más adelante.
2. **`dataset.py`** — Cómo se descargan y preparan las imágenes
   reales que le vamos a mostrar a la red.
3. **`models.py`** — Las dos redes neuronales: `Generador` y
   `Discriminador`. Aquí está la "arquitectura" del proyecto.
4. **`utils.py`** — Una función chica para guardar imágenes de
   ejemplo como PNG usando matplotlib, y así ver el progreso.
5. **`train.py`** — El archivo que amarra todo: aquí es donde
   realmente ocurre el entrenamiento. Es el más largo.

## Cómo correrlo 

```bash
pip install -r requirements.txt
python train.py
```

La primera vez va a descargar el dataset MNIST automáticamente
(pesa poco, unos cuantos MB). Después de cada "época" (una pasada
completa por todo el dataset) se va a guardar una imagen en la
carpeta `resultados/` mostrando 16 ejemplos generados, y se puede notar
como mejora con el tiempo.

Con `NUM_EPOCHS = 5` en CPU puede tardar varios minutos — MNIST es
chico, pero una GAN sigue siendo un modelo pesado de entrenar sin GPU.

## Conceptos clave para entender el código

- **Tensor**: la forma en que PyTorch representa números (como un
  array de numpy, pero optimizado para redes neuronales).
- **Ruido / vector latente**: una lista de números aleatorios que le
  damos al Generador como "punto de partida" para crear una imagen.
  Cambiar el ruido cambia la imagen que sale.
- **Época (epoch)**: una pasada completa por todo el dataset.
- **Batch (lote)**: un grupo pequeño de imágenes que se procesan
  juntas, en vez de una por una (es más rápido y estable).
- **Loss (pérdida)**: un número que mide qué tan "mal" lo está
  haciendo la red. Entrenar = ir bajando ese número poco a poco.
- **Backward / gradientes**: el mecanismo con el que PyTorch calcula
  "en qué dirección" hay que ajustar cada número interno de la red
  para que la pérdida baje.
- **Optimizador (Adam)**: el que realmente mueve esos números
  internos, usando los gradientes calculados.

