import os
import shutil
import random
import pathlib 

# Directorio de origen y directorios de destino
source_dir = pathlib.Path("imagenesBasura")  
dest_dir = pathlib.Path("ImagenesOrdenadas")
categories = os.listdir(source_dir)

# Proporciones para dividir los datos
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# Asegúrate de que la suma sea 1
assert train_ratio + validation_ratio + test_ratio == 1, "Las proporciones no suman 1."

# Crear carpetas de train, validation y test
for folder in ["train", "validation", "test"]:
    for category in categories:
        os.makedirs(os.path.join(dest_dir, folder, category), exist_ok=True)

# Distribuir los archivos
for category in categories:
    # Lista de imágenes en la categoría
    images = os.listdir(os.path.join(source_dir, category))
    random.shuffle(images)  # Barajar las imágenes para hacer la selección aleatoria

    # Calcular los límites para las divisiones
    train_limit = int(len(images) * train_ratio)
    validation_limit = train_limit + int(len(images) * validation_ratio)

    # Mover imágenes a train, validation y test
    for i, image in enumerate(images):
        src = os.path.join(source_dir, category, image)
        
        if i < train_limit:
            dest = os.path.join(dest_dir, "train", category, image)
        elif i < validation_limit:
            dest = os.path.join(dest_dir, "validation", category, image)
        else:
            dest = os.path.join(dest_dir, "test", category, image)

        shutil.copy2(src, dest)  # Copiar la imagen a la carpeta de destino

print("Organización de imágenes completada.")
