import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model

IMG_WIDTH, IMG_HEIGHT = 128, 128

test_dir = r'sliwki_test'

# przgotowanie danych testowych
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=16,
    class_mode=None,
    shuffle=False)

model = load_model('model_sliwki.tf')


def choose_file(directory):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir=directory)
    return file_path


file_path = choose_file(test_dir)
if file_path:
    img = image.load_img(file_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    prediction = model.predict(img_tensor)
    print(f'File: {os.path.basename(file_path)}, Prediction: {"dobre" if prediction[0][0] > 0.5 else "zepsute"}')
