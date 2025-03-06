import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog

IMG_WIDTH, IMG_HEIGHT = 128, 128

# CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


data_dir = r'C:\Users\Lenovo\Documents\studia\semestr 5\computer_vision\projekt\sliwki'
validation_dir = r'C:\Users\Lenovo\Documents\studia\semestr 5\computer_vision\projekt\sliwki_walidacja'
test_dir = r'C:\Users\Lenovo\Documents\studia\semestr 5\computer_vision\projekt\sliwki_test'

# przygotowanie danych treningowych
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=16,
    class_mode='binary')

#przgotowanie danych walidacycjnych
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=16,
    class_mode='binary')

#przgotowanie danych testowych
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=16,
    class_mode=None,
    shuffle=False)


model.fit(
    train_generator,
    steps_per_epoch=18,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=4)


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