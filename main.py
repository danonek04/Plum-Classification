import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing import image


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

data_dir = r'sliwki'
validation_dir = r'sliwki_walidacja'
test_dir = r'sliwki_test'


# przygotowanie danych treningowych
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=16,
    class_mode='binary')

# przgotowanie danych walidacycjnych
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=16,
    class_mode='binary')

# przgotowanie danych testowych
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=16,
    class_mode=None,  # brak labelek
    shuffle=False)

# trening
model.fit(
    train_generator,
    steps_per_epoch=18,  # 280/batch_size
    epochs=20,
    validation_data=validation_generator,
    validation_steps=4)  # 60/batch_size


for file in os.listdir(test_dir):
    img_path = os.path.join(test_dir, file)
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    prediction = model.predict(img_tensor)
    print(f'File: {file}, Prediction: {"dobre" if prediction[0][0] > 0.5 else "zepsute"}')