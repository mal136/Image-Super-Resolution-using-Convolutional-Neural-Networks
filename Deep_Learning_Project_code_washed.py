import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import os


def load_images(hr_dir, lr_dir, img_size=(128, 128)):
    high_res_images = []
    low_res_images = []
    hr_filenames = sorted(os.listdir(hr_dir))  
    lr_filenames = sorted(os.listdir(lr_dir))  
    
    for hr_filename in hr_filenames:
        base_filename = os.path.splitext(hr_filename)[0]  
        lr_filename = f"{base_filename}x4.png"

        if lr_filename in lr_filenames:
            hr_img = img_to_array(load_img(os.path.join(hr_dir, hr_filename), target_size=(img_size[0]*2, img_size[1]*2))) / 255.0
            high_res_images.append(hr_img)
            lr_img = img_to_array(load_img(os.path.join(lr_dir, lr_filename), target_size=img_size)) / 255.0
            low_res_images.append(lr_img)
        else:
            print(f"Warning: No corresponding LR file for {hr_filename}")

    return np.array(low_res_images), np.array(high_res_images)




hr_dir = "INSERT YOUR PATH HERE"
lr_dir = "INSERT YOUR PATH HERE"


low_res_images, high_res_images = load_images(hr_dir, lr_dir)

def build_model(input_shape=(128, 128, 3)):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])
    return model


model = build_model()
model.summary()

history = model.fit(low_res_images, high_res_images, epochs=50, batch_size=16, validation_split=0.2)

def plot_results(low_res_img, high_res_img, predicted_img):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Low Resolution")
    plt.imshow(low_res_img)
    plt.axis('off')
    ##
    plt.subplot(1, 3, 2)
    plt.title("High Resolution")
    plt.imshow(high_res_img)
    plt.axis('off')
    ###
    plt.subplot(1, 3, 3)
    plt.title("Predicted Resolution")
    plt.imshow(predicted_img)
    plt.axis('off')
    ####
    plt.show()


test_img = low_res_images[0] 
predicted_img = model.predict(np.expand_dims(test_img, axis=0))[0]

plot_results(test_img, high_res_images[0], predicted_img)