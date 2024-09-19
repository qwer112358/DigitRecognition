import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model('cnn_digit_recognition_32x32.h5')


def prepare_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((32, 32))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array


def predict_digit(image_path):
    return np.argmax(model.predict(prepare_image(image_path)))


def process_images_in_folder(folder_path):
    digit_count = [0] * 10
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            predicted_digit = predict_digit(os.path.join(folder_path, filename))
            digit_count[predicted_digit] += 1
    return digit_count


digit_count = process_images_in_folder('digits')
print(digit_count)
