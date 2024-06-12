# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def load_images_from_directory(directory, target_size=(28, 28), label=None):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(directory, filename)).convert('L')
            img = img.resize(target_size, resample=Image.LANCZOS)
            img_array = np.array(img)
            images.append(img_array)
            if label is not None:
                labels.append(label)
            else:
                labels.append(1 if '1' in directory else 2)
    return np.array(images), np.array(labels)

def load_mnist_images(images_file_path, labels_file_path):
    with open(images_file_path, 'rb') as f:
        f.read(16)
        images = np.frombuffer(f.read(), np.uint8)
        images = images.reshape(-1, 28, 28)
    with open(labels_file_path, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(), np.uint8)
    # Filtracja tylko etykiet 1 i 2
    filter_indices = np.where((labels == 1) | (labels == 2))[0]
    images = images[filter_indices]
    labels = labels[filter_indices]
    return images, labels

def process_dataset(images, labels):
    # Wyodrębnianie cech HOG
    hog_features = [hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True) for image in images]
    hog_features = np.array(hog_features)
    # Podział na zestawy treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)
    # Trenowanie modelu SVM
    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    # Ewaluacja modelu
    y_pred = svm_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return X_train, X_test, y_train, y_test, svm_model

# Ładowanie danych z poszczególnych zbiorów
assets_images_1, assets_labels_1 = load_images_from_directory('data/assets/1', label=1)
assets_images_2, assets_labels_2 = load_images_from_directory('data/assets/2', label=2)
assets_images = np.concatenate((assets_images_1, assets_images_2), axis=0)
assets_labels = np.concatenate((assets_labels_1, assets_labels_2), axis=0)

chars74k_images_1, chars74k_labels_1 = load_images_from_directory('data/chars74k/digits updated/1', label=1)
chars74k_images_2, chars74k_labels_2 = load_images_from_directory('data/chars74k/digits updated/2', label=2)
chars74k_images = np.concatenate((chars74k_images_1, chars74k_images_2), axis=0)
chars74k_labels = np.concatenate((chars74k_labels_1, chars74k_labels_2), axis=0)

mnist_images_1, mnist_labels_1 = load_mnist_images('data/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte', 'data/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte')
mnist_images_2, mnist_labels_2 = load_mnist_images('data/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte', 'data/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
mnist_images = np.concatenate((mnist_images_1, mnist_images_2), axis=0)
mnist_labels = np.concatenate((mnist_labels_1, mnist_labels_2), axis=0)

# Przetwarzanie i trenowanie modelu dla każdego zbioru
print("Przetwarzanie zbioru assets...")
process_dataset(assets_images, assets_labels)

print("Przetwarzanie zbioru chars74k...")
process_dataset(chars74k_images, chars74k_labels)

print("Przetwarzanie zbioru MNIST...")
process_dataset(mnist_images, mnist_labels)