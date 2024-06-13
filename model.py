# -*- coding: utf-8 -*-

import os
import numpy as np
import evaluation_visualization
from PIL import Image
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

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
    report = classification_report(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision_1 = precision_score(y_test, y_pred, pos_label=1)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    f1_1 = f1_score(y_test, y_pred, pos_label=1)

    precision_2 = precision_score(y_test, y_pred, pos_label=2)
    recall_2 = recall_score(y_test, y_pred, pos_label=2)
    f1_2 = f1_score(y_test, y_pred, pos_label=2)

    avg_precision = (precision_1 + precision_2) / 2
    avg_recall = (recall_1 + recall_2) / 2
    avg_f1 = (f1_1 + f1_2) / 2

    print("Precision:", avg_precision)
    print("Recall:", avg_recall)
    print("F1:", avg_f1)
    print("Accuracy:", accuracy)
    return {'report': report, 'accuracy': accuracy, 'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1}

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
assets_results = process_dataset(assets_images, assets_labels)

print("Przetwarzanie zbioru chars74k...")
chars74k_results = process_dataset(chars74k_images, chars74k_labels)

print("Przetwarzanie zbioru MNIST...")
mnist_results = process_dataset(mnist_images, mnist_labels)

# Łączenie obrazów i etykiet ze wszystkich zbiorów
combined_images = np.concatenate((assets_images, chars74k_images, mnist_images), axis=0)
combined_labels = np.concatenate((assets_labels, chars74k_labels, mnist_labels), axis=0)

# Przetwarzanie i trenowanie modelu dla połączonych zbiorów
print("Przetwarzanie wszystkich zbiorów...")
combined_results = process_dataset(combined_images, combined_labels)

# Przygotowanie danych do wizualizacji
results = [
    {'accuracy': assets_results['accuracy'], 'precision': assets_results['precision'], 'recall': assets_results['recall'], 'f1-score': assets_results['f1'], 'dataset': 'Assets'},
    {'accuracy': chars74k_results['accuracy'], 'precision': chars74k_results['precision'], 'recall': chars74k_results['recall'], 'f1-score': chars74k_results['f1'], 'dataset': 'Chars74k'},
    {'accuracy': mnist_results['accuracy'], 'precision': mnist_results['precision'], 'recall': mnist_results['recall'], 'f1-score': mnist_results['f1'], 'dataset': 'MNIST'},
    {'accuracy': combined_results['accuracy'], 'precision': combined_results['precision'], 'recall': combined_results['recall'], 'f1-score': combined_results['f1'], 'dataset': 'Combined'}
]

# Wizualizacja wyników ewaluacji modelu
evaluation_visualization.plot_evaluation_results(results)