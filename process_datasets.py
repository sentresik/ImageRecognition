import os
import numpy as np
from PIL import Image
import gzip
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
            img = img.resize(target_size, Image.ANTIALIAS)
            img_array = np.array(img)
            images.append(img_array)
            if label is not None:
                labels.append(label)
            else:
                labels.append(1 if '1' in directory else 2)
    return np.array(images), np.array(labels)

def load_mnist_images(file_path, label):
    with gzip.open(file_path, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
        images = images.reshape(-1, 28, 28)
        labels = np.full(images.shape[0], label)
    return images, labels

def process_dataset(images, labels):
    # Wyodrêbnianie cech HOG
    hog_features = [hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True) for image in images]
    hog_features = np.array(hog_features)
    
    # Podzia³ na zestawy treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)
    
    # Trenowanie modelu SVM
    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    
    # Ewaluacja modelu
    y_pred = svm_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

# £adowanie danych z poszczególnych zbiorów
assets_images_1, assets_labels_1 = load_images_from_directory('data/assets/1', label=1)
assets_images_2, assets_labels_2 = load_images_from_directory('data/assets/2', label=2)
assets_images = np.concatenate((assets_images_1, assets_images_2), axis=0)
assets_labels = np.concatenate((assets_labels_1, assets_labels_2), axis=0)

chars74k_images_1, chars74k_labels_1 = load_images_from_directory('data/chars74k/1', label=1)
chars74k_images_2, chars74k_labels_2 = load_images_from_directory('data/chars74k/2', label=2)
chars74k_images = np.concatenate((chars74k_images_1, chars74k_images_2), axis=0)
chars74k_labels = np.concatenate((chars74k_labels_1, chars74k_labels_2), axis=0)

mnist_images_1, mnist_labels_1 = load_mnist_images('data/mnist/train-images-idx3-ubyte.gz', 1)
mnist_images_2, mnist_labels_2 = load_mnist_images('data/mnist/train-images-idx3-ubyte.gz', 2)
mnist_images = np.concatenate((mnist_images_1, mnist_images_2), axis=0)
mnist_labels = np.concatenate((mnist_labels_1, mnist_labels_2), axis=0)

# Przetwarzanie i trenowanie modelu dla ka¿dego zbioru
print("Processing assets dataset...")
process_dataset(assets_images, assets_labels)

print("Processing chars74k dataset...")
process_dataset(chars74k_images, chars74k_labels)

print("Processing MNIST dataset...")
process_dataset(mnist_images, mnist_labels)
