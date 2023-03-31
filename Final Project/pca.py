import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def add_faces_to_dataset(X, y, faces, label):
    '''
    Add our own faces and labels to the dataset

    :param X: features
    :param y: labels
    :param faces: list of face files to add
    :param label: label to add
    :return: revised features and label dataset
    '''
    for face in faces:
        face_data = cv2.imread(face, cv2.IMREAD_GRAYSCALE)
        face_data_flatten = face_data.flatten()

        X = np.vstack((X, face_data_flatten))
        y = np.append(y, label)

    return X, y


def prepare_dataset():
    '''
    Get the features and labels from the olivetti_faces, and adding our own features and lables to the dataset

    :return: the features (x) and labels (y)
    '''
    # Get features and labels from dataset
    data = fetch_olivetti_faces()
    X = data.data
    y = data.target

    # Add my own face to dataset
    faces1 = ['images/selfie_train.jpg', 'images/selfie_train2.jpg', 'images/selfie_train3.jpg']
    faces2 = ['images/selfie_train4.jpg']
    X, y = add_faces_to_dataset(X, y, faces1, 40)
    X, y = add_faces_to_dataset(X, y, faces2, 41)

    return X, y


def try_with_own_faces(pca, svm, x, y):
    '''
    Make a prediction given our own test data

    :param pca: pca model
    :param svm: svm model
    :param x: features
    :param y: labels
    :return: None
    '''

    # Do prediction
    new_image = cv2.imread('images/selfie_test3.jpg', cv2.IMREAD_GRAYSCALE)
    flattened_image = new_image.flatten()
    new_image_pca = pca.transform([flattened_image])
    new_image_label = svm.predict(new_image_pca)[0]

    print('Predicted label:', new_image_label)

    # Show image
    img_index = np.where(y == new_image_label)[0][0]

    if img_index >= 0:
        plt.subplot(1, 2, 1)
        plt.imshow(new_image, cmap='gray')
        plt.title('Inputted Image')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        image = x[img_index]
        plt.imshow(image.reshape(64,64), cmap=plt.cm.gray)
        plt.title('Predicted Image Stored in DB')

        plt.xticks([])
        plt.yticks([])

        plt.show()


def plot_n_components(X_train, X_test, y_train, y_test):
    '''
    Plot the effect that components have on the accuracy of the model

    :param X_train: training features
    :param X_test: testing features
    :param y_train: training labels
    :param y_test: testing labels
    :return: None
    '''
    n_components = [1, 3, 5, 10, 20, 40, 50]
    accuracies = []
    for n in n_components:
        pca = PCA(n_components=n)
        pca.fit(X_train)

        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        svm = SVC(kernel='linear')
        svm.fit(X_train_pca, y_train)

        y_pred = svm.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    plt.title("Components v. Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Components")
    plt.xticks(n_components)
    plt.plot(n_components, accuracies)
    plt.show()


def main():
    # Training model with olivetti dataset and my own selfies
    X, y = prepare_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pca = PCA(n_components=80)
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    svm = SVC(kernel='linear')
    svm.fit(X_train_pca, y_train)

    y_pred = svm.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    plot_n_components(X_train, X_test, y_train, y_test)
    try_with_own_faces(pca, svm, X, y)


main()
