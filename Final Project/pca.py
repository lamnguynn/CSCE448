import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
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
        plt.title('Predicted Image of label ' + str(new_image_label))

        plt.xticks([])
        plt.yticks([])

        plt.show()


def try_with_dataset_faces(pca, svm, x, X_test_pca):
    random_5 = X_test_pca[np.random.choice(len(X_test_pca), size=5, replace=False)]

    for image in random_5:
        new_image_label = svm.predict(image.reshape(1, -1))[0]
        reverted_image = pca.inverse_transform(image).reshape(64,64)

        plt.subplot(1, 2, 1)
        plt.imshow(reverted_image, cmap='gray')
        plt.title('Inputted Image')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        image = x[new_image_label * 10]
        plt.imshow(image.reshape(64, 64), cmap=plt.cm.gray)
        plt.title('Predicted Image of label ' + str(new_image_label))

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
    n_components = [5, 10, 20, 40, 50, 100, 200]
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
    plt.plot(n_components, accuracies, marker='o')
    plt.show()


def plot_eigen_faces(eigenfaces, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)

    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(eigenfaces[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

    plt.show()

def main():
    # Training model with olivetti dataset and my own selfies
    X, y = prepare_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pca = PCA(n_components=100)
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    svm = SVC()
    clf = GridSearchCV(svm, { 'C':[0.1,1,100],'kernel':['rbf','poly','sigmoid','linear'],'degree':[1,2,3,4],'gamma': [1, 0.1, 0.01, 0.001, 0.0001]})
    clf.fit(X_train_pca, y_train)
    print(clf.best_params_)
    print('Accuracy:', clf.score(X_test_pca, y_test))

    n_components = 100
    eigenfaces = pca.components_.reshape((n_components, 64, 64))
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    #plot_eigen_faces(eigenfaces, eigenface_titles, 64, 64)
    #plot_n_components(X_train, X_test, y_train, y_test)

    # Uncomment these to test
    #try_with_own_faces(pca, clf, X, y)
    #try_with_dataset_faces(pca, clf, X, X_test_pca)


main()
