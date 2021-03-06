from __future__ import annotations

from matplotlib.pyplot import fill_between
import pandas as pd
import numpy as np
import scipy
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, LinearRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neural_network import MLPClassifier
from time import time

from tensorflow.keras import utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

from time import time
import pandas as pd
import numpy as np


# Также в этом классе будут функции для статистического анализа и построения графиков

# Класс для методов машинного обучения
class WifiLearn:
    def __init__(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.lens = { 'train': x_train.shape[0], 'test': x_test.shape[0] }
        self.results = []
        self.__to_categorical()


    def __to_categorical(self):
        self.types = sorted(self.y_train.unique())
        i = 0
        y_train = self.y_train.copy()
        y_test = self.y_test.copy()
        for t in self.types:
            y_train[y_train == t] = i
            y_test[y_test == t] = i
            i += 1

        self.y_train_cat = utils.to_categorical(y_train, len(self.types))
        self.y_test_cat = utils.to_categorical(y_test, len(self.types))


    def augment(self, part=1):
        pass


    def normalize(self):
        pass


    def shuffle(self, part: int=1):
        pass


    def print(self) -> WifiLearn:
        print(pd.DataFrame(self.results))
        return self
    
    def select_best(self, k=5):
        selector = SelectKBest(f_classif, k=k)
        selector.fit(self.x_train, self.y_train)
        cols = selector.get_support(indices=True)
        return self.x_train.iloc[:,cols]

    
    def fit_classic(self) -> WifiLearn:
        classifiers = {  # You can add your clfs or change params here:
            'RidgeClassifier':                  RidgeClassifier(class_weight='balanced'),
            'QuadraticDiscriminantAnalysis':    QuadraticDiscriminantAnalysis(),
            'PassiveAggressiveClassifier':      PassiveAggressiveClassifier(),
            'Logistic Regression':              LogisticRegression(),
            'K-nearest neighbors':              KNeighborsClassifier(),
            'Gaussian Naive Bayes':             GaussianNB(),
            'Perceptron':                       Perceptron(),
            'Stochastic Gradient Descent':      SGDClassifier(),
            'Random Forest':                    RandomForestClassifier(max_depth=20),
            'sk-learn Neural Net':              MLPClassifier(hidden_layer_sizes=(200, 20)),
            'Ada Boost':                        AdaBoostClassifier(),
            'SVC':                              SVC(),
            'Linear SVC':                       LinearSVC(max_iter=1000)
        }

        classifiers = {
            'Random Forest 1': RandomForestClassifier(min_samples_leaf=5),
            'Logistic Regression':              LogisticRegression(),
            'K-nearest neighbors':              KNeighborsClassifier(),
            # 'Linear SVC':                       LinearSVC(max_iter=1000)
        }
        res = []

        for clf in classifiers:
            start_fit = time()
            classifiers[clf].fit(self.x_train, self.y_train)
            res.append({'name': clf, 'accuracy': round(classifiers[clf].score(self.x_test, self.y_test) * 100, 2),'duration': round(time() - start_fit, 2)})

        self.results += res

        return self


    def fit_classic_sum(self) -> WifiLearn:
        classifiers = {  # You can add your clfs or change params here:
            # 'RidgeClassifier':                  RidgeClassifier(class_weight='balanced'),
            'QuadraticDiscriminantAnalysis':    QuadraticDiscriminantAnalysis(),
            # 'PassiveAggressiveClassifier':      PassiveAggressiveClassifier(),
            # 'Logistic Regression':              LogisticRegression(),
            'K-nearest neighbors':              KNeighborsClassifier(),
            # 'Gaussian Naive Bayes':             GaussianNB(),
            # 'Perceptron':                       Perceptron(),
            # 'Linear SVC':                       LinearSVC(max_iter=1000),
            # 'Stochastic Gradient Descent':      SGDClassifier(),
            'Random Forest':                    RandomForestClassifier(max_depth=20),
            # 'sk-learn Neural Net':              MLPClassifier(hidden_layer_sizes=(200, 20)),
            'Ada Boost':                        AdaBoostClassifier(),
            # 'SVC':                              SVC(),
        }
        data = {

        }

        for clf in classifiers:
            start_fit = time()
            classifiers[clf].fit(self.x_train, self.y_train)
            # res.append({'name': clf, 'accuracy': round(classifiers[clf].score(self.x_test, self.y_test) * 100, 2),'duration': round(time() - start_fit, 2)})
            data[clf] = pd.DataFrame(classifiers[clf].predict_proba(self.x_test), columns=classifiers[clf].classes_)
            print(data[clf])

        
        df = pd.DataFrame(np.zeros(data['Ada Boost'].shape), columns=data['Ada Boost'].columns)
        for clf in data:
            df = df.add(data[clf], fill_value=0)
        
        df = df.idxmax(axis=1)
        print(df)
        print(self.y_test)
        print((df == self.y_test).sum() / df.shape[0])

        return self


    def fit_cnn(self, batch_size: int=50, epochs: int=50) -> WifiLearn:
        self.x_train = np.reshape(self.x_train.to_numpy(), (-1, 4, 56, 1)) / 1 # 400
        self.x_test = np.reshape(self.x_test.to_numpy(), (-1, 4, 56, 1)) / 1

        model = Sequential()
        model.add(Conv2D(28, (3, 3), padding='same',input_shape=(4, 56, 1), activation='relu'))
        model.add(Dropout(0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))
        model.add(Conv2D(12, (3, 3), padding='same', activation='relu'))
        model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(80, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(40, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(len(self.types), activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        start_fit = time()
        model.fit(self.x_train, self.y_train_cat, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1, use_multiprocessing=True)
        clf_res = pd.DataFrame(columns=('method name', 'accuracy', 'time'))
        clf_res.loc[len(clf_res)] = ['CNN', round(model.evaluate(self.x_test, self.y_test_cat, verbose=0)[1] * 100, 2), round(time() - start_fit, 2)]

        print('keras CNN accuracy:', round(model.evaluate(self.x_test, self.y_test_cat, verbose=0)[1] * 100, 2), '-->', round(time() - start_fit, 2))
        print(model.summary())

        return self


    def fit_ffnn(self, batch_size: int=10, epochs: int=10):
        model = Sequential()
        model.add(Dropout(0.5))
        model.add(Dense(360, input_dim=self.x_train.shape[1], activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(36, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.types), activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

        start_fit = time()
        model.fit(self.x_train, self.y_train_cat, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1, use_multiprocessing=True)
        clf_res = pd.DataFrame(columns=('method name', 'accuracy', 'time'))
        clf_res.loc[len(clf_res)] = ['FFNN', round(model.evaluate(self.x_test, self.y_test_cat, verbose=0)[1] * 100, 2), round(time() - start_fit, 2)]

        print('keras FFNN accuracy:', round(model.evaluate(self.x_test, self.y_test_cat, verbose=0)[1] * 100, 2), '-->', round(time() - start_fit, 2))
        return self