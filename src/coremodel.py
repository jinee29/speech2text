import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from keras import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


class RandomForestModel:
    def __init__(self):
        self.clf = RandomForestClassifier(max_depth=2, random_state=0)

    def train(self, x_train, y_train, x_valid, y_valid):
        x_train = x_train.reshape(x_train.shape[0],
                                  x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
        x_valid = x_valid.reshape(x_valid.shape[0],
                                  x_valid.shape[1] * x_valid.shape[2] * x_valid.shape[3])
        self.clf.fit(x_train, y_train)
        self.predict = self.clf.predict(x_valid)
        return self.clf


class DecisionTreeModel:
    def __init__(self):
        self.clf = DecisionTreeClassifier()

    def train(self, x_train, y_train, x_valid, y_valid):
        x_train = x_train.reshape(x_train.shape[0],
                                  x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
        x_valid = x_valid.reshape(x_valid.shape[0],
                                  x_valid.shape[1] * x_valid.shape[2] * x_valid.shape[3])
        self.clf.fit(x_train, y_train)
        self.predict = self.clf.predict(x_valid)
        return self.clf


class SVMModel:
    def __init__(self):
        self.clf = svm.SVC()

    def train(self, x_train, y_train, x_valid, y_valid):
        x_train = x_train.reshape(x_train.shape[0],
                                  x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
        x_valid = x_valid.reshape(x_valid.shape[0],
                                  x_valid.shape[1] * x_valid.shape[2] * x_valid.shape[3])
        self.clf.fit(x_train, y_train)
        self.predict = self.clf.predict(x_valid)
        return self.clf


class MLPClass:
    def __init__(self):
        self.clf = MLPClassifier(solver='lbfgs',
                                 alpha=1e-5,
                                 hidden_layer_sizes=(50, 3), random_state=1)
    def train(self, x_train, y_train, x_valid, y_valid):
        x_train = x_train.reshape(x_train.shape[0],
                                  x_train.shape[1] * x_train.shape[2] * x_train.shape[3])

        x_valid = x_valid.reshape(x_valid.shape[0],
                                  x_valid.shape[1] * x_valid.shape[2] * x_valid.shape[3])
        self.clf.fit(x_train, y_train)
        self.predict = self.clf.predict(x_valid)
        return self.clf


class DeepLearningModel:
    def __init__(self):
        self.clf = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(1025, 50, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dense(32),
            Dense(16),
            Flatten(),
            Dense(5),
            Activation('softmax')
        ])
        self.clf.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        self.clf.summary()
        self.early_stop = EarlyStopping(monitor='val_accuracy', mode='max',
                                   verbose=1, patience=10, min_delta=0.0001)

    def train(self, x_train, y_train, x_valid, y_valid):
        self.clf.fit(
            x=x_train,
            y=y_train,
            epochs=5,
            callbacks=[self.early_stop],
            batch_size=32,
            verbose=1,
            validation_data=(x_valid, y_valid)
        )
        self.predict = self.clf.predict(x_valid)
        self.predict = np.argmax(self.predict, 1)


class Tuning:
    def __init__(self, label_map, list_solutions=["DecisionTree", "DeepLearning", "RandomForest", "SVM", "MLP"]):
        self.label_map = label_map
        self.list_solutions = list_solutions
        map_solution = {"DeepLearning": DeepLearningModel,
                        "RandomForest": RandomForestModel,
                        "MLP": MLPClass,
                        "SVM": SVMModel,
                        "DecisionTree": DecisionTreeModel}
        self.models = {}
        for solution in self.list_solutions:
            self.models[solution] =map_solution[solution]()

    def run(self, x_train, y_train, x_valid, y_valid):
        self.scores = {}
        for solution in self.list_solutions:
            print("Solution running: ", solution)
            self.models[solution].train(x_train, y_train, x_valid, y_valid)

            score = classification_report(y_valid,
                                          self.models[solution].predict,
                                          output_dict=True,
                                          target_names=list(self.label_map.keys()))
            print("Score: ", score)
            print("\n")
            self.scores[solution] = score["weighted avg"]["f1-score"]
        self.max_solution = max(self.scores, key=self.scores.get)
        print("Max score solution: ", self.max_solution)
        self.save_model()

    def save_model(self):
        outfile = open('model/scores.json', "w")
        json.dump(self.scores, outfile, indent=4)
        filename = "model/speech2text_model"
        if self.max_solution != "DeepLearning":
            with open(filename + '.sav', "wb") as f:
                pickle.dump(self.models[self.max_solution].clf, f)
            f.close()
        else:
            self.models[self.max_solution].clf.save(filename)
