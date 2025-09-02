#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 01 18:29:16 2025

@author: ndw
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import codecs
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("dataset/votingPreprocessed.csv")

X = dataset.iloc[:, 8:10].values
y = dataset.iloc[:, -1].values

print('>>: X: ', X)
print('>>: y: ', y)

'''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder = "passthrough")
X = np.array(ct.fit_transform(X))
print(X)
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.04, random_state = 1)
    
def compileModel():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score
    import joblib
    import coremltools as ct
    from coremltools.models import datatypes
    
    # Train model
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Save sklearn model
    joblib.dump(classifier, "hxh4_fishing_prediction_new.pkl")
    
    # Convert to CoreML
    # Input features must be defined properly for CoreML conversion
    input_features = [("dayNumberOfYear", datatypes.Int64()), ("moonPhase", datatypes.Int64())]
    output_feature = "quality"
    
    # Convert sklearn model to CoreML
    coreml_model = ct.converters.sklearn.convert(classifier, input_features, output_feature)
    
    coreml_model.author = 'Gennady Dmitrik'
    coreml_model.license = 'Private'
    coreml_model.short_description = 'Predicts the fishing quality for tomorrow'
    coreml_model.version = '2.5'
    coreml_model.input_description['dayNumberOfYear'] = 'Day number of the year'
    coreml_model.input_description['moonPhase'] = 'Current moon phase'
    coreml_model.output_description['quality'] = 'Predicted quality 0(bad) or 1(good)'
    
    # Save CoreML model
    coreml_model.save("HXH4.v.2.5.mlmodel")


def all():
    # =============================================================================
    #     from sklearn.preprocessing import StandardScaler
    #     sc = StandardScaler()
    #     X_train = sc.fit_transform(X_train)
    #     X_test = sc.transform(X_test)
    # =============================================================================    

    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print('>>: DecisionTreeClassifier\n', cm)
    print('>>: Accuracy score: ', accuracy_score(y_test, y_pred))
    pred = classifier.predict([[235, 0]])
    print('>>: pred1: ', pred)
    
    # Plotting Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test)), y_test, color="red", label="Actual", alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, color="blue", label="Predicted", alpha=0.6, marker="x")
    plt.title("Actual vs Predicted Quality")
    plt.xlabel("DecisionTreeClassifier")
    plt.ylabel("Quality")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
    
    
    # Training the K-NN model on the Training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print('>>: KNeighborsClassifier\n', cm)
    print('>>: Accuracy score: ', accuracy_score(y_test, y_pred))
    pred = classifier.predict([[235, 0]])
    print('>>: pred2:  ', pred)
    
    # Plotting Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test)), y_test, color="red", label="Actual", alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, color="blue", label="Predicted", alpha=0.6, marker="x")
    plt.title("Actual vs Predicted Quality")
    plt.xlabel("KNeighborsClassifier")
    plt.ylabel("Quality")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
    
    
    # Training the Naive Bayes model on the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print('>>: GaussianNB\n', cm)
    print('>>: Accuracy score: ', accuracy_score(y_test, y_pred))
    pred = classifier.predict([[235, 0]])
    print('>>: pred3: ', pred)
    
    # Plotting Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test)), y_test, color="red", label="Actual", alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, color="blue", label="Predicted", alpha=0.6, marker="x")
    plt.title("Actual vs Predicted Quality")
    plt.xlabel("GaussianNB")
    plt.ylabel("Quality")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    
    # Training the Random Forest Classification model on the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print('>>: RandomForestClassifier\n', cm)
    print('>>: Accuracy score: ', accuracy_score(y_test, y_pred))
    pred = classifier.predict([[235, 0]])
    print('>>: pred4: ', pred)
    
    # Plotting Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test)), y_test, color="red", label="Actual", alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, color="blue", label="Predicted", alpha=0.6, marker="x")
    plt.title("Actual vs Predicted Quality")
    plt.xlabel("RandomForestClassifier")
    plt.ylabel("Quality")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


    X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X, y, test_size = 0.2, random_state = 1)
    from sklearn.preprocessing import StandardScaler
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units = 6, activation = "relu"))
    ann.add(tf.keras.layers.Dense(units = 8, activation = "relu"))
    ann.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid"))

    ann.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    ann.fit(X_train_tf, y_train_tf, batch_size = 32, epochs = 40)
    
    from sklearn.metrics import confusion_matrix, accuracy_score
    
    y_pred_probs = ann.predict(X_test_tf)
    y_pred_classes = (y_pred_probs > 0.5).astype("int32")
    print('>>: Accuracy score: ', accuracy_score(y_test_tf, y_pred_classes))
    print(">>: pred5: ", ann.predict(np.array([[235, 0]])))


    y_pred = ann.predict(X_test_tf)
    #print('>>: y_pre: ', y_pred)
    #y_pred = y_pred.reshape(len(y_pred),1)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test_tf, y_pred)
    print(cm)

    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test_tf)), y_test_tf, 
                color="red", label="Actual", alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, 
                color="blue", label="Predicted", alpha=0.6, marker="x")
    plt.title("Actual vs Predicted Quality", fontsize=14)
    plt.xlabel("Tensorflow", fontsize=12)
    plt.ylabel("Quality", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y_test_tf)), y_test_tf, label="Actual", color="red", linewidth=2)
    plt.plot(np.arange(len(y_pred)), y_pred, label="Predicted", color="blue", linestyle="--")
    plt.title("Actual vs Predicted Quality (Line Plot)", fontsize=14)
    plt.xlabel("Sample Index1")
    plt.ylabel("Quality")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


all()
compileModel()













