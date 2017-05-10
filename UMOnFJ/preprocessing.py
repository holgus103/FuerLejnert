"""
Simple preprocessing utilities including loading data from CSV and faking foreign elements.
"""

import csv
import numpy as np

def getData(fileName):
    """
    Reads CSV and returns a tuple of features matrix and classes vector.

    fileName -- path to the CSV to read
    """

    # X -- features matrix
    X = []
    # y -- classes vector
    y = []

    with open(fileName) as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            X.append(row[1:])
            y.append(row[0])

    X = np.array(X)
    y = np.array(y)

    return (X, y)

def getLabels(fileName):
    """
    Reads CSV and returns an array of features' names (labels).

    fileName -- path to the CSV to read
    """

    labels = []

    with open(fileName) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            for label in row:
                labels.append(label)

    return np.array(labels).reshape((1, -1))

def rejectClasses(X, y, rejectedClasses):
    """
    Marks classes as foreign and returns a new training set tuple.

    X -- features matrix
    y -- classes vector
    rejectedClasses -- classes to mark as foreign
    """

    mask = np.ones_like(y)

    for c in rejectedClasses:
        mask = np.logical_and(mask, y != c)

    # -1 ought to be the class label of any foreign element
    y[~mask] = -1

    # include only native elements in the training set
    X_train = X[mask]
    y_train = y[mask]

    return (X_train, y_train)
