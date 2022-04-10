'''This Py file defines the TensorFlow dataset Class for the TensorFlow basic image
classification tutorial, found here:
https://www.tensorflow.org/tutorials/keras/classification'''

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


class TensorFlowModel:
#this class defines a dataset from a TensorFlow perspective
    def __init__(self, inputShape, firstDenseNeurons, activation, secondDenseNeurons):
        self.inputShape = inputShape
        self.firstDenseNeurons = firstDenseNeurons
        self.activation = activation
        self.secondDenseNeurons = secondDenseNeurons
        self.model = None
        self.probabilityModel = None

    def compileModel(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=self.inputShape),
            tf.keras.layers.Dense(self.firstDenseNeurons, activation=self.activation),
            tf.keras.layers.Dense(self.secondDenseNeurons)
        ])
        print("Model compiled.")

    def feedModel(self, trainImages, trainLabels, epochs):
        self.model.fit(trainImages, trainLabels, epochs=epochs)
        print("Model fed training data.")

    def evaluateModelAccuracy(self, verbose):
        testLoss, testAcc = self.model.evaluate(test_images, test_labels, verbose=verbose)
        result = {}
        result["Test loss"] = testLoss
        result["Test accuracy"] = testAcc
        print("Model accuracy evaluated.")
        return result

    def createProbabilityModel(self):
        self.probabilityModel = tf.keras.Sequential([self.model,
                                             tf.keras.layers.Softmax()])
        print("Probability model created.")

    def createPredictions(self, imageset):
        predictions = self.probabilityModel.predict(imageset)
        print("Predictions for this imageset created:")
        print("Predictions: " + predictions[0])
        print("Label with highest confidence value: " + np.argmax(predictions[0]))
        return predictions

    def predict(self, image):
        result = self.probabilityModel.predict(image)
        print("Prediction result: " + result)
        return result