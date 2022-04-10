'''This Py file defines the TensorFlow dataset Class for the TensorFlow basic image
classification tutorial, found here:
https://www.tensorflow.org/tutorials/keras/classification'''

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


class TensorFlowDataSet:
#this class defines a dataset from a TensorFlow perspective
    def __init__(self, dataset, class_names):
        self.rawDataset = dataset
        self.classNames = class_names
        self.trainImages = None
        self.trainLabels = None
        self.testImages = None
        self.testLabels = None

    def loadData(self):
        (self.trainImages, self.trainLabels), (self.testImages, self.testLabels) = self.rawDataset.load_data()
        print("Data loaded.")

    def exploreData(self):
        print("Train images shape:")
        print(self.trainImages.shape)
        print("Train labels length:")
        print(len(self.trainLabels))
        print("Train labels raw:")
        print(self.trainLabels)
        print("Test images shape:")
        print(self.testImages.shape)
        print("Test labels length:")
        print(len(self.testLabels))

    def preprocessData(self):
        plt.figure()
        plt.imshow(self.trainImages[0])
        plt.colorbar()
        plt.grid(False)
        plt.show()

        pixelValue = "255"
        print("Max pixel value is " + pixelValue)

        self.trainImages = self.trainImages / 255.0
        self.testImages = self.testImages / 255.0

        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.trainImages[i], cmap=plt.cm.binary)
            plt.xlabel(self.classNames[self.trainLabels[i]])
        plt.show()
        print("Preprocessing complete.")