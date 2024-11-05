#!/usr/bin/bash
# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import pathlib
from typing import List

import cv2 as cv
import numpy as np

from .Calibration import Calibration


class SVRCalibration(Calibration):
    def __init__(self, model_save_path: str = ""):
        """
        Initializes the Calibration class with two SVM models for x and y coordinates.
        :param model_save_path: default path is {HOME}/GazeFollower/Calibration when `model_save_path` == ""
        """
        super().__init__()

        if model_save_path == "":
            self.workplace_calibration_dir = pathlib.Path.home().joinpath("GazeFollower", "Calibration")
            if not self.workplace_calibration_dir.exists():
                self.workplace_calibration_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.workplace_calibration_dir = pathlib.Path(model_save_path)

        self.svr_x_path = self.workplace_calibration_dir.joinpath("svr_x.xml")
        self.svr_y_path = self.workplace_calibration_dir.joinpath("svr_y.xml")

        self.svr_x = cv.ml.SVM.create()  # SVM model for the x coordinate
        self.svr_y = cv.ml.SVM.create()  # SVM model for the y coordinate

        if self.svr_y_path.exists() and self.svr_x_path.exists():
            self.svr_x.load(str(self.svr_x_path))
            self.svr_y.load(str(self.svr_y_path))
            self.has_calibrated = True
        else:
            # Set default parameters for the SVM models
            self._set_svm_params(self.svr_x)
            self._set_svm_params(self.svr_y)
            self.has_calibrated = False

    def _set_svm_params(self, svr):
        """
        Sets the default SVM parameters for the model.

        :param svr: The SVM model to set parameters for.
        """
        svr.setType(cv.ml.SVM_EPS_SVR)
        svr.setKernel(cv.ml.SVM_RBF)
        svr.setC(1.0)  # Example C value, adjust as needed
        svr.setGamma(0.005)  # Example gamma value, adjust as needed
        svr.setP(0.001)  # Epsilon for loss function
        term_criteria = (cv.TERM_CRITERIA_MAX_ITER, 10000, 1e-4)
        svr.setTermCriteria(term_criteria)

    def predict(self, features, estimated_coordinate) -> List:
        """
        Predicts the x and y coordinates using the trained SVM models.

        :param features: np.array of shape (1, m), where m is the number of features.
                        This represents a single sample feature vector for prediction.
        :param estimated_coordinate: the estimated coordinate from the gaze estimation model.
        :return: A list containing the predicted x and y coordinates [x_pred, y_pred].
        """
        # Ensure the feature is a numpy array with the correct dtype and shape
        features = np.array(features, dtype=np.float32).reshape(1, -1)

        # Check if models are trained before making predictions
        if not self.has_calibrated:
            print("SVM models are not trained. It will return estimated coordinate from gaze estimation model.")
            return estimated_coordinate

        # Predict x and y coordinates using the trained SVM models
        predicted_x = self.svr_x.predict(features)[1].flatten()[0]
        predicted_y = self.svr_y.predict(features)[1].flatten()[0]

        # Return the predictions as a list
        return [predicted_x, predicted_y]

    def calibrate(self, features, labels) -> float:
        """
        Trains the SVM models using the provided features and labels, and calculates the mean Euclidean error.

        :param features: np.array of shape (n, m), where n is the number of samples and m is the number of features.
        :param labels: np.array of shape (n, 2), where n is the number of samples.
                       The first column represents x labels, and the second column represents y labels.
        :return: fitness (mean Euclidean distance)
        """
        # Ensure that features and labels are numpy arrays
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        # Split labels into x and y components
        labels_x = labels[:, 0].reshape(-1, 1)  # Extract x labels
        labels_y = labels[:, 1].reshape(-1, 1)  # Extract y labels

        try:
            # Train SVM models
            self.svr_x.train(features, cv.ml.ROW_SAMPLE, labels_x)
            self.svr_y.train(features, cv.ml.ROW_SAMPLE, labels_y)
            self.has_calibrated = True
        except Exception as e:
            self.has_calibrated = False

            print("Failed to train SVM model: {}".format(e.args))
            print("Try to delete previously trained model.")
            if self.svr_x_path.exists():
                self.svr_x_path.unlink()  # Deletes svr_x.xml
                print(f"Deleted: {self.svr_x_path}")
            else:
                print(f"No trained model found at {self.svr_x_path}")

            if self.svr_y_path.exists():
                self.svr_y_path.unlink()  # Deletes svr_y.xml
                print(f"Deleted: {self.svr_y_path}")
            else:
                print(f"No trained model found at {self.svr_y_path}")

            return float('inf')  # Return infinity as fitness to indicate training failure

        # Predict using the trained models
        predicted_x = self.svr_x.predict(features)[1]
        predicted_y = self.svr_y.predict(features)[1]

        # Calculate the Euclidean distance between the predicted and actual labels
        euclidean_distances = np.sqrt((labels_x - predicted_x) ** 2 + (labels_y - predicted_y) ** 2)

        # Calculate the mean Euclidean error
        mean_euclidean_error = np.mean(euclidean_distances)

        print(f"Calibration completed with mean Euclidean error: {mean_euclidean_error:.4f}")
        return mean_euclidean_error

    def save_model(self) -> bool:
        """
        Saves the trained SVM models to XML files.

        :return: True if the models were saved, False otherwise.
        """

        # Check if the SVM models are trained
        if self.svr_x.isTrained() and self.svr_y.isTrained():
            self.svr_x.save(str(self.svr_x_path))
            self.svr_y.save(str(self.svr_y_path))
            print(f"SVR model for x coordinate saved at: {self.svr_x_path}")
            print(f"SVR model for y coordinate saved at: {self.svr_y_path}")
            return True
        else:
            print("SVR model for x coordinate has not been trained yet.")
            print("SVR model for y coordinate has not been trained yet.")
            return False

    def release(self):
        pass
