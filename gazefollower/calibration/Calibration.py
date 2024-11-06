# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

from typing import List


class Calibration:
    def __init__(self):
        """
        Initializes the Calibration class.

        Attributes:
        ----------
        has_calibrated : bool
            A flag indicating whether the calibration process has been completed.
        """
        self.has_calibrated = False

    def calibrate(self, features, labels) -> float:
        """
        Calibrates the model using the provided features and labels.

        Parameters:
        ----------
        features : array-like of shape (n, m)
            The input features used for calibration, where n is the number of samples 
            and m is the number of features per sample.

        labels : array-like of shape (n, 2)
            The target labels corresponding to the input features, where n is the number 
            of samples. Each label should have two components.

        Returns:
        -------
        float
            The fitness score of the calibration, indicating how well the model has 
            been calibrated using the provided data.
        """
        # Implementation goes here
        raise NotImplementedError("Subclasses must implement this method.")

    def save_model(self) -> bool:
        """
        Saves the calibrated model to a persistent storage.

        This method should be called after the calibration process is completed 
        to save the model's state and make it available for future use.

        :return: True if the models were saved, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def predict(self, features, estimated_coordinate) -> List:
        """
        Predicts the target labels for the provided features using the calibration process.
        :param estimated_coordinate: estimated coordinates from the gaze estimation model.
        :param features: The input features from the gaze estimation model.
        :return: X and Y coordinates of the predicted target.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def release(self):
        raise NotImplementedError("Subclasses must implement this method.")
