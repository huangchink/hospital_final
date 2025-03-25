# -*- coding: utf-8 -*-

"""
Polynomial calibration module for gaze correction
X = a0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * y + a_5 * x * y + a_6 * x^2*y + a_7 * x^3*y
Y = b0 + b_1 * x + b_2 * x^2 + b_3 * y + b_4 * y^2 + b_5 * x * y + b_6 * x^2 * y

Please see:
Blignaut, P. (2014). Mapping the pupil-glint vector to gaze coordinates in a simple video-based eye tracker.
Journal of Eye Movement Research, 7(1).
"""

import os
import pathlib
from typing import Tuple

import joblib
import numpy as np
import pygame
from numpy import ndarray
from pygame.locals import KEYDOWN, K_RETURN
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from gazefollower import GazeFollower
from gazefollower.calibration import Calibration
from gazefollower.logger import Log


class PolyCalibration(Calibration):
    """
    Polynomial regression-based calibration for gaze coordinates correction
    Uses separate linear models for X and Y coordinates with polynomial features
    """

    def __init__(self, model_save_path=""):
        """
        Initialize calibration model
        Args:
            model_save_path: Optional custom path for model persistence
        """
        super().__init__()
        self.linear_x = LinearRegression()
        self.linear_y = LinearRegression()
        self.standard_scaler_x = StandardScaler()
        self.standard_scaler_y = StandardScaler()

        # Configure model storage paths
        if model_save_path:
            self.workplace_calibration_dir = pathlib.Path(model_save_path)
        else:
            self.workplace_calibration_dir = pathlib.Path.home().joinpath("GazeFollower", "calibration")

        # Create directory if not exists
        self.workplace_calibration_dir.mkdir(parents=True, exist_ok=True)

        # Define model file paths
        self.cali_x_path = self.workplace_calibration_dir.joinpath("poly_cali_x.bin")
        self.cali_y_path = self.workplace_calibration_dir.joinpath("poly_cali_y.bin")
        self.scaler_x_path = self.workplace_calibration_dir.joinpath("poly_scaler_x.bin")
        self.scaler_y_path = self.workplace_calibration_dir.joinpath("poly_scaler_y.bin")

        # Load pre-trained models and scalers if available
        if (self.cali_x_path.exists() and self.cali_y_path.exists() and
                self.scaler_x_path.exists() and self.scaler_y_path.exists()):
            try:
                self.linear_x = joblib.load(self.cali_x_path)
                self.linear_y = joblib.load(self.cali_y_path)
                self.standard_scaler_x = joblib.load(self.scaler_x_path)
                self.standard_scaler_y = joblib.load(self.scaler_y_path)
                self.has_calibrated = True
                Log.d("Loaded pre-trained calibration models and scalers")
            except Exception as load_error:
                Log.d(f"Model loading failed: {str(load_error)}")
                self.has_calibrated = False
        else:
            self.has_calibrated = False

    def calibrate(self, features: np.ndarray, labels: np.ndarray, ids=None) -> Tuple[bool, float, ndarray]:
        """
        Train calibration models using polynomial features
        Args:
            features: Raw gaze coordinates [N x 2] (x, y)
            labels: Target screen coordinates [N x 2] (x, y)
            ids: Optional subject IDs for personalized calibration
        Returns:
            Tuple: (success status, mean error, calibrated coordinates)
        """
        features = features.astype(np.float32)
        labels = labels.astype(np.float32)

        # Split target coordinates
        labels_x = labels[:, 0].reshape(-1, 1)
        labels_y = labels[:, 1].reshape(-1, 1)

        # Generate polynomial features
        # Reshape features[:,0] and features[:,1] to 2D for scaler
        x_raw = features[:, 0].reshape(-1, 1)
        y_raw = features[:, 1].reshape(-1, 1)
        x = self.standard_scaler_x.fit_transform(x_raw)
        y = self.standard_scaler_y.fit_transform(y_raw)

        features_x = np.column_stack((
            x,  # x
            x ** 2,  # x²
            x ** 3,  # x³
            y,  # y
            x * y,  # xy
            (x ** 2) * y,  # x²y
            (x ** 3) * y  # x³y
        ))

        features_y = np.column_stack((
            x,  # x
            x ** 2,  # x²
            y,  # y
            y ** 2,  # y²
            x * y,  # xy
            (x ** 2) * y  # x²y
        ))

        try:
            # Train separate models for X and Y coordinates
            self.linear_x.fit(features_x, labels_x)
            self.linear_y.fit(features_y, labels_y)
            self.has_calibrated = True
        except Exception as train_error:
            self._handle_training_error(train_error)
            return False, float('inf'), np.zeros_like(labels)

        # Calculate calibration metrics
        pred_x = self.linear_x.predict(features_x)
        pred_y = self.linear_y.predict(features_y)
        predictions = np.hstack((pred_x, pred_y))
        error = self._calculate_euclidean_error(labels, predictions)

        Log.d(f"Calibration complete. Mean error: {error:.2f}px")
        return True, error, predictions

    def _handle_training_error(self, error: Exception):
        """Clean up failed training artifacts"""
        self.has_calibrated = False
        for path in [self.cali_x_path, self.cali_y_path, self.scaler_x_path, self.scaler_y_path]:
            if path.exists():
                path.unlink()
                Log.d(f"Removed invalid model/scaler file: {path}")

    def _calculate_euclidean_error(self, true: np.ndarray, pred: np.ndarray) -> float:
        """Compute mean Euclidean distance between predictions and ground truth"""
        return np.mean(np.sqrt(np.sum((true - pred) ** 2, axis=1)))

    def save_model(self) -> bool:
        """Persist trained models and scalers to disk
        Returns:
            bool: True if save successful
        """
        if not self.has_calibrated:
            Log.d("No trained models to save")
            return False

        try:
            joblib.dump(self.linear_x, self.cali_x_path)
            joblib.dump(self.linear_y, self.cali_y_path)
            joblib.dump(self.standard_scaler_x, self.scaler_x_path)
            joblib.dump(self.standard_scaler_y, self.scaler_y_path)
            Log.d(f"Models and scalers saved to {self.cali_x_path.parent}")
            return True
        except Exception as save_error:
            Log.d(f"Model save failed: {str(save_error)}")
            return False

    def predict(self, features: np.ndarray, estimated_coordinate: Tuple[float, float]) -> Tuple[
        bool, Tuple[float, float]]:
        """
        Correct gaze coordinates using trained models
        Args:
            features: Raw gaze features [x, y]
            estimated_coordinate: Fallback coordinates if prediction fails
        Returns:
            Tuple: (success status, (calibrated_x, calibrated_y))
        """
        if not self.has_calibrated:
            Log.d("Using uncalibrated estimates")
            return False, estimated_coordinate

        try:
            # Expect features as a 1D array [x, y]
            x_val, y_val = features[0], features[1]
            # Reshape to 2D for scaler
            x = np.array([[x_val]])
            y = np.array([[y_val]])
            x_scaled = self.standard_scaler_x.transform(x)
            y_scaled = self.standard_scaler_y.transform(y)

            features_x = np.column_stack((
                x_scaled,  # x
                x_scaled ** 2,  # x²
                x_scaled ** 3,  # x³
                y_scaled,  # y
                x_scaled * y_scaled,  # xy
                (x_scaled ** 2) * y_scaled,  # x²y
                (x_scaled ** 3) * y_scaled  # x³y
            ))

            features_y = np.column_stack((
                x_scaled,  # x
                x_scaled ** 2,  # x²
                y_scaled,  # y
                y_scaled ** 2,  # y²
                x_scaled * y_scaled,  # xy
                (x_scaled ** 2) * y_scaled  # x²y
            ))

            calibrated_x = self.linear_x.predict(features_x)[0][0]
            calibrated_y = self.linear_y.predict(features_y)[0][0]
            return True, (float(calibrated_x), float(calibrated_y))
        except Exception as pred_error:
            Log.d(f"Prediction error: {str(pred_error)}")
            return False, estimated_coordinate

    def release(self):
        """Clean up model resources"""
        del self.linear_x
        del self.linear_y
        self.has_calibrated = False


if __name__ == '__main__':
    # init pygame
    pygame.init()
    win = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)

    # init GazeFollower
    gf = GazeFollower(calibration=PolyCalibration())
    # previewing
    gf.preview(win=win)
    # calibrating
    gf.calibrate(win=win)
    # sampling
    gf.start_sampling()
    pygame.time.wait(100)  # sleep for 100 ms so the tracker caches some samples

    # A free viewing task, in which we show a picture and overlay the gaze cursor
    img_folder = 'images'
    images = ['grid.jpg']

    # show the images one by one in a loop, press the ENTER key to exit the program
    for _img in images:
        # show the image on screen
        win.fill((128, 128, 128))
        im = pygame.image.load(os.path.join(img_folder, _img))
        win.blit(im, (0, 0))
        pygame.display.flip()
        # send a trigger to record in the eye movement data to mark picture onset
        gf.send_trigger(202)

        # now let's show the gaze point, press any key to close the window
        got_key = False
        max_duration = 20000
        t_start = pygame.time.get_ticks()
        pygame.event.clear()  # clear all cached events if there were any
        while not (got_key or (pygame.time.get_ticks() - t_start) >= max_duration):
            # check key presses
            for ev in pygame.event.get():
                if ev.type == KEYDOWN:
                    if ev.key == K_RETURN:
                        got_key = True

            # update the visual (image and cursor)
            win.blit(im, (0, 0))
            # show gaze cursor; when in a formal experiment you can remove this code
            gaze_info = gf.get_gaze_info()
            if gaze_info and gaze_info.status:
                gx = int(gaze_info.filtered_gaze_coordinates[0])
                gy = int(gaze_info.filtered_gaze_coordinates[1])
                pygame.draw.circle(win, (0, 255, 0), (gx, gy), 50, 5)  # cursor for the left eye

            # flip the frame
            pygame.display.flip()

    # stop sampling
    pygame.time.wait(100)  # sleep for 100 ms to capture ending samples
    gf.stop_sampling()

    # save the sample data to file
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    file_name = "free_viewing_pygame_demo.csv"
    gf.save_data(os.path.join(data_dir, file_name))
    # release gaze follower
    gf.release()
