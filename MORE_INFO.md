# How to use GazeFollower with PsychoPy and Pygame.

GazeFollower has supported two mainstream psychophysics experiment software
packages, PsychoPy and PyGame, since version 1.0.0. For simple experimental
tasks, we strongly recommend using PyGame due to noticeable lag issues in
PsychoPy camera preview functionality, which we have been unable to resolve
at this time. Below is code compatible with both backends.

## PsychoPy

The code source can be found [here](example/psychopy_example.py).

```python
# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import os

from psychopy import visual, core, event

from gazefollower import GazeFollower

if __name__ == '__main__':
    win = visual.Window(fullscr=True, color='white')
    gaze_cursor = visual.ShapeStim(win, vertices='circle', size=(100, 100), lineWidth=5,
                                   fillColor=None, lineColor=(0, 255, 0), colorSpace='rgb255',
                                   units='pix')
    # init GazeFollower
    gf = GazeFollower()
    gf.preview(win=win)
    gf.calibrate(win=win)
    gf.start_sampling()
    core.wait(0.1)

    # images need to show
    img_folder = 'images'
    images = ['grid.jpg']

    # show the images one by one in a loop, press a ENTER key to exit the program
    for _img in images:
        # show the image on screen
        im = visual.ImageStim(win, os.path.join(img_folder, _img))
        im.draw()
        win.flip()
        # send a trigger to record in the eye movement data to mark picture onset
        gf.send_trigger(202)

        # now lets show the gaze point, press any key to close the window
        got_key = False
        max_duration = 20.0
        t_start = core.getTime()
        event.clearEvents()  # clear all cached events if there were any
        gx, gy = -65536, -65536
        while not (got_key or (core.getTime() - t_start) >= max_duration):

            # check keyboard events
            if event.getKeys():
                got_key = True

            # redraw the screen
            win.color = (0, 0, 0)
            im.draw()

            # show gaze cursor, when formal experiment 
            # you can remove this code
            # ++++++++++++++++++++++++
            gaze_info = gf.get_gaze_info()

            screen_width = win.size[0]
            screen_height = win.size[1]
            if gaze_info and gaze_info.status:
                raw_gx = int(gaze_info.filtered_gaze_coordinates[0])
                raw_gy = int(gaze_info.filtered_gaze_coordinates[1])
                if (raw_gx != -65536) and (raw_gy != -65536):
                    gx = raw_gx - screen_width // 2
                    gy = (screen_height // 2) - raw_gy
                    gaze_cursor.pos = (gx, gy)
                    gaze_cursor.draw()
                else:
                    gaze_cursor.pos = (9999, 9999)
            else:
                gaze_cursor.pos = (9999, 9999)
            # ++++++++++++++++++++++++

            # flip the frame
            win.flip()

    core.wait(0.1)
    gf.stop_sampling()
    win.close()
    # save the sample data to file
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    # save data
    file_name = "free_viewing_psychopy_demo.csv"
    gf.save_data(os.path.join(data_dir, file_name))
    gf.release()
    # quit
    core.quit()
```

## PyGame

The code source can be found [here](example/pygame_example.py).

```python
# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import os

import pygame
from pygame.locals import KEYDOWN, K_RETURN

from gazefollower import GazeFollower
from gazefollower.gaze_estimator import MGazeNetGazeEstimator

if __name__ == '__main__':
    # init pygame
    pygame.init()
    win = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)

    # init GazeFollower
    gf = GazeFollower()
    # previewing
    gf.preview(win=win)
    # calibrating
    gf.calibrate(win=win)
    # sampling
    gf.start_sampling()
    pygame.time.wait(100)  # sleep for 100 ms so the tracker cache some sample

    # A free viewing task, in which we show a picture and overlay the gaze cursor
    img_folder = 'images'
    images = ['grid.jpg']

    # show the images one by one in a loop, press a ENTER key to exit the program
    for _img in images:
        # show the image on screen
        win.fill((128, 128, 128))
        im = pygame.image.load(os.path.join(img_folder, _img))
        win.blit(im, (0, 0))
        pygame.display.flip()
        # send a trigger to record in the eye movement data to mark picture onset
        gf.send_trigger(202)

        # now lets show the gaze point, press any key to close the window
        got_key = False
        max_duration = 20000
        t_start = pygame.time.get_ticks()
        pygame.event.clear()  # clear all cached events if there were any
        gx, gy = -65536, -65536
        while not (got_key or (pygame.time.get_ticks() - t_start) >= max_duration):
            # check key presses
            for ev in pygame.event.get():
                if ev.type == KEYDOWN:
                    if ev.key == K_RETURN:
                        got_key = True

            # update the visual (image and cursor)
            win.blit(im, (0, 0))
            # show gaze cursor, when formal experiment
            # you can remove this code
            # ++++++++++++++++++++++++
            gaze_info = gf.get_gaze_info()
            if gaze_info and gaze_info.status:
                gx = int(gaze_info.filtered_gaze_coordinates[0])
                gy = int(gaze_info.filtered_gaze_coordinates[1])
                pygame.draw.circle(win, (0, 255, 0), (gx, gy), 50, 5)  # cursor for the left eye
            # ++++++++++++++++++++++++

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
```

# Customize your components

GazeFollower consists of four components, [calibration](gazefollower/calibration/Calibration.py),
[camera](gazefollower/camera/Camera.py), [face alignment](gazefollower/face_alignment/FaceAlignment.py),
[filter](gazefollower/filter/Filter.py), and [gaze estimator](gazefollower/gaze_estimator/GazeEstimator.py).
If an abstract class for calibration has been implemented, it can be instantiated and passed to the GazeFollower
class. For example, in the following code snippet, gf = GazeFollower(calibration=PolyCalibration()) demonstrates
how to integrate a customized calibration module.

```python
from numpy as np
from numpy import ndarray
from gazefollower import GazeFollower
from gazefollower.calibration import Calibration
from gazefollower.logger import Log
from typing import Tuple


class PolyCalibration(Calibration):
    def __init__(self):

    # your code here

    def calibrate(self, features: np.ndarray, labels: np.ndarray, ids=None) -> Tuple[bool, float, ndarray]:

    # your code here

    def save_model(self) -> bool:

    # your code here

    def predict(self, features: np.ndarray, estimated_coordinate: Tuple[float, float]) -> Tuple:

    # your code here

    def release(self):


# your code here


if __name__ == '__main__':
    # init GazeFollower
    gf = GazeFollower(calibration=PolyCalibration())
    # previewing
    gf.preview()
    # calibrating
    gf.calibrate()
    # sampling
    gf.start_sampling()

```

Below is an example of how each component is used.

## Calibration

We implemented [a polynomial calibration](https://bop.unibe.ch/JEMR/article/view/2373), but its performance is not
satisfactory

```python
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

```

## Camera

```python

# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import ctypes
import os
import sys
import threading
import time
from ctypes import wintypes

import cv2
import numpy as np
import pygame
from pygame.locals import KEYDOWN, K_RETURN

from gazefollower import GazeFollower
from gazefollower.camera import Camera
from gazefollower.logger import Log

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]


class PupilioCamera(Camera):
    """
    Pupil.IO (https://github.com/GanchengZhu/Pupilio) is an eye-tracking device based on a dual-camera, 
    single-light-source system. This script utilizes Pupil.IO’s built-in cameras to capture near-infrared images 
    and performs appearance-based eye tracking.
    """

    def __init__(self):
        """
        Initialize the camera and the constant.
        """
        # Initializes the camera state to CLOSING and prepares a lock for callback management
        super().__init__()
        self.camera_utils = None
        # image size
        self.source_img_width = 1280
        self.source_img_height = 1024
        self.target_img_width = 640
        self.target_img_height = 480
        self.dll_handle = None
        self.camera_utils = ctypes.CDLL('customized_camera_native_library/lib_pupil_io_camera.dll', winmode=0)
        self.dll_handle = self.camera_utils._handle
        # camera util
        self.camera_utils.init.restype = ctypes.c_int
        self.camera_utils.deinit.restype = ctypes.c_int
        self.camera_utils.retrieve_images.restype = ctypes.c_int
        # initialize the camera
        result = self.camera_utils.init()
        self._create_capture_thread()

    def open(self):
        # Load the DLL
        pass

    def _create_capture_thread(self):
        """
        Creates and starts a daemon thread for continuously capturing frames from the camera.
        """
        self._camera_thread_running = True
        self._camera_thread = threading.Thread(target=self._processing_frame)
        self._camera_thread.daemon = True
        self._camera_thread.start()

    def _processing_frame(self):
        while self._camera_thread_running:
            timestamp = time.time_ns()
            img_1 = np.zeros((self.source_img_height, self.source_img_width), dtype=np.uint8)
            img_2 = np.zeros((self.source_img_height, self.source_img_width), dtype=np.uint8)
            img_1_ptr = img_1.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            img_2_ptr = img_2.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

            result = self.camera_utils.retrieve_images(img_1_ptr, img_2_ptr)

            if result == 1:
                # only use left image
                frame = cv2.cvtColor(img_1, cv2.COLOR_GRAY2RGB)
                # Resize the frame to 640x480 if necessary
                frame0 = cv2.resize(frame, (self.target_img_width, self.target_img_height))
                try:
                    with self.callback_and_param_lock:
                        if self.callback_and_params:
                            func, args, kwargs = self.callback_and_params
                            func(self.camera_running_state, timestamp, frame0, *args, **kwargs)
                except Exception as e:
                    Log.e(str(e))
                    # sys.exit(1)
            else:
                Log.d("Failed to retrieve images.")

    def close(self):
        Log.i("WebCam closed")

    def release(self):
        try:
            super().release()

            if self.camera_utils is not None:
                self.camera_utils.deinit()

                if self.dll_handle is not None:
                    kernel32.FreeLibrary(self.dll_handle)
                    self.dll_handle = None

                self.camera_utils = None

        except Exception as e:
            Log.e(f"Release failed: {str(e)}")


if __name__ == '__main__':
    # init pygame
    pygame.init()
    win = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)

    # init GazeFollower
    gf = GazeFollower(camera=PupilioCamera())
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

```

# Eye tracking config

More details about configuration can be found [here](gazefollower/misc/DefaultConfig.py).

```python
# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
from gazefollower import GazeFollower
from gazefollower.misc import DefaultConfig

config = DefaultConfig()
config.cali_mode = 9  # 5-, 9-, 13-point calibration

gf = GazeFollower(config=config)

```