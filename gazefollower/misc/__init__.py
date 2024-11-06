# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

from .Enumeration import *
from .FaceInfo import *
from .GazeInfo import *
from .Recorder import *

def clip_patch(frame, rect):
    """
    Clip a region from the frame based on the provided rectangle.

    :param frame: The input image frame as a NumPy array (height x width x channels).
    :param rect: A tuple (x, y, w, h) defining the rectangle to clip.
    :return: A NumPy array representing the clipped image patch, or None if the rectangle is invalid.
    """
    x, y, w, h = rect

    # Check for valid rectangle dimensions
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return None

    # Check if the rectangle is within the frame bounds
    if x >= frame.shape[1] or y >= frame.shape[0]:
        return None

    x_end = min(x + w, frame.shape[1])
    y_end = min(y + h, frame.shape[0])

    # Clip the region from the frame
    clipped_patch = frame[y:y_end, x:x_end].copy()

    return clipped_patch
