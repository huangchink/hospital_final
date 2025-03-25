# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import pygame

from gazefollower import GazeFollower

gaze_follower = GazeFollower()

gaze_follower.preview()
gaze_follower.calibrate()

gaze_follower.start_sampling()
# your experiment code
gaze_follower.send_trigger(10)
pygame.time.wait(5)
# your experiment code
gaze_follower.stop_sampling()

gaze_follower.save_data("demo.csv")
gaze_follower.release()
