#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import copy
import math
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pygame

from ..misc import GazeInfo


class CalibrationUI:
    def __init__(self, camera_position, screen_size, calibration_percentage_points, validation_percentage_points):
        """
        Initializes the Calibration UI.

        :param camera_position: The (x, y) position of the camera.
        :param screen_size: The (width, height) of the screen.
        :param calibration_percentage_points: List of (x, y) calibration points as fractions of the screen size.
        """

        self._color_white = (255, 255, 255)
        self._color_red = (255, 0, 0)
        self._color_black = (0, 0, 0)
        self._color_blue = (0, 0, 255)
        self.camera_position = camera_position
        self.screen_width, self.screen_height = screen_size
        self.calibration_points = [(round(x * self.screen_width), round(y * self.screen_height)) for x, y in
                                   calibration_percentage_points]
        self.validation_points = [(round(x * self.screen_width), round(y * self.screen_height)) for x, y in
                                  validation_percentage_points]
        self.error_bar_color = (0, 255, 0)  # Green color for the error bar
        self.error_bar_thickness = 2        # Thickness of the error bar lin

        # Initialize the font
        self.guidance_font = pygame.font.SysFont('Microsoft YaHei', 28)

        # Initialize the mixer and load sound
        pygame.mixer.init()
        _audio_path = Path(__file__).parent.parent.absolute() / 'res/audio/beep.wav'
        self.feedback_sound = pygame.mixer.Sound(_audio_path)  # Replace with the path to your sound file

        # Load the arrow images
        _left_arrow_path = Path(__file__).parent.parent.absolute() / 'res/image/left_arrow.png'

        self._arrow_image_size = 36
        # Load and resize the left arrow image
        self.left_arrow_image = pygame.transform.smoothscale(pygame.image.load(_left_arrow_path),
                                                             (self._arrow_image_size, self._arrow_image_size))

        # Create and resize the right arrow image by rotating the left arrow image
        self.right_arrow_image = pygame.transform.rotate(self.left_arrow_image, 180)

        self._new_session()

    def generate_calibration_directions(self):
        num_points = len(self.calibration_points)
        # Generate lists for directions
        self.calibration_directions = ['left'] * (num_points // 2) + ['right'] * (num_points - num_points // 2)
        # Shuffle the list with a fixed seed for reproducibility
        np.random.seed(2024)
        np.random.shuffle(self.calibration_directions)

    def generate_validation_directions(self):
        num_points = len(self.validation_points)
        # Generate lists for directions
        self.validation_directions = ['left'] * (num_points // 2) + ['right'] * (num_points - num_points // 2)
        # Shuffle the list with a fixed seed for reproducibility
        np.random.seed(912)
        np.random.shuffle(self.validation_directions)

    def set_calibration_points(self, calibration_points):
        """

        :param calibration_points:
        :return:
        """
        self.calibration_points = [(int(x * self.screen_width), int(y * self.screen_height)) for x, y in
                                   calibration_points]
        self.generate_calibration_directions()

    def set_validation_points(self, validation_points):
        self.validation_points = [(int(x * self.screen_width), int(y * self.screen_height)) for x, y in
                                  validation_points]
        self.generate_validation_directions()

    def draw_breathing_effect(self, screen, center, outer_radius: int, inner_radius: int, elapsed_time: float):
        """Draws a breathing light effect with a deeper color gradient towards the inner circle."""

        pulse_period = 4  # seconds for one full pulse cycle
        pulse_amplitude = outer_radius - inner_radius  # Maximum expansion relative to inner circle
        if elapsed_time > pulse_period:
            return

        # Calculate the pulse offset to animate the gradient effect
        pulse_offset = math.sin(elapsed_time / pulse_period * math.pi / 2)  # Oscillates between 0 and 1
        current_radius = inner_radius + pulse_amplitude * (1 - pulse_offset)  # Decreases from max to min

        # Create a surface for the gradient effect with transparency
        gradient_surface = pygame.Surface((2 * current_radius, 2 * current_radius), pygame.SRCALPHA)

        # Use a higher resolution surface for anti-aliasing effect
        scale_factor = 4  # Increase the resolution by this factor
        high_res_radius = int(current_radius * scale_factor)
        high_res_surface = pygame.Surface((2 * high_res_radius, 2 * high_res_radius), pygame.SRCALPHA)

        # Draw concentric circles with varying intensity to create a gradient effect
        for i in range(high_res_radius, int(inner_radius * scale_factor), -2 * scale_factor):
            color_intensity = int(255 * ((i - inner_radius * scale_factor) / (pulse_amplitude * scale_factor)))
            gradient_color = (255, color_intensity, color_intensity, 128)  # Red gradient with varying alpha

            pygame.draw.circle(high_res_surface, gradient_color, (high_res_radius, high_res_radius), i)

        # Scale down the high-resolution surface to the original size to achieve anti-aliasing
        gradient_surface = pygame.transform.smoothscale(high_res_surface,
                                                        (2 * int(current_radius), 2 * int(current_radius)))

        # Draw the gradient surface on the screen
        screen.blit(gradient_surface, (center[0] - current_radius, center[1] - current_radius))

        # # Draw the inner white circle separately to ensure it stays the correct size
        pygame.draw.circle(screen, (255, 255, 255), center, inner_radius)

    def draw_arrows(self, screen, center: Tuple[int, int], direction: str):
        """Draws left or right arrows based on the direction."""
        if direction == 'left':
            screen.blit(self.left_arrow_image, (
                center[0] - self.left_arrow_image.get_width() // 2,
                center[1] - self.left_arrow_image.get_height() // 2))
        elif direction == 'right':
            screen.blit(self.right_arrow_image, (center[0] - self.left_arrow_image.get_width() // 2,
                                                 center[1] - self.right_arrow_image.get_height() // 2))

    def draw_guidance_text(self, screen):
        """Draws the guidance text for the user."""
        instruction_text = [
            "An arrow will appear on the screen.",
            "Press F when the left arrow appears and J when the right arrow appears.",
            "Each arrow will be displayed for two seconds before responding.",
            "Press Space to start."
        ]

        text_surfaces = [self.guidance_font.render(line, True, self._color_black) for line in instruction_text]

        total_text_height = sum(text_surface.get_height() for text_surface in text_surfaces) + (
                len(instruction_text) - 1) * 10
        start_y = (self.screen_height - total_text_height) // 2

        for i, text_surface in enumerate(text_surfaces):
            text_rect = text_surface.get_rect(
                center=(self.screen_width // 2, start_y + i * (text_surface.get_height() + 10)))
            screen.blit(text_surface, text_rect)

    def _new_session(self):
        self.running = True
        self.current_point_index = 0  # Start at the first calibration point
        self.start_time = None  # Initialize the start time
        self.sound_played = False  # Flag to track if the sound has been played
        self.calibration_directions = []
        self.validation_directions = []
        self.generate_calibration_directions()
        self.generate_validation_directions()
        # Initialize list to record responses
        self.responses = []
        self.point_showing = False
        self.point_elapsed_time = 0
        self.gaze_info = GazeInfo()

    def draw_calibration(self, screen):
        self.draw(screen, "calibration")

    def draw_validation(self, screen):
        self.draw(screen, "validation")

    def draw(self, screen, draw_type="calibration"):
        """Draws each calibration point one at a time and advances when space is pressed or after minimum display
        time."""
        self._new_session()

        is_calibration = draw_type == "calibration"

        min_display_time = 1.5  # Minimum display time in seconds
        # Show initial guidance screen
        while self.running:
            screen.fill(self._color_white)
            self.draw_guidance_text(screen)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.running = False
                    pygame.quit()
                    return pd.DataFrame(self.responses,
                                        columns=['point_x', 'point_y', 'arrow_direction',
                                                 'response_key', 'response_time'])
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.running = False
                    self.start_time = time.time()  # Initialize the start time

        self.running = True
        self.point_showing = True
        points_need_to_draw = self.calibration_points if is_calibration else self.validation_points
        directions_need_to_draw = self.calibration_directions if is_calibration else self.validation_directions
        while self.running:
            current_time = time.time()  # Get the current time
            self.point_elapsed_time = current_time - self.start_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    # Record response if F or J key is pressed
                    if event.key == pygame.K_f or event.key == pygame.K_j:
                        response_key = 'F' if event.key == pygame.K_f else 'J'
                        response_time = time.time() - self.start_time
                        if self.current_point_index < len(points_need_to_draw):
                            self.responses.append({
                                'point_x': points_need_to_draw[self.current_point_index][0],
                                'point_y': points_need_to_draw[self.current_point_index][1],
                                'arrow_direction':  directions_need_to_draw[self.current_point_index],
                                'response_key': response_key,
                                'response_time': response_time
                            })

                        # Advance to the next calibration point if minimum display time has passed
                        if response_time >= min_display_time:
                            self.current_point_index += 1
                            if self.current_point_index >= len(points_need_to_draw):
                                self.running = False  # Exit if all points are shown
                            self.start_time = time.time()  # Reset the start time for the next point
                            self.sound_played = False  # Reset sound flag
                        else:
                            if not self.sound_played:
                                self.feedback_sound.play()
                                self.sound_played = True

            # Fill the screen with a white background
            screen.fill(self._color_white)

            # Draw the current calibration point
            if self.current_point_index < len(points_need_to_draw):
                current_point = points_need_to_draw[self.current_point_index]
                # Decide on the direction of the arrow
                direction =  directions_need_to_draw[self.current_point_index]
                # Draw the breathing effect
                self.draw_breathing_effect(screen, current_point, self._arrow_image_size // 2 * 3,
                                           self._arrow_image_size // 2,
                                           self.point_elapsed_time)
                # Draw the arrows inside the calibration point
                self.draw_arrows(screen, current_point, direction)

                # Draw error bar
                if self.gaze_info.status and self.point_elapsed_time > 0.3:
                    self.draw_error_bar(screen, current_point, self.gaze_info.filtered_gaze_coordinates)


            # Update the display
            pygame.display.flip()

            # Check if the sound has finished playing
            if self.sound_played and not pygame.mixer.get_busy():
                self.sound_played = False  # Reset sound flag to allow replay if needed

        self.point_showing = False
        return pd.DataFrame(self.responses,
                            columns=['point_x', 'point_y', 'arrow_direction', 'response_key',
                                     'response_time'])

    def validation_sample_subscriber(self, face_info, gaze_info, *args, **kwargs):
        """

        :param face_info:
        :param gaze_info:
        :param args:
        :param kwargs:
        :return:
        """
        self.face_info = face_info
        self.gaze_info = gaze_info

    def calculate_d3_metric(self, window=8, stride=1):
        """

        :param window:
        :param stride:
        :return:
        """
        pass

    def draw_error_bar(self, screen, current_point, gaze_coordinates):
        # Draws a green error bar (line) between the current point and gaze coordinates
        pygame.draw.line(
            screen,
            self.error_bar_color,
            current_point,
            gaze_coordinates,
            self.error_bar_thickness
        )

# Example usage:
if __name__ == "__main__":
    pygame.init()
    # Example screen size and calibration points
    screen_size = (1920, 1080)  # Replace with the actual screen size
    camera_position = (0, 0)  # Example camera position, not used in this demo
    calibration_points = [
        (0.5, 0.5), (0.5, 0.08), (0.08, 0.5), (0.92, 0.5), (0.5, 0.92), (0.08, 0.08),
        (0.92, 0.08), (0.08, 0.92), (0.92, 0.92), (0.25, 0.25), (0.75, 0.25),
        (0.25, 0.75), (0.75, 0.75), (0.5, 0.5)
    ]

    screen = pygame.display.set_mode(screen_size, pygame.FULLSCREEN)
    pygame.display.set_caption("Calibration UI")
    # Initialize and run the Calibration UI
    calibration_ui = CalibrationUI(camera_position, screen_size, calibration_points, copy.deepcopy(calibration_points))
    responses = calibration_ui.draw(screen)

    # Save responses to a CSV file
    # responses.to_csv('calibration_responses.csv', index=False)
