# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com


from pathlib import Path

import cv2
import pygame

from ..misc import FaceInfo


class CameraPreviewerUI:
    """
    Previewer for the camera device
    """

    def __init__(self):
        self._margin = 25

        self._face_image_size = (400, 400)
        self._eye_image_size = (400, 200)

        self._tip_text_size = (self._face_image_size[0] + 2 * self._margin, 40)

        self.camera_rect = pygame.Rect(self._margin * 2, self._margin * 2 + self._tip_text_size[1],
                                       *self._face_image_size)
        self.face_rect = pygame.Rect(self._margin * 4 + self._face_image_size[0],
                                     self._margin * 2 + self._tip_text_size[1], *self._face_image_size)
        self.right_eye_rect = pygame.Rect(self._margin * 2,
                                         self._face_image_size[1] + self._margin * 4 + self._tip_text_size[1] * 2,
                                         *self._eye_image_size)
        self.left_eye_rect = pygame.Rect(self._margin * 4 + self._face_image_size[0],
                                          self._face_image_size[1] + self._margin * 4 + self._tip_text_size[1] * 2,
                                          *self._eye_image_size)

        self.stop_button_rect = None

        # Define text rectangles based on the surfaces
        self._camera_text_rect = pygame.Rect(self._margin, self._margin,
                                             self._face_image_size[0] + self._margin, self._tip_text_size[1])

        self._face_text_rect = pygame.Rect(self._margin * 3 + self._face_image_size[0], self._margin,
                                           self._face_image_size[0] + 2 * self._margin, self._tip_text_size[1])

        self._left_eye_text_rect = pygame.Rect(self._margin,
                                               self._face_image_size[1] + self._margin * 3 + self._tip_text_size[1],
                                               self._face_image_size[0] + 2 * self._margin, self._tip_text_size[1])
        self._right_eye_text_rect = pygame.Rect(self._margin * 3 + self._face_image_size[0],
                                                self._face_image_size[1] + self._margin * 3 + self._tip_text_size[1],
                                                self._face_image_size[0] + 2 * self._margin, self._tip_text_size[1])

        # Update _rect_list using the calculated sizes and margins
        self._rect_list = [
            (self._margin, self._margin, self._face_image_size[0] + 2 * self._margin, self._tip_text_size[1]),  # 0
            (self._margin * 3 + self._face_image_size[0], self._margin, self._face_image_size[0] + 2 * self._margin,
             self._tip_text_size[1]),  # 1
            (self._margin, self._margin + self._tip_text_size[1], self._face_image_size[0] + 2 * self._margin,
             self._face_image_size[1] + 2 * self._margin),  # 2
            (self._margin * 3 + self._face_image_size[0], self._margin + self._tip_text_size[1],
             self._face_image_size[0] + 2 * self._margin, self._face_image_size[1] + 2 * self._margin),  # 3
            (self._margin, self._margin * 3 + self._tip_text_size[1] * 1 + self._face_image_size[1],
             self._eye_image_size[0] + 2 * self._margin, self._tip_text_size[1]),  # 4
            (self._margin * 3 + self._face_image_size[0],
             self._margin * 3 + self._tip_text_size[1] * 1 + self._face_image_size[1],
             self._eye_image_size[0] + 2 * self._margin, self._tip_text_size[1]),  # 5
            (self._margin,
             self._margin * 3 + self._tip_text_size[1] * 2 + self._face_image_size[1],
             self._eye_image_size[0] + 2 * self._margin, self._eye_image_size[1] + self._margin * 2),  # 6
            (self._margin * 3 + self._face_image_size[0],
             self._margin * 3 + self._tip_text_size[1] * 2 + self._face_image_size[1],
             self._eye_image_size[0] + 2 * self._margin, self._eye_image_size[1] + 2 * self._margin),  # 7
            (self._margin - 1, self._margin - 1,
             self._face_image_size[0] * 2 + 4 * self._margin,
             self._face_image_size[1] + self._eye_image_size[1] + 2 * self._tip_text_size[1] + 4 * self._margin,),  # 8
            # 8
        ]

        self._table_left_top_position = (self._margin * 7 + self.face_rect.width * 2, self._margin * 2)

        self._color_gradient_1 = (180, 180, 180)
        self._color_gradient_2 = (250, 200, 190)
        self._color_white = (255, 255, 255)
        self._color_black = (0, 0, 0)

        # constants for UI element
        self._table_row_height = 50  # Define row height here
        self._table_column_width = 180  # Width of each column
        self._table_width = 2 * self._table_column_width  # Total width for two columns
        self._table_even_color = (230, 230, 230)
        self._table_odd_color = (250, 250, 250)
        self._table_text_color = self._color_black
        self._table_line_color = (150, 150, 150)  # Darker gray for clearer line visibility

        self._layout_width = (self._face_image_size[0] + self._eye_image_size[0]
                              + 9 * self._margin + self._table_column_width * 2)
        self._layout_height = (self._tip_text_size[1] * 2 + self._face_image_size[1] + self._eye_image_size[1]
                               + 6 * self._margin)
        self._layout_start_x = None
        self._layout_start_y = None

        self.camera_gradient_surface = self._create_gradient_surface(self._face_image_size, self._color_gradient_1,
                                                                     self._color_gradient_2)
        self.face_gradient_surface = self._create_gradient_surface(self._face_image_size, self._color_gradient_1,
                                                                   self._color_gradient_2)
        self.left_eye_gradient_surface = self._create_gradient_surface(self._eye_image_size, self._color_gradient_1,
                                                                       self._color_gradient_2)
        self.right_eye_gradient_surface = self._create_gradient_surface(self._eye_image_size, self._color_gradient_1,
                                                                        self._color_gradient_2)

        self.camera_image = self.camera_gradient_surface
        self.face_image = self.face_gradient_surface
        self.left_eye_image = self.left_eye_gradient_surface
        self.right_eye_image = self.right_eye_gradient_surface

        self.font = pygame.font.SysFont('Microsoft YaHei', 20)
        self.small_font = pygame.font.SysFont('Microsoft YaHei', 16)
        self.small_font.set_bold(True)
        self.table_font = pygame.font.SysFont('Microsoft YaHei', 18)

        _current_dir = Path(__file__).parent.absolute()
        _package_dir = _current_dir.parent
        _icon_path = _package_dir.joinpath('res', 'image', 'gazefollower.png')
        _icon = pygame.transform.scale(pygame.image.load(str(_icon_path)), (50, 50))
        pygame.display.set_caption("GazeFollower Previewer")
        pygame.display.set_icon(_icon)  # 设置窗口图标

        self._button_color = (0, 123, 255)
        self._button_hover_color = (0, 86, 179)
        self._button_text = "Stop Previewing"
        self._button_size = (self._table_width + 2 * self._margin, 48)

        self.face_info_dict = {}
        self.running = True
        # self.main_loop()

    def _create_gradient_surface(self, size, color1, color2):
        """Creates a gradient surface."""
        surface = pygame.Surface(size)
        for y in range(size[1]):
            color = [
                color1[i] + (color2[i] - color1[i]) * y // size[1]
                for i in range(3)
            ]
            pygame.draw.line(surface, color, (0, y), (size[0], y))
        return surface

    def draw_text(self, text, font, color, surface, rect: pygame.Rect, align='center'):
        """Draws text with antialiasing for smoother look and a border around it."""
        text_obj = font.render(text, True, color)
        text_rect = text_obj.get_rect()
        if align == 'center':
            text_rect.center = rect.center
        elif align == 'left':
            text_rect.midleft = rect.midleft

        surface.blit(text_obj, text_rect)

    def draw_rounded_button(self, screen, rect):
        """Draws a button with rounded corners."""
        mouse_pos = pygame.mouse.get_pos()
        # button_x = rect.left + (rect.width - self._button_size[0]) // 2
        # button_y = rect.top + (rect.height - self._button_size[1]) // 2

        # button_rect = pygame.Rect(button_x, button_y, *self._button_size)
        button_color = self._button_hover_color if rect.collidepoint(mouse_pos) else self._button_color
        # pygame.draw.rect(screen, button_color, button_rect, border_radius=10)
        # pygame.draw.rect(screen, self._color_white, button_rect, 2, border_radius=10)  # Border
        pygame.draw.rect(screen, button_color, rect, border_radius=10)
        pygame.draw.rect(screen, self._color_white, rect, 2, border_radius=10)  # Border
        # pygame.draw.rect(screen, self._color_black, rect, 2)
        self.draw_text(self._button_text, self.small_font, self._color_white, screen, rect)

    def update_face_info(self, face_info):
        # Example face info dictionary; replace with actual data
        self.face_info_dict = face_info.to_dict()

    def draw_table(self, screen, data, start_pos):
        """Draws a table with alternating row colors, borders, and better alignment."""
        x, y = start_pos
        _num_rows = len(data)

        for i, (key, value) in enumerate(data.items()):
            row_rect = pygame.Rect(x, y + i * self._table_row_height, self._table_column_width * 2,
                                   self._table_row_height)
            row_color = self._table_even_color if i % 2 == 0 else self._table_odd_color
            pygame.draw.rect(screen, row_color, row_rect)

            self.draw_text(f'{key}', self.table_font, self._table_text_color, screen,
                           pygame.Rect(x + 5, y + i * self._table_row_height, self._table_column_width - 10,
                                       self._table_row_height), align='left')
            self.draw_text(f'{value}', self.table_font, self._table_text_color, screen,
                           pygame.Rect(x + self._table_column_width + 5, y + i * self._table_row_height,
                                       self._table_column_width - 10, self._table_row_height),
                           align='left')
        _rect = (x - self._margin, y - self._margin, self._table_width + 2 * self._margin,
                 self._table_row_height * _num_rows + 5 * self._margin)
        pygame.draw.rect(screen, self._color_black, _rect, 2)

        _button_boundary_width = self._table_width
        _button_boundary_height = self._margin * 2
        _button_boundary_x = x
        _button_boundary_y = self._table_row_height * _num_rows + y + self._margin
        self.stop_button_rect = pygame.Rect(_button_boundary_x, _button_boundary_y,
                                            _button_boundary_width, _button_boundary_height)
        self.draw_rounded_button(screen, self.stop_button_rect)

    def _shifting_layout(self, rect: pygame.Rect):
        return pygame.Rect(self._layout_start_x + rect.x,
                           self._layout_start_y + rect.y,
                           rect.width, rect.height)

    def draw(self, screen: pygame.Surface):
        """
        Draw content on the screen
        :param screen: PyGame screen
        :return:
        """
        # self.load_sample_images()
        _screen_width = screen.get_width()
        _screen_height = screen.get_height()

        self._layout_start_x = (_screen_width - self._layout_width) / 2
        self._layout_start_y = (_screen_height - self._layout_height) / 2

        # shifting the layout
        self.camera_rect = self._shifting_layout(self.camera_rect)
        self.face_rect = self._shifting_layout(self.face_rect)
        self.left_eye_rect = self._shifting_layout(self.left_eye_rect)
        self.right_eye_rect = self._shifting_layout(self.right_eye_rect)

        self._camera_text_rect = self._shifting_layout(self._camera_text_rect)
        self._face_text_rect = self._shifting_layout(self._face_text_rect)
        self._left_eye_text_rect = self._shifting_layout(self._left_eye_text_rect)
        self._right_eye_text_rect = self._shifting_layout(self._right_eye_text_rect)
        _tmp_x, _tmp_y = self._table_left_top_position
        self._table_left_top_position = (self._layout_start_x + _tmp_x, self._layout_start_y + _tmp_y)

        self._rect_list = [self._shifting_layout(pygame.Rect(i)) for i in self._rect_list]

        self.update_face_info(FaceInfo())

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if (event.type == pygame.MOUSEBUTTONDOWN and self.stop_button_rect is not None
                        and self.stop_button_rect.collidepoint(event.pos)):
                    self.running = False

            screen.fill(self._color_white)  # Light gray background

            # Draw images
            self.draw_image(screen, self.camera_image, self.camera_rect)
            self.draw_image(screen, self.face_image, self.face_rect)
            self.draw_image(screen, self.left_eye_image, self.left_eye_rect)
            self.draw_image(screen, self.right_eye_image, self.right_eye_rect)

            # Draw labels
            self.draw_text('Camera Image', self.font, self._color_black, screen, self._camera_text_rect)
            self.draw_text('Face Image', self.font, self._color_black, screen, self._face_text_rect)
            self.draw_text('Left Eye Image', self.font, self._color_black, screen, self._left_eye_text_rect)
            self.draw_text('Right Eye Image', self.font, self._color_black, screen, self._right_eye_text_rect)

            # Draw face info table
            self.draw_table(screen, self.face_info_dict, self._table_left_top_position)

            # Draw grid
            self.draw_grid_rect(screen)

            pygame.display.flip()

    def draw_grid_rect(self, screen):
        for _rect in self._rect_list:
            pygame.draw.rect(screen, self._color_black, _rect, 1)

    def update_images(self, camera_image, face_image, left_eye_image, right_eye_image):
        """
        update the previewer
        :param camera_image: raw image from camera
        :param face_image: facial image
        :param left_eye_image: left eye image
        :param right_eye_image: right eye image
        :return:
        """
        if camera_image is not None:
            camera_image = cv2.rotate(camera_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.camera_image = pygame.surfarray.make_surface(camera_image)
        else:
            self.camera_image = self.camera_gradient_surface

        if face_image is not None:
            face_image = cv2.rotate(face_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.face_image = pygame.surfarray.make_surface(face_image)
        else:
            self.face_image = self.face_gradient_surface

        if left_eye_image is not None:
            left_eye_image = cv2.rotate(left_eye_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.left_eye_image = pygame.surfarray.make_surface(left_eye_image)
        else:
            self.left_eye_image = self.left_eye_gradient_surface

        if right_eye_image is not None:
            right_eye_image = cv2.rotate(right_eye_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.right_eye_image = pygame.surfarray.make_surface(right_eye_image)
        else:
            self.right_eye_image = self.right_eye_gradient_surface

    def draw_image(self, screen: pygame.Surface, image, rect: pygame.Rect):
        """Draws an image with a subtle shadow effect while maintaining the original aspect ratio."""

        original_width, original_height = image.get_size()
        target_width, target_height = rect.width, rect.height

        aspect_ratio = original_width / original_height

        if target_width / target_height > aspect_ratio:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)

        scaled_image = pygame.transform.smoothscale(image, (new_width, new_height))

        x = rect.x + (target_width - new_width) // 2
        y = rect.y + (target_height - new_height) // 2

        screen.blit(scaled_image, (x, y))


if __name__ == '__main__':
    CameraPreviewerUI()
