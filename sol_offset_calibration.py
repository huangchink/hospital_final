# -*- coding: utf-8 -*-
"""
Sol Glasses Offset Calibration Module

This module provides functionality to calibrate angular offsets in Sol glasses
gaze data. When the built-in calibration fails, there's a consistent angular
offset that can be corrected by applying pitch/yaw rotations to the gaze
direction vector in the camera frame.
"""

import json
import numpy as np
import cv2
import pygame
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any


def dir_to_angles(d: np.ndarray) -> Tuple[float, float]:
    """
    Convert a unit direction vector to pitch and yaw angles.

    Args:
        d: Unit direction vector [x, y, z] in camera frame

    Returns:
        Tuple of (pitch, yaw) in radians
        - pitch: rotation around X-axis (vertical angle, positive = looking down)
        - yaw: rotation around Y-axis (horizontal angle, positive = looking right)
    """
    d = np.array(d).flatten()
    pitch = np.arcsin(-d[1])  # Y-axis: vertical (negative because Y is down in camera frame)
    yaw = np.arctan2(d[0], d[2])  # X/Z: horizontal
    return pitch, yaw


def angles_to_dir(pitch: float, yaw: float) -> np.ndarray:
    """
    Convert pitch and yaw angles to a unit direction vector.

    Args:
        pitch: Rotation around X-axis in radians
        yaw: Rotation around Y-axis in radians

    Returns:
        Unit direction vector [x, y, z]
    """
    x = np.sin(yaw) * np.cos(pitch)
    y = -np.sin(pitch)
    z = np.cos(yaw) * np.cos(pitch)
    return np.array([x, y, z])


def apply_angular_offset(gaze_dir_unit: np.ndarray, pitch_offset: float, yaw_offset: float) -> np.ndarray:
    """
    Apply angular offset correction to a gaze direction unit vector.

    The offset is applied by rotating the gaze vector by the negative of the
    measured error (to correct it). Rotation order: pitch (X-axis) then yaw (Y-axis).

    Args:
        gaze_dir_unit: Unit gaze direction vector [x, y, z] in camera frame
        pitch_offset: Pitch offset to correct (in radians)
        yaw_offset: Yaw offset to correct (in radians)

    Returns:
        Corrected unit gaze direction vector
    """
    gaze_dir_unit = np.array(gaze_dir_unit).flatten()

    # Rotation around X-axis (pitch correction) - apply negative to correct
    cos_p, sin_p = np.cos(-pitch_offset), np.sin(-pitch_offset)
    Rx = np.array([
        [1, 0, 0],
        [0, cos_p, -sin_p],
        [0, sin_p, cos_p]
    ])

    # Rotation around Y-axis (yaw correction) - apply negative to correct
    cos_y, sin_y = np.cos(-yaw_offset), np.sin(-yaw_offset)
    Ry = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])

    # Apply rotations: first pitch, then yaw
    corrected = Ry @ Rx @ gaze_dir_unit.reshape(3, 1)
    return corrected.flatten()


def load_sol_offset(username: str, calibration_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load Sol offset calibration from JSON file.

    Args:
        username: User identifier
        calibration_dir: Path to calibration_profiles directory

    Returns:
        Dictionary with offset data or None if not found
    """
    offset_file = calibration_dir / f"{username}_sol_offset.json"

    if not offset_file.exists():
        return None

    try:
        with open(offset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[Sol Offset] Failed to load offset file: {e}")
        return None


def save_sol_offset(username: str, calibration_dir: Path, offset_data: Dict[str, Any]) -> bool:
    """
    Save Sol offset calibration to JSON file.

    Args:
        username: User identifier
        calibration_dir: Path to calibration_profiles directory
        offset_data: Dictionary with offset data

    Returns:
        True if saved successfully, False otherwise
    """
    calibration_dir = Path(calibration_dir)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    offset_file = calibration_dir / f"{username}_sol_offset.json"

    try:
        with open(offset_file, 'w', encoding='utf-8') as f:
            json.dump(offset_data, f, indent=2, ensure_ascii=False)
        print(f"[Sol Offset] Saved offset to {offset_file}")
        return True
    except Exception as e:
        print(f"[Sol Offset] Failed to save offset file: {e}")
        return False


def clear_sol_offset(username: str, calibration_dir: Path) -> bool:
    """
    Delete Sol offset calibration file.

    Args:
        username: User identifier
        calibration_dir: Path to calibration_profiles directory

    Returns:
        True if deleted (or didn't exist), False on error
    """
    offset_file = Path(calibration_dir) / f"{username}_sol_offset.json"

    try:
        if offset_file.exists():
            offset_file.unlink()
            print(f"[Sol Offset] Cleared offset file: {offset_file}")
        return True
    except Exception as e:
        print(f"[Sol Offset] Failed to clear offset file: {e}")
        return False


# Calibration point positions (as fraction of screen)
CALIBRATION_POSITIONS = [
    ("center", 0.5, 0.5),
    ("upper-left", 0.15, 0.15),
    ("upper-right", 0.85, 0.15),
    ("bottom-right", 0.85, 0.85),
    ("bottom-left", 0.15, 0.85),
]


def get_calibration_positions(num_points: int, screen_width: int, screen_height: int) -> List[Dict]:
    """
    Get calibration target positions for the given number of points.

    Position order: Center -> Upper-left -> Upper-right -> Bottom-right -> Bottom-left -> (repeat)

    Args:
        num_points: Number of calibration points (1-10)
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels

    Returns:
        List of position dictionaries with 'name', 'x', 'y' keys
    """
    positions = []
    for i in range(num_points):
        idx = i % len(CALIBRATION_POSITIONS)
        name, fx, fy = CALIBRATION_POSITIONS[idx]
        # Add cycle number to name if repeating
        cycle = i // len(CALIBRATION_POSITIONS)
        if cycle > 0:
            name = f"{name}_{cycle + 1}"
        positions.append({
            'name': name,
            'x': int(fx * screen_width),
            'y': int(fy * screen_height),
            'index': i
        })
    return positions


class SolOffsetCalibrator:
    """
    Handles the Sol glasses offset calibration process.

    The calibration displays targets at known screen positions while the user
    looks at them. The tester captures gaze samples when the user is looking
    at the target, and the angular error between expected and measured gaze
    is computed.
    """

    def __init__(self,
                 sol_projector,
                 sol_gaze_queue: queue.Queue,
                 sol_scene_queue: queue.Queue,
                 screen_width_m: float,
                 aruco_markers_px: Dict,
                 aruco_imgs: Dict,
                 marker_container_size: int):
        """
        Initialize the calibrator.

        Args:
            sol_projector: ScreenProjector3D instance for pose estimation
            sol_gaze_queue: Queue receiving gaze data from Sol SDK
            sol_scene_queue: Queue receiving scene frames from Sol SDK
            screen_width_m: Physical screen width in meters
            aruco_markers_px: Dictionary of ArUco marker positions
            aruco_imgs: Dictionary of ArUco marker images
            marker_container_size: Size of marker container in pixels
        """
        self.sol_projector = sol_projector
        self.sol_gaze_queue = sol_gaze_queue
        self.sol_scene_queue = sol_scene_queue
        self.screen_width_m = screen_width_m
        self.aruco_markers_px = aruco_markers_px
        self.aruco_imgs = aruco_imgs
        self.marker_container_size = marker_container_size

        self.calibration_points = []
        self.current_offset = None
        self.running = False

    def run_calibration(self,
                       num_points: int,
                       target_image_path: Optional[str],
                       target_size_px: int,
                       user_screen_idx: int,
                       tester_screen_idx: int) -> Optional[Dict]:
        """
        Run the calibration process.

        Args:
            num_points: Number of calibration points (1-10)
            target_image_path: Optional path to target image
            target_size_px: Target size in pixels
            user_screen_idx: Display index for user screen (shows target)
            tester_screen_idx: Display index for tester screen (shows monitoring view)

        Returns:
            Dictionary with calibration results, or None if cancelled
        """
        # Initialize pygame displays
        pygame.init()

        # Get display info
        displays = []
        for i in range(pygame.display.get_num_displays()):
            try:
                info = pygame.display.get_desktop_sizes()[i]
                displays.append({'index': i, 'width': info[0], 'height': info[1]})
            except:
                pass

        if len(displays) < 2:
            print("[Sol Offset] Warning: Less than 2 displays detected. Using single display mode.")
            # Single display mode: split screen
            info = pygame.display.Info()
            user_screen = {'width': info.current_w, 'height': info.current_h}
            tester_screen = user_screen
            single_display = True
        else:
            user_screen = displays[min(user_screen_idx, len(displays) - 1)]
            tester_screen = displays[min(tester_screen_idx, len(displays) - 1)]
            single_display = (user_screen_idx == tester_screen_idx)

        # Screen dimensions
        user_w, user_h = user_screen['width'], user_screen['height']
        tester_w, tester_h = tester_screen['width'], tester_screen['height']

        # Create windows
        if single_display:
            # Single window mode - user screen fullscreen
            win = pygame.display.set_mode((user_w, user_h), pygame.FULLSCREEN)
            tester_win = None
        else:
            # Dual window mode
            import os
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"0,0"
            win = pygame.display.set_mode((user_w, user_h), pygame.FULLSCREEN)
            # Note: pygame doesn't easily support multiple windows
            # We'll render tester view as overlay or use a second process
            # For simplicity, use single window with split view
            tester_win = None
            single_display = True

        # Load target image
        target_surf = None
        if target_image_path and Path(target_image_path).exists():
            try:
                target_surf = pygame.image.load(target_image_path).convert_alpha()
                target_surf = pygame.transform.scale(target_surf, (target_size_px, target_size_px))
            except Exception as e:
                print(f"[Sol Offset] Failed to load target image: {e}")

        if target_surf is None:
            # Default target: red circle with crosshair
            target_surf = pygame.Surface((target_size_px, target_size_px), pygame.SRCALPHA)
            target_surf.fill((0, 0, 0, 0))
            center = target_size_px // 2
            pygame.draw.circle(target_surf, (255, 0, 0), (center, center), center - 2, 3)
            pygame.draw.line(target_surf, (255, 0, 0), (center - 10, center), (center + 10, center), 2)
            pygame.draw.line(target_surf, (255, 0, 0), (center, center - 10), (center, center + 10), 2)

        # Get calibration positions
        positions = get_calibration_positions(num_points, user_w, user_h)

        # Calibration state
        self.calibration_points = []
        self.running = True
        current_pos_idx = 0
        paused = False
        transition_start = None
        TRANSITION_DURATION = 1.0

        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 36)
        small_font = pygame.font.SysFont(None, 24)

        # Cache for latest gaze data
        latest_gaze = None
        latest_sol_frame = None
        current_gaze_pt = None  # Projected screen point

        while self.running and current_pos_idx < len(positions):
            pos = positions[current_pos_idx]

            # Handle events
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                    break
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_q:
                        self.running = False
                        break
                    elif ev.key == pygame.K_SPACE:
                        paused = not paused
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if paused and latest_gaze is not None:
                        # Record calibration point
                        point_data = self._record_calibration_point(
                            pos, latest_gaze, user_w, user_h
                        )
                        if point_data:
                            self.calibration_points.append(point_data)
                            print(f"[Sol Offset] Recorded point {current_pos_idx + 1}/{num_points}: "
                                  f"pitch_err={np.degrees(point_data['pitch_error_rad']):.2f}deg, "
                                  f"yaw_err={np.degrees(point_data['yaw_error_rad']):.2f}deg")
                            # Start transition to next position
                            transition_start = time.time()
                            paused = False
                            current_pos_idx += 1

            if not self.running:
                break

            # Get latest gaze data (drain queue)
            while True:
                try:
                    gaze = self.sol_gaze_queue.get_nowait()
                    if not paused:  # Only update if not paused
                        latest_gaze = gaze
                except queue.Empty:
                    break

            # Get latest scene frame (drain queue)
            while True:
                try:
                    frame = self.sol_scene_queue.get_nowait()
                    if not paused:
                        latest_sol_frame = frame
                except queue.Empty:
                    break

            # Process gaze for display
            if latest_gaze is not None and self.sol_projector.is_calibrated():
                current_gaze_pt = self._process_gaze_for_display(latest_gaze, user_w)

            # Check if in transition
            in_transition = (transition_start is not None and
                           time.time() - transition_start < TRANSITION_DURATION)

            # Render user screen
            win.fill((128, 128, 128))  # Gray background

            # Draw ArUco markers
            self._draw_aruco_markers(win)

            # Draw target (if not in transition or at valid position)
            if current_pos_idx < len(positions) and not in_transition:
                target_x = pos['x'] - target_size_px // 2
                target_y = pos['y'] - target_size_px // 2
                win.blit(target_surf, (target_x, target_y))

            # Draw gaze point (if available and not paused for cleaner capture)
            if current_gaze_pt and not paused:
                pygame.draw.circle(win, (0, 255, 0), current_gaze_pt, 15, 3)

            # Status bar
            status_text = f"Position: {pos['name']} ({current_pos_idx + 1}/{num_points})"
            if paused:
                status_text += " | PAUSED - Click on target center to record"
            else:
                status_text += " | Press SPACE to pause"
            status_text += " | Q to cancel"

            # Check ArUco status
            if not self.sol_projector.is_calibrated():
                status_text += " | WARNING: ArUco markers not detected!"

            text_surf = font.render(status_text, True, (255, 255, 255))
            pygame.draw.rect(win, (0, 0, 0), (0, user_h - 50, user_w, 50))
            win.blit(text_surf, (10, user_h - 40))

            # Draw sol camera view in corner (if available)
            if latest_sol_frame is not None:
                self._draw_sol_preview(win, latest_sol_frame, latest_gaze, user_w, user_h)

            pygame.display.flip()
            clock.tick(60)

        # Cleanup
        pygame.quit()

        if not self.running or len(self.calibration_points) == 0:
            return None

        # Calculate final offset
        result = self._compute_final_offset(user_w, user_h)
        return result

    def _draw_aruco_markers(self, surface):
        """Draw ArUco markers on the surface."""
        for mid, pos in self.aruco_markers_px.items():
            if mid in self.aruco_imgs:
                cv_img = self.aruco_imgs[mid]
                if len(cv_img.shape) == 2:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                elif cv_img.shape[2] == 4:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
                else:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                py_img = pygame.image.frombuffer(cv_img.tobytes(), cv_img.shape[1::-1], "RGB")
                surface.blit(py_img, (pos[0], pos[1]))

    def _draw_sol_preview(self, surface, sol_frame, gaze_data, screen_w, screen_h):
        """Draw Sol camera preview in corner of screen."""
        preview_w, preview_h = 320, 240

        # Get frame as numpy array
        if hasattr(sol_frame, 'img'):
            frame = sol_frame.img
        elif isinstance(sol_frame, np.ndarray):
            frame = sol_frame
        else:
            return

        # Resize
        frame_resized = cv2.resize(frame, (preview_w, preview_h))

        # Draw raw gaze_2d if available
        if gaze_data and hasattr(gaze_data, 'combined') and hasattr(gaze_data.combined, 'gaze_2d'):
            g2d = gaze_data.combined.gaze_2d
            # Scale to preview size
            orig_h, orig_w = frame.shape[:2]
            gx = int(g2d.x * preview_w / orig_w)
            gy = int(g2d.y * preview_h / orig_h)
            cv2.circle(frame_resized, (gx, gy), 8, (0, 0, 255), 2)

        # Convert to pygame
        if len(frame_resized.shape) == 2:
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2RGB)
        else:
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        py_surf = pygame.image.frombuffer(frame_resized.tobytes(), (preview_w, preview_h), "RGB")

        # Draw in bottom-right corner
        x = screen_w - preview_w - 10
        y = screen_h - preview_h - 60  # Above status bar
        pygame.draw.rect(surface, (0, 0, 0), (x - 2, y - 2, preview_w + 4, preview_h + 4), 2)
        surface.blit(py_surf, (x, y))

    def _process_gaze_for_display(self, gaze_data, screen_w: int) -> Optional[Tuple[int, int]]:
        """
        Process gaze data and project to screen coordinates.

        Returns:
            Screen coordinates (x, y) or None
        """
        try:
            if not hasattr(gaze_data, 'left_eye') or not hasattr(gaze_data, 'right_eye'):
                return None

            # Get gaze origin
            left_o = gaze_data.left_eye.gaze.origin
            right_o = gaze_data.right_eye.gaze.origin
            gaze_origin_mm = np.array([
                (left_o.x + right_o.x) / 2.0,
                (left_o.y + right_o.y) / 2.0,
                (left_o.z + right_o.z) / 2.0
            ])

            # Get gaze point
            g3d = gaze_data.combined.gaze_3d
            gaze_point_mm = np.array([g3d.x, g3d.y, g3d.z])

            # Compute direction
            gaze_direction_vec = gaze_point_mm - gaze_origin_mm
            norm = np.linalg.norm(gaze_direction_vec)

            if norm <= 0:
                return None

            gaze_direction_unit = gaze_direction_vec / norm
            gaze_origin_m = gaze_origin_mm / 1000.0

            # Project to screen
            screen_pt_m = self.sol_projector.project_gaze_to_screen(gaze_origin_m, gaze_direction_unit)
            if screen_pt_m is None:
                return None

            pix = self.sol_projector.physical_to_pixels(screen_pt_m, screen_w, self.screen_width_m)
            if pix:
                return (int(pix[0]), int(pix[1]))

        except Exception as e:
            print(f"[Sol Offset] Gaze processing error: {e}")

        return None

    def _record_calibration_point(self, position: Dict, gaze_data, screen_w: int, screen_h: int) -> Optional[Dict]:
        """
        Record a calibration point with measured and expected gaze directions.

        Args:
            position: Target position dictionary
            gaze_data: Sol SDK gaze data
            screen_w: Screen width in pixels
            screen_h: Screen height in pixels

        Returns:
            Calibration point data dictionary or None on error
        """
        try:
            # Get current pose
            rvec, tvec = self.sol_projector.get_current_pose()
            if rvec is None or tvec is None:
                print("[Sol Offset] No valid pose available")
                return None

            # Get measured gaze direction
            left_o = gaze_data.left_eye.gaze.origin
            right_o = gaze_data.right_eye.gaze.origin
            gaze_origin_mm = np.array([
                (left_o.x + right_o.x) / 2.0,
                (left_o.y + right_o.y) / 2.0,
                (left_o.z + right_o.z) / 2.0
            ])

            g3d = gaze_data.combined.gaze_3d
            gaze_point_mm = np.array([g3d.x, g3d.y, g3d.z])

            gaze_direction_vec = gaze_point_mm - gaze_origin_mm
            norm = np.linalg.norm(gaze_direction_vec)
            if norm <= 0:
                return None

            measured_gaze_dir = gaze_direction_vec / norm
            gaze_origin_m = gaze_origin_mm / 1000.0

            # Calculate expected gaze direction
            # Target position in screen coordinates (meters)
            px_to_m = self.screen_width_m / screen_w
            target_x_m = position['x'] * px_to_m
            target_y_m = position['y'] * px_to_m

            # Back-project to camera frame
            target_cam = self.sol_projector.back_project_screen_to_camera(
                np.array([target_x_m, target_y_m]), rvec, tvec
            )

            # Expected direction from eye to target
            expected_dir = target_cam - gaze_origin_m
            expected_norm = np.linalg.norm(expected_dir)
            if expected_norm <= 0:
                return None
            expected_gaze_dir = expected_dir / expected_norm

            # Calculate angular error
            measured_pitch, measured_yaw = dir_to_angles(measured_gaze_dir)
            expected_pitch, expected_yaw = dir_to_angles(expected_gaze_dir)

            pitch_error = measured_pitch - expected_pitch
            yaw_error = measured_yaw - expected_yaw

            return {
                'position_name': position['name'],
                'target_screen_px': [position['x'], position['y']],
                'measured_gaze_dir': measured_gaze_dir.tolist(),
                'expected_gaze_dir': expected_gaze_dir.tolist(),
                'gaze_origin_m': gaze_origin_m.tolist(),
                'pitch_error_rad': float(pitch_error),
                'yaw_error_rad': float(yaw_error),
                'timestamp': time.time()
            }

        except Exception as e:
            print(f"[Sol Offset] Failed to record calibration point: {e}")
            return None

    def _compute_final_offset(self, screen_w: int, screen_h: int) -> Dict:
        """
        Compute final offset by averaging all calibration points.

        Returns:
            Offset data dictionary ready for saving
        """
        if not self.calibration_points:
            return None

        pitch_errors = [p['pitch_error_rad'] for p in self.calibration_points]
        yaw_errors = [p['yaw_error_rad'] for p in self.calibration_points]

        avg_pitch = float(np.mean(pitch_errors))
        avg_yaw = float(np.mean(yaw_errors))

        return {
            'pitch_offset_rad': avg_pitch,
            'yaw_offset_rad': avg_yaw,
            'pitch_offset_deg': float(np.degrees(avg_pitch)),
            'yaw_offset_deg': float(np.degrees(avg_yaw)),
            'num_calibration_points': len(self.calibration_points),
            'calibration_points': self.calibration_points,
            'calibration_timestamp': datetime.now().isoformat(),
            'screen_width_px': screen_w,
            'screen_height_px': screen_h
        }
