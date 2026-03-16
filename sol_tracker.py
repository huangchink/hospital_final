import cv2
import numpy as np
import asyncio
import threading
import queue
import time

# --- SDK Import ---
try:
    from ganzin.sol_sdk.asynchronous.async_client import AsyncClient, recv_gaze, recv_video
    from ganzin.sol_sdk.common_models import Camera
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    class AsyncClient: pass

# --- Constants ---
BORDER_PIXEL_WIDTH = 15
# Margin from screen edge for ArUco markers
# Must be > status bar height (50px) + buffer to ensure markers are fully visible
SCREEN_MARGIN_PIXELS = 60
# Bottom margin is larger: the monitor bezel merges with ArUco black border,
# eating the white quiet zone and preventing detection.
SCREEN_MARGIN_BOTTOM_PIXELS = 90

def create_calibration_assets(screen_width, screen_height, aruco_dict, config):
    marker_container_size = config['marker_pattern_size'] + (BORDER_PIXEL_WIDTH * 2)
    marker_positions_px, individual_marker_images = {}, {}
    marker_id_counter = 0
    if config['marker_k'] < 2 or config['marker_n'] < 2: return None, None
    margin_top = SCREEN_MARGIN_PIXELS
    margin_bottom = SCREEN_MARGIN_BOTTOM_PIXELS
    spacing_x = (screen_width - 2*SCREEN_MARGIN_PIXELS - marker_container_size) / (config['marker_k'] - 1)
    spacing_y = (screen_height - margin_top - margin_bottom - marker_container_size) / (config['marker_n'] - 1)
    all_points = []
    for i in range(config['marker_k']):
        all_points.append((int(SCREEN_MARGIN_PIXELS + i*spacing_x), margin_top))
        all_points.append((int(SCREEN_MARGIN_PIXELS + i*spacing_x), int(screen_height - margin_bottom - marker_container_size)))
    for i in range(1, config['marker_n'] - 1):
        all_points.append((SCREEN_MARGIN_PIXELS, int(margin_top + i*spacing_y)))
        all_points.append((int(screen_width - SCREEN_MARGIN_PIXELS - marker_container_size), int(margin_top + i*spacing_y)))
    unique_points = sorted(list(set(all_points)))
    for pos in unique_points:
        marker_positions_px[marker_id_counter] = pos; marker_id_counter += 1
    border_offset = (marker_container_size - config['marker_pattern_size']) // 2
    for marker_id, position in marker_positions_px.items():
        white_canvas = np.full((marker_container_size, marker_container_size, 3), 255, dtype=np.uint8)
        marker_pattern = cv2.aruco.generateImageMarker(aruco_dict, marker_id, config['marker_pattern_size'], borderBits=1)
        marker_pattern_bgr = cv2.cvtColor(marker_pattern, cv2.COLOR_GRAY2BGR)
        white_canvas[border_offset:border_offset+config['marker_pattern_size'], border_offset:border_offset+config['marker_pattern_size']] = marker_pattern_bgr
        marker_bgra = cv2.cvtColor(white_canvas, cv2.COLOR_BGR2BGRA)
        marker_bgra[:, :, 3] = 255
        individual_marker_images[marker_id] = marker_bgra
    print(f"總共預先產生了 {len(marker_positions_px)} 個 Markers 影像。")
    return marker_positions_px, individual_marker_images

class ScreenProjector3D:
    def __init__(self, camera_matrix, dist_coeffs, aruco_dict, smoothing_factor):
        self.camera_matrix, self.dist_coeffs, self.aruco_dict = camera_matrix, dist_coeffs, aruco_dict
        self.smoothing_factor = smoothing_factor
        self.aruco_params = cv2.aruco.DetectorParameters()
        # Pose of Screen relative to Camera (rvec, tvec)
        self.smoothed_rvec, self.smoothed_tvec = None, None
        self.is_pose_valid = False

        # [OPT] Background ArUco Detection
        self.pose_lock = threading.Lock()
        self.pose_queue = queue.Queue(maxsize=2)  # Small queue for latest frames
        self.pose_thread = None
        self.pose_running = False

        # [2D Gaze Mapping] Homography from camera image to screen pixels
        self.image_to_screen_homography = None
        self.homography_valid = False
        self.last_homography_marker_count = 0  # Track marker count for stability
        self.last_homography_update_time = 0  # Timestamp of last valid homography
        self.homography_persistence_sec = 0.5  # Keep using homography for this long after markers lost
        self.screen_width_px = None  # Set during start_background_detection
        self.screen_height_px = None
        self.detected_marker_ids = []  # List of currently detected marker IDs (for visualization)

        # [Marker FIFO Buffer] Accumulate marker detections over multiple frames
        # Each entry: marker_id -> (image_point, timestamp, frame_num)
        self.marker_detection_buffer = {}  # marker_id -> list of (img_center, timestamp, frame_num)
        self.marker_buffer_max_age_sec = 0.5  # Use detections from last 0.5 seconds
        self.marker_buffer_max_frames = 10  # Keep last N frames of detections per marker
        self.marker_frame_counter = 0  # Global frame counter

        # [Initial Homography Quality] Require minimum frames/markers before accepting first homography
        self.min_frames_for_init_homography = 10  # Wait at least N frames before accepting initial homography
        self.min_markers_for_init_homography = 8  # Require at least N unique markers for initial homography
        self.homography_from_cache = False  # Track if homography was restored from cache

        # [2D Gaze Smoothing] Smooth the 2D gaze output to reduce jitter
        self.smoothed_gaze_2d_screen = None
        self.gaze_2d_smoothing_factor = 0.15  # Default, will be overridden by set_gaze_2d_smoothing_factor()
        self.gaze_2d_frame_count = 0  # Debug frame counter

        # [2D Gaze Offset] SVR-based offset correction model
        self.gaze_2d_offset_model = None

    def is_calibrated(self):
        with self.pose_lock:
            return self.is_pose_valid

    def start_background_detection(self, marker_physical_size_m, marker_screen_positions_px, marker_container_size, screen_width_px, screen_height_px, screen_width_m):
        """Start background ArUco detection thread"""
        if self.pose_running:
            return
        # Store screen dimensions for bounds checking
        self.screen_width_px = screen_width_px
        self.screen_height_px = screen_height_px
        self.pose_running = True
        self.pose_thread = threading.Thread(
            target=self._pose_detection_worker,
            args=(marker_physical_size_m, marker_screen_positions_px, marker_container_size, screen_width_px, screen_height_px, screen_width_m),
            daemon=True
        )
        self.pose_thread.start()
        print("[ScreenProjector3D] Background ArUco detection started")

    def stop_background_detection(self, clear_homography=True):
        """Stop background ArUco detection thread

        Args:
            clear_homography: If True, clears the homography. If False, preserves it
                              for use by subsequent operations (e.g., calibration).
        """
        self.pose_running = False
        if self.pose_thread and self.pose_thread.is_alive():
            self.pose_thread.join(timeout=2.0)
        # Clear the buffer to prevent stale data on next start
        with self.pose_lock:
            self.marker_detection_buffer.clear()
            if clear_homography:
                self.image_to_screen_homography = None
                self.homography_valid = False
            self.smoothed_gaze_2d_screen = None
        # Clear the queue
        while not self.pose_queue.empty():
            try:
                self.pose_queue.get_nowait()
            except queue.Empty:
                break
        print("[ScreenProjector3D] Background ArUco detection stopped")

    def submit_frame_for_pose(self, image):
        """Submit frame for background pose detection (non-blocking)"""
        try:
            # Overwrite old frame if queue is full
            if self.pose_queue.full():
                try:
                    self.pose_queue.get_nowait()
                except queue.Empty:
                    pass
            self.pose_queue.put_nowait(image)
        except queue.Full:
            pass  # Drop frame if still full

    def _pose_detection_worker(self, marker_physical_size_m, marker_screen_positions_px, marker_container_size, screen_width_px, screen_height_px, screen_width_m):
        """Background thread for ArUco marker detection"""
        frame_count = 0
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 50

        while self.pose_running:
            try:
                image = self.pose_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_count += 1
            try:
                # [FIX] Validate image before processing
                if image is None:
                    continue
                if not hasattr(image, 'shape') or len(image.shape) < 2:
                    continue
                if image.shape[0] < 10 or image.shape[1] < 10:
                    continue

                # Perform expensive ArUco detection
                result, detected_ids = self._update_pose_internal(image, marker_physical_size_m, marker_screen_positions_px, marker_container_size, screen_width_px, screen_height_px, screen_width_m)

                # Reset error counter on success
                consecutive_errors = 0

                # Print status every 200 frames (reduced logging)
                if frame_count % 200 == 0:
                    status = "CALIBRATED" if self.is_pose_valid else "NOT_CALIBRATED"
                    num_markers = len(detected_ids) if detected_ids else 0
                    print(f"[ArUco] Frame {frame_count}: {status}, markers={num_markers}")
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 5 or consecutive_errors % 100 == 0:
                    import traceback
                    print(f"[ArUco] Error in frame {frame_count}: {e}")
                    if consecutive_errors == 1:
                        traceback.print_exc()

                # Safety check: if too many consecutive errors, pause briefly
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"[ArUco] WARNING: {consecutive_errors} consecutive errors, pausing 1s")
                    time.sleep(1.0)
                    consecutive_errors = 0
                # Continue processing to avoid thread death

    def update_pose(self, image, marker_physical_size_m, marker_screen_positions_px, marker_container_size, screen_width_px, screen_height_px, screen_width_m):
        """Legacy synchronous update_pose - calls internal method"""
        return self._update_pose_internal(image, marker_physical_size_m, marker_screen_positions_px, marker_container_size, screen_width_px, screen_height_px, screen_width_m)

    def _update_pose_internal(self, image, marker_physical_size_m, marker_screen_positions_px, marker_container_size, screen_width_px, screen_height_px, screen_width_m):
        corners, ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_params)

        # Store detected marker IDs for visualization (even if < 4)
        detected_ids_list = ids.flatten().tolist() if ids is not None else []
        with self.pose_lock:
            self.detected_marker_ids = detected_ids_list
            self.marker_frame_counter += 1
            current_frame = self.marker_frame_counter

        current_time = time.time()
        px_to_m = screen_width_m / screen_width_px

        # [FIFO Buffer] Update marker detection buffer with current detections
        current_frame_markers = {}  # marker_id -> (img_center, screen_center)
        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(detected_ids_list):
                if marker_id not in marker_screen_positions_px:
                    continue
                current_corners = corners[i][0]  # shape (4, 2)
                marker_center_img = np.mean(current_corners, axis=0)  # Average of 4 corners

                container_top_left_px = marker_screen_positions_px[marker_id]
                cx_px = container_top_left_px[0] + marker_container_size / 2.0
                cy_px = container_top_left_px[1] + marker_container_size / 2.0

                current_frame_markers[marker_id] = (marker_center_img, np.array([cx_px, cy_px]))

                # Update FIFO buffer for this marker
                with self.pose_lock:
                    if marker_id not in self.marker_detection_buffer:
                        self.marker_detection_buffer[marker_id] = []
                    self.marker_detection_buffer[marker_id].append((marker_center_img, current_time, current_frame))
                    # Trim to max frames
                    if len(self.marker_detection_buffer[marker_id]) > self.marker_buffer_max_frames:
                        self.marker_detection_buffer[marker_id] = self.marker_detection_buffer[marker_id][-self.marker_buffer_max_frames:]

        # [CURRENT FRAME ONLY] Use only current frame markers for homography
        # This ensures head movement immediately updates the homography
        # (Previous buffered approach caused stale data when head tilted)
        src_points_2d = []  # Marker centers in camera image coordinates
        dst_points_2d = []  # Marker centers in screen pixel coordinates
        markers_used_ids = []

        for marker_id, (img_center, screen_center) in current_frame_markers.items():
            src_points_2d.append(img_center)
            dst_points_2d.append(screen_center)
            markers_used_ids.append(marker_id)

        # Clean up old entries from buffer (separate pass to avoid iteration issues)
        with self.pose_lock:
            expired_markers = []
            marker_ids_to_check = list(self.marker_detection_buffer.keys())
            for marker_id in marker_ids_to_check:
                detections = self.marker_detection_buffer.get(marker_id, [])
                # Remove detections older than max_age
                filtered = [(c, t, f) for (c, t, f) in detections
                           if current_time - t <= self.marker_buffer_max_age_sec]
                if filtered:
                    self.marker_detection_buffer[marker_id] = filtered
                else:
                    expired_markers.append(marker_id)
            for marker_id in expired_markers:
                self.marker_detection_buffer.pop(marker_id, None)

        # For 3D pose, still need current frame detections (PnP requires precise corners)
        if ids is None or len(ids) < 4:
            self.is_pose_valid = False
            # Don't immediately invalidate homography - let persistence handle it
            with self.pose_lock:
                self.homography_valid = False
            # Still continue to try homography with buffered markers
        else:
            # Build 3D pose from current frame detections only
            image_points = []
            object_points = []

            for i, marker_id in enumerate(detected_ids_list):
                if marker_id not in marker_screen_positions_px:
                    continue
                current_corners = corners[i][0]
                container_top_left_px = marker_screen_positions_px[marker_id]
                cx_px = container_top_left_px[0] + marker_container_size / 2.0
                cy_px = container_top_left_px[1] + marker_container_size / 2.0
                cx_m = cx_px * px_to_m
                cy_m = cy_px * px_to_m

                half_size_m = marker_physical_size_m / 2.0
                obj_pts = np.array([
                    [cx_m - half_size_m, cy_m - half_size_m, 0],
                    [cx_m + half_size_m, cy_m - half_size_m, 0],
                    [cx_m + half_size_m, cy_m + half_size_m, 0],
                    [cx_m - half_size_m, cy_m + half_size_m, 0]
                ], dtype=np.float32)

                object_points.append(obj_pts)
                image_points.append(current_corners)

            if object_points:
                object_points = np.vstack(object_points)
                image_points = np.vstack(image_points)
                retval, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE)

                if retval:
                    with self.pose_lock:
                        if self.smoothed_rvec is None:
                            self.smoothed_rvec, self.smoothed_tvec = rvec, tvec
                        else:
                            self.smoothed_rvec = self.smoothing_factor * rvec + (1 - self.smoothing_factor) * self.smoothed_rvec
                            self.smoothed_tvec = self.smoothing_factor * tvec + (1 - self.smoothing_factor) * self.smoothed_tvec
                        self.is_pose_valid = True

        # Check if buffered markers have enough coverage for homography
        if len(src_points_2d) < 4:
            return (self.smoothed_rvec, self.smoothed_tvec) if self.is_pose_valid else None, detected_ids_list

        # [2D Gaze Mapping] Compute homography from buffered marker positions
        src_pts = np.array(src_points_2d, dtype=np.float32)
        dst_pts = np.array(dst_points_2d, dtype=np.float32)

        # Check if markers span enough area in BOTH X and Y directions
        # This prevents degenerate homography from markers on a single row/column
        dst_x_range = np.max(dst_pts[:, 0]) - np.min(dst_pts[:, 0])
        dst_y_range = np.max(dst_pts[:, 1]) - np.min(dst_pts[:, 1])
        min_span_ratio = 0.3  # Require at least 30% of screen dimension
        min_x_span = screen_width_px * min_span_ratio
        min_y_span = screen_height_px * min_span_ratio

        if dst_x_range < min_x_span or dst_y_range < min_y_span:
            # Markers don't span enough area - skip this frame
            # This prevents degenerate homography from single-row markers
            # Debug: show marker Y distribution
            y_coords = dst_pts[:, 1]
            top_count = np.sum(y_coords < screen_height_px * 0.3)
            bottom_count = np.sum(y_coords > screen_height_px * 0.7)
            print(f"[Homography] SKIPPED: markers span {dst_x_range:.0f}x{dst_y_range:.0f}px "
                  f"(need {min_x_span:.0f}x{min_y_span:.0f}px), "
                  f"top_row={top_count}, bottom_row={bottom_count}")
            return (self.smoothed_rvec, self.smoothed_tvec) if self.is_pose_valid else None, detected_ids_list

        # Since all markers are placed by us on known screen positions, use least-squares
        # when we have good coverage. RANSAC with small threshold rejects too many valid points.
        # Try multiple methods and pick the best one based on inlier coverage.
        current_marker_count = len(src_points_2d)

        # Method 1: Regular least-squares fit (no outlier rejection) - best when markers are accurate
        H_ls, _ = cv2.findHomography(src_pts, dst_pts, 0)  # method=0 means regular DLT

        # Method 2: LMEDS - robust to outliers without requiring threshold parameter
        H_lmeds, status_lmeds = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)

        # Method 3: RANSAC with larger threshold (10px) - more tolerant of lens distortion
        H_ransac, status_ransac = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

        # Evaluate each homography and pick the best one
        def eval_homography(H, src, dst, screen_w, screen_h):
            """Evaluate homography quality - returns (num_good_points, avg_error)"""
            if H is None:
                return 0, float('inf')
            good_count = 0
            total_error = 0
            for sp, dp in zip(src, dst):
                pt = H @ np.array([sp[0], sp[1], 1.0])
                if abs(pt[2]) < 1e-6:
                    continue
                px, py = pt[0] / pt[2], pt[1] / pt[2]
                error = np.sqrt((px - dp[0])**2 + (py - dp[1])**2)
                total_error += error
                if error < 50:  # Within 50px is considered good
                    good_count += 1
            avg_error = total_error / len(src) if len(src) > 0 else float('inf')
            return good_count, avg_error

        # Evaluate all methods
        ls_good, ls_err = eval_homography(H_ls, src_pts, dst_pts, screen_width_px, screen_height_px)
        lmeds_good, lmeds_err = eval_homography(H_lmeds, src_pts, dst_pts, screen_width_px, screen_height_px)
        ransac_good, ransac_err = eval_homography(H_ransac, src_pts, dst_pts, screen_width_px, screen_height_px)

        # Pick the best method based on number of good points and average error
        best_method = "LS"
        H = H_ls
        status = None
        best_good = ls_good

        if lmeds_good > best_good or (lmeds_good == best_good and lmeds_err < ls_err):
            best_method = "LMEDS"
            H = H_lmeds
            status = status_lmeds
            best_good = lmeds_good

        if ransac_good > best_good or (ransac_good == best_good and ransac_err < lmeds_err):
            best_method = "RANSAC"
            H = H_ransac
            status = status_ransac
            best_good = ransac_good

        if H is not None:
            # Count inliers (for LS, all points are "inliers")
            if status is not None:
                inlier_count = int(np.sum(status))
                inlier_mask = status.flatten() == 1
            else:
                inlier_count = len(src_points_2d)
                inlier_mask = np.ones(len(dst_pts), dtype=bool)

            inlier_dst_pts = dst_pts[inlier_mask]
            if len(inlier_dst_pts) >= 4:
                inlier_x_range = np.max(inlier_dst_pts[:, 0]) - np.min(inlier_dst_pts[:, 0])
                inlier_y_range = np.max(inlier_dst_pts[:, 1]) - np.min(inlier_dst_pts[:, 1])
                # Require inliers to span at least 20% of screen in both directions
                min_inlier_x = screen_width_px * 0.2
                min_inlier_y = screen_height_px * 0.2
                if inlier_x_range < min_inlier_x or inlier_y_range < min_inlier_y:
                    # Inliers don't span enough area - likely degenerate
                    # Debug: show inlier Y distribution
                    inlier_y_coords = inlier_dst_pts[:, 1]
                    inlier_top = np.sum(inlier_y_coords < screen_height_px * 0.3)
                    inlier_bottom = np.sum(inlier_y_coords > screen_height_px * 0.7)
                    print(f"[Homography] REJECTED ({best_method}): inliers span {inlier_x_range:.0f}x{inlier_y_range:.0f}px "
                          f"(need {min_inlier_x:.0f}x{min_inlier_y:.0f}px), "
                          f"inlier_top={inlier_top}, inlier_bottom={inlier_bottom}")
                    H = None  # Mark as invalid

        if H is not None:
            with self.pose_lock:
                # Require at least 8 inliers for stability (was 4, too low)
                if inlier_count >= 8:
                    if self.image_to_screen_homography is None:
                        current_frame_count = len(detected_ids_list)
                        # For fresh start (no cache), require minimum frames and markers
                        if not self.homography_from_cache:
                            if self.marker_frame_counter < self.min_frames_for_init_homography:
                                if self.marker_frame_counter % 50 == 0:
                                    print(f"[Homography] Waiting for initial stability: frame {self.marker_frame_counter}/{self.min_frames_for_init_homography}")
                                H = None  # Don't accept yet
                            elif current_marker_count < self.min_markers_for_init_homography:
                                if self.marker_frame_counter % 50 == 0:
                                    print(f"[Homography] Waiting for more markers: {current_marker_count}/{self.min_markers_for_init_homography}")
                                H = None  # Don't accept yet

                        if H is not None:
                            print(f"[Homography DEBUG] INIT ({best_method}): markers={current_marker_count}, inliers={inlier_count}, marker_ids={markers_used_ids}")
                            self._debug_print_homography(H, src_pts, dst_pts, status, screen_width_px, screen_height_px)
                            self.image_to_screen_homography = H
                            self.last_homography_marker_count = current_marker_count
                            self.homography_valid = True
                            self.last_homography_update_time = time.time()
                            self.homography_from_cache = False  # Now we have a computed homography
                    else:
                        # Check how much the homography changed
                        h_diff = np.abs(H - self.image_to_screen_homography).max()

                        # Since we now use only CURRENT FRAME markers (not buffered),
                        # large changes likely represent real head movement, not detection errors.
                        # Use faster smoothing to be responsive to head tilt.
                        if h_diff > 1000:
                            # Extreme change - likely detection error, reject
                            pass
                        elif h_diff > 200:
                            # Large change (head movement) - fast update
                            smooth_factor = 0.5
                            self.image_to_screen_homography = smooth_factor * H + (1 - smooth_factor) * self.image_to_screen_homography
                            self.last_homography_update_time = time.time()
                        elif h_diff > 50:
                            # Moderate change - moderate smoothing
                            smooth_factor = 0.3
                            self.image_to_screen_homography = smooth_factor * H + (1 - smooth_factor) * self.image_to_screen_homography
                            self.last_homography_update_time = time.time()
                        else:
                            # Small change - normal smoothing for stability
                            self.image_to_screen_homography = self.smoothing_factor * H + (1 - self.smoothing_factor) * self.image_to_screen_homography
                            self.last_homography_update_time = time.time()

                        self.last_homography_marker_count = current_marker_count
                        self.homography_valid = True
                elif inlier_count >= 6:
                    # 6-7 inliers: still update but with more smoothing
                    if self.image_to_screen_homography is not None:
                        h_diff = np.abs(H - self.image_to_screen_homography).max()
                        if h_diff < 200:
                            # Moderate smoothing for fewer inliers
                            smooth_factor = 0.15
                            self.image_to_screen_homography = smooth_factor * H + (1 - smooth_factor) * self.image_to_screen_homography
                            self.last_homography_update_time = time.time()
                    else:
                        # No homography yet - don't accept low-inlier homography for fresh start
                        # Only accept if we have cached homography and need to fall back
                        if self.homography_from_cache:
                            # We have a cache, so we can accept low-inlier updates
                            src_center = np.mean(src_pts, axis=0)
                            test_pt = H @ np.array([src_center[0], src_center[1], 1.0])
                            if abs(test_pt[2]) > 1e-6:
                                test_x = test_pt[0] / test_pt[2]
                                test_y = test_pt[1] / test_pt[2]
                                if 0 <= test_x < screen_width_px and 0 <= test_y < screen_height_px:
                                    print(f"[Homography DEBUG] INIT (low inliers, from cache): markers={current_marker_count}, inliers={inlier_count}")
                                    self._debug_print_homography(H, src_pts, dst_pts, status, screen_width_px, screen_height_px)
                                    self.image_to_screen_homography = H
                                    self.last_homography_marker_count = current_marker_count
                                    self.homography_valid = True
                                    self.last_homography_update_time = time.time()
                                    self.homography_from_cache = False
                        # For fresh start, skip low-inlier initial homography - wait for better one

        # Return current pose state
        return (self.smoothed_rvec, self.smoothed_tvec) if self.is_pose_valid else None, detected_ids_list

    def project_gaze_to_screen(self, gaze_origin_cam_m, gaze_direction_cam_unit):
        """
        將 Camera Frame 的 Gaze Ray 投射到 Screen Frame 並計算與 Z=0 平面的交點
        """
        # [OPT] Thread-safe pose read
        with self.pose_lock:
            if not self.is_pose_valid:
                return None
            rvec = self.smoothed_rvec.copy() if self.smoothed_rvec is not None else None
            tvec = self.smoothed_tvec.copy() if self.smoothed_tvec is not None else None

        if rvec is None or tvec is None:
            return None

        R_screen_to_cam, _ = cv2.Rodrigues(rvec)
        R_cam_to_screen = R_screen_to_cam.T
        t_screen_to_cam = tvec
        
        # 轉換 Origin
        gaze_origin_screen_m = R_cam_to_screen @ (gaze_origin_cam_m.reshape(3,1) - t_screen_to_cam)
        gaze_origin_screen_m = gaze_origin_screen_m.flatten()
        
        # 轉換 Direction
        gaze_dir_screen_unit = R_cam_to_screen @ gaze_direction_cam_unit.reshape(3,1)
        gaze_dir_screen_unit = gaze_dir_screen_unit.flatten()
        
        if abs(gaze_dir_screen_unit[2]) < 1e-6: # 平行於螢幕
            return None
            
        t = -gaze_origin_screen_m[2] / gaze_dir_screen_unit[2]
        
        if t < 0: # 交點在背後
            return None
            
        intersection_point_screen_m = gaze_origin_screen_m + t * gaze_dir_screen_unit
        
        return intersection_point_screen_m # (x, y, 0) in meters

    def physical_to_pixels(self, point_screen_m, screen_width_px, screen_width_m):
        """
        將 Screen Frame 的物理座標 (meters) 轉換為 Pixel 座標
        """
        if point_screen_m is None:
            return None
            
        px_to_m = screen_width_m / screen_width_px
        
        x_px = point_screen_m[0] / px_to_m
        y_px = point_screen_m[1] / px_to_m
        
        return (int(x_px), int(y_px))

    # Helper method for specialized use cases (like recording) that might need 3d point projection logic exposed
    def project_gaze_to_3d_point(self, gaze_origin_m, gaze_direction_unit, plane):
        # Plane defined by (point, normal)
        plane_point, plane_normal = plane
        
        # Flatten inputs to ensure they are 1D arrays
        gaze_direction_unit = np.array(gaze_direction_unit).flatten()
        plane_point = np.array(plane_point).flatten()
        plane_normal = np.array(plane_normal).flatten()

        # Check if parallel
        denom = np.dot(gaze_direction_unit, plane_normal)
        if abs(denom) < 1e-6:
            return None
        
        t = np.dot(plane_point - gaze_origin_m, plane_normal) / denom
        if t < 0:
            return None
            
        return gaze_origin_m + t * gaze_direction_unit

    def map_3d_to_2d_pixel(self, point_3d, H):
        # Legacy method support if H is passed, but we prefer rvec/tvec
        # This is a placeholder or we can implement if we know what H is.
        # But given the refactoring plan, we will add a new robust method.
        pass

    def camera_point_to_screen_pixels(self, point_camera_m, rvec, tvec, screen_width_px, screen_width_m):
        """
        Convert a 3D point in Camera Frame to Screen Pixels using provided pose (rvec, tvec).
        """
        if point_camera_m is None: return None

        # T_cam_to_screen = T_screen_to_cam ^ -1
        R_screen_to_cam, _ = cv2.Rodrigues(rvec)
        R_cam_to_screen = R_screen_to_cam.T
        t_screen_to_cam = tvec

        # P_screen = R^T * (P_cam - t)
        point_screen_m = R_cam_to_screen @ (point_camera_m.reshape(3,1) - t_screen_to_cam.reshape(3,1))
        point_screen_m = point_screen_m.flatten()

        return self.physical_to_pixels(point_screen_m, screen_width_px, screen_width_m)

    def get_current_pose(self):
        """
        Thread-safe retrieval of current pose for calibration.

        Returns:
            Tuple of (rvec, tvec) copies, or (None, None) if pose is not valid
        """
        with self.pose_lock:
            if not self.is_pose_valid or self.smoothed_rvec is None:
                return None, None
            return self.smoothed_rvec.copy(), self.smoothed_tvec.copy()

    def back_project_screen_to_camera(self, screen_point_m, rvec, tvec):
        """
        Convert screen point (in meters, Z=0 plane) to camera frame coordinates.

        This is the inverse of project_gaze_to_screen - given a point on the screen,
        compute its 3D position in the camera coordinate frame.

        Args:
            screen_point_m: 2D point on screen in meters [x, y] (Z=0 assumed)
            rvec: Rotation vector from screen to camera
            tvec: Translation vector from screen to camera

        Returns:
            3D point in camera frame coordinates (meters)
        """
        # Screen point in 3D (Z=0 plane)
        screen_3d = np.array([screen_point_m[0], screen_point_m[1], 0], dtype=np.float64)

        # Transform from screen frame to camera frame
        # P_cam = R * P_screen + t
        R_screen_to_cam, _ = cv2.Rodrigues(rvec)
        point_camera_m = R_screen_to_cam @ screen_3d.reshape(3, 1) + tvec.reshape(3, 1)

        return point_camera_m.flatten()

    def _debug_print_homography(self, H, src_pts, dst_pts, status, screen_w, screen_h):
        """Print detailed homography debug info."""
        print(f"[Homography DEBUG] === HOMOGRAPHY DETAILS ===")
        print(f"[Homography DEBUG] Screen size: {screen_w}x{screen_h}")
        print(f"[Homography DEBUG] H matrix:\n{H}")

        # Print marker correspondences
        print(f"[Homography DEBUG] Marker correspondences (inlier marked with *):")
        for i, (src, dst) in enumerate(zip(src_pts, dst_pts)):
            is_inlier = status[i][0] if status is not None else True
            marker = "*" if is_inlier else " "
            print(f"  {marker} cam({src[0]:.1f}, {src[1]:.1f}) -> screen({dst[0]:.1f}, {dst[1]:.1f})")

        # Test transformation at camera center and corners
        camera_w, camera_h = 1328, 1200  # Sol scene camera resolution
        test_points = [
            (camera_w // 2, camera_h // 2, "camera_center"),
            (0, 0, "camera_top_left"),
            (camera_w, 0, "camera_top_right"),
            (camera_w, camera_h, "camera_bottom_right"),
            (0, camera_h, "camera_bottom_left"),
        ]

        print(f"[Homography DEBUG] Test transformations:")
        for cx, cy, name in test_points:
            pt = np.array([cx, cy, 1.0])
            dst = H @ pt
            if abs(dst[2]) > 1e-6:
                sx, sy = dst[0] / dst[2], dst[1] / dst[2]
                in_bounds = 0 <= sx <= screen_w and 0 <= sy <= screen_h
                status_str = "OK" if in_bounds else "OUT_OF_BOUNDS"
                print(f"  {name}: cam({cx}, {cy}) -> screen({sx:.1f}, {sy:.1f}) [{status_str}]")
            else:
                print(f"  {name}: cam({cx}, {cy}) -> INVALID (w={dst[2]:.6f})")
        print(f"[Homography DEBUG] ========================")

    def is_homography_valid(self, strict=False):
        """Check if 2D gaze mapping homography is available.

        Args:
            strict: If True, require currently detecting markers (for calibration).
                    If False (default), use cached homography even if markers lost (for preview).

        For preview mode (strict=False):
            Once a homography is established, it remains valid indefinitely.
            This allows gaze tracking to continue even when markers are temporarily occluded.

        For calibration mode (strict=True):
            Requires active marker detection for accurate calibration.
        """
        with self.pose_lock:
            if strict:
                # Strict mode: must be actively detecting markers
                return self.homography_valid and self.image_to_screen_homography is not None
            else:
                # Lenient mode: use any available homography (for preview)
                return self.image_to_screen_homography is not None

    def get_detected_marker_ids(self):
        """Get list of currently detected marker IDs (for visualization)."""
        with self.pose_lock:
            return self.detected_marker_ids.copy()

    def get_homography(self):
        """
        Get the current image-to-screen homography matrix.

        Returns:
            3x3 homography matrix, or None if not available
        """
        with self.pose_lock:
            if self.image_to_screen_homography is None:
                return None
            return self.image_to_screen_homography.copy()

    def set_homography(self, H):
        """
        Set the image-to-screen homography matrix externally.

        This is useful for restoring a previously computed homography,
        e.g., when starting a calibration session after a preview.

        Args:
            H: 3x3 numpy array, or None to clear
        """
        with self.pose_lock:
            if H is not None:
                self.image_to_screen_homography = H.copy()
                self.homography_valid = True
                self.homography_from_cache = True  # Mark as restored from cache
                self.last_homography_update_time = time.time()
                print(f"[ScreenProjector3D] External homography set")
            else:
                self.image_to_screen_homography = None
                self.homography_valid = False
                self.homography_from_cache = False

    def get_screen_to_image_homography(self):
        """
        Get the inverse homography that maps screen pixels to camera image coordinates.

        This is the inverse of image_to_screen_homography, useful for projecting
        known screen positions (like calibration targets) back to camera image space.

        Returns:
            3x3 homography matrix, or None if not available
        """
        with self.pose_lock:
            if self.image_to_screen_homography is None:
                return None
            try:
                H_inv = np.linalg.inv(self.image_to_screen_homography)
                return H_inv.copy()
            except np.linalg.LinAlgError:
                return None

    def project_screen_to_image(self, screen_point_px):
        """
        Project a screen pixel position to camera image coordinates using the inverse homography.

        Args:
            screen_point_px: (x, y) in screen pixels

        Returns:
            (x, y) in camera image coordinates, or None if homography not available
        """
        H_inv = self.get_screen_to_image_homography()
        if H_inv is None:
            return None

        # Apply inverse homography: H_inv @ [x, y, 1]^T = [x', y', w']^T
        src_pt = np.array([screen_point_px[0], screen_point_px[1], 1.0], dtype=np.float64)
        dst_pt = H_inv @ src_pt

        if abs(dst_pt[2]) < 1e-6:
            return None

        x_img = dst_pt[0] / dst_pt[2]
        y_img = dst_pt[1] / dst_pt[2]

        return (x_img, y_img)

    def reset_gaze_2d_smoothing(self):
        """Reset the 2D gaze smoothing state. Call this when starting a new session."""
        with self.pose_lock:
            self.smoothed_gaze_2d_screen = None

    def set_gaze_2d_smoothing_factor(self, factor):
        """
        Set the smoothing factor for 2D gaze output.
        Lower values = more smoothing (less responsive but smoother)
        Higher values = less smoothing (more responsive but jittery)
        Recommended range: 0.1 to 0.5
        """
        with self.pose_lock:
            self.gaze_2d_smoothing_factor = max(0.05, min(1.0, factor))
            print(f"[ScreenProjector3D] 2D gaze smoothing factor set to: {self.gaze_2d_smoothing_factor} (input: {factor})")

    def set_gaze_2d_offset_model(self, model):
        """
        Set the SVR-based 2D gaze offset correction model.

        Args:
            model: Sol2DOffsetModel instance, or None to disable offset correction
        """
        with self.pose_lock:
            self.gaze_2d_offset_model = model
            if model is not None and model.is_trained:
                print(f"[ScreenProjector3D] 2D gaze offset model set ({len(model.calibration_points)} points)")
            else:
                print("[ScreenProjector3D] 2D gaze offset model cleared")

    def apply_gaze_2d_offset(self, gaze_2d_point):
        """
        Apply the 2D offset correction to a gaze_2d point.

        Args:
            gaze_2d_point: (x, y) - raw gaze_2d from Sol SDK

        Returns:
            (x, y) - corrected gaze_2d, or original if no model set
        """
        with self.pose_lock:
            if self.gaze_2d_offset_model is None or not self.gaze_2d_offset_model.is_trained:
                return gaze_2d_point
            model = self.gaze_2d_offset_model
            frame_count = self.gaze_2d_frame_count

        # Debug output every 100 frames
        debug = (frame_count % 100 == 1)
        return model.predict(gaze_2d_point, debug=debug)

    def project_gaze_2d_to_screen(self, gaze_2d_point, screen_width=None, screen_height=None, apply_smoothing=True, apply_offset=True):
        """
        Project 2D gaze point (in scene camera image coordinates) to screen pixels
        using the computed homography.

        This is the 2D gaze mapping method - directly maps the 2D gaze point from
        the Sol glasses scene camera to screen pixel coordinates using a homography
        computed from ArUco marker positions.

        Args:
            gaze_2d_point: Tuple (x, y) - gaze point in scene camera image coordinates
            screen_width: Optional screen width for bounds checking
            screen_height: Optional screen height for bounds checking
            apply_smoothing: Whether to apply temporal smoothing (default True)
            apply_offset: Whether to apply SVR-based offset correction (default True)

        Returns:
            Tuple (x, y) - screen pixel coordinates, or None if homography not available
        """
        with self.pose_lock:
            # Use lenient check: work as long as we have ANY homography (for preview)
            if self.image_to_screen_homography is None:
                return None
            H = self.image_to_screen_homography.copy()
            prev_smoothed = self.smoothed_gaze_2d_screen
            # Use stored screen dimensions if not provided
            if screen_width is None:
                screen_width = self.screen_width_px
            if screen_height is None:
                screen_height = self.screen_height_px

        # Validate input
        if gaze_2d_point is None:
            return None
        try:
            gx, gy = float(gaze_2d_point[0]), float(gaze_2d_point[1])
            if not (np.isfinite(gx) and np.isfinite(gy)):
                return None
        except (TypeError, ValueError, IndexError):
            return None

        # Apply SVR-based offset correction to gaze_2d BEFORE homography
        if apply_offset:
            gaze_2d_point = self.apply_gaze_2d_offset(gaze_2d_point)
            if gaze_2d_point is None:
                return None

        # Apply homography transformation
        # H @ [x, y, 1]^T = [x', y', w']^T  =>  screen_pt = [x'/w', y'/w']
        try:
            src_pt = np.array([gaze_2d_point[0], gaze_2d_point[1], 1.0], dtype=np.float64)
            dst_pt = H @ src_pt

            if abs(dst_pt[2]) < 1e-6:
                return None

            x_screen = dst_pt[0] / dst_pt[2]
            y_screen = dst_pt[1] / dst_pt[2]

            if not (np.isfinite(x_screen) and np.isfinite(y_screen)):
                return None
        except Exception:
            return None

        raw_pt = (x_screen, y_screen)

        # Check if point is within screen bounds (with margin for edge tracking)
        margin = 100
        is_on_screen = True
        if screen_width is not None and screen_height is not None:
            is_on_screen = (-margin <= raw_pt[0] < screen_width + margin and
                           -margin <= raw_pt[1] < screen_height + margin)

        # If point is off-screen, return None (don't render, don't update smoothing)
        if not is_on_screen:
            return None

        # Apply temporal smoothing
        if apply_smoothing:
            with self.pose_lock:
                if self.gaze_2d_smoothing_factor >= 0.95:
                    # No smoothing requested
                    self.smoothed_gaze_2d_screen = raw_pt
                    result = raw_pt
                elif self.smoothed_gaze_2d_screen is None:
                    # First point
                    self.smoothed_gaze_2d_screen = raw_pt
                    result = raw_pt
                else:
                    # Check if previous was off-screen (contaminated) - if so, reset
                    prev_on_screen = True
                    if screen_width is not None and screen_height is not None:
                        prev_on_screen = (-margin <= self.smoothed_gaze_2d_screen[0] < screen_width + margin and
                                         -margin <= self.smoothed_gaze_2d_screen[1] < screen_height + margin)

                    if not prev_on_screen:
                        # Reset to current (coming back from off-screen)
                        self.smoothed_gaze_2d_screen = raw_pt
                        result = raw_pt
                    else:
                        # Normal exponential smoothing
                        sf = self.gaze_2d_smoothing_factor
                        sx = sf * raw_pt[0] + (1 - sf) * self.smoothed_gaze_2d_screen[0]
                        sy = sf * raw_pt[1] + (1 - sf) * self.smoothed_gaze_2d_screen[1]
                        self.smoothed_gaze_2d_screen = (sx, sy)
                        result = self.smoothed_gaze_2d_screen
        else:
            result = raw_pt

        return (int(result[0]), int(result[1]))

class SolConnector:
    """
    A pure Python class to manage Ganzin Sol SDK connection and streaming.
    Decoupled from PyQt.
    """
    def __init__(self, ip, port, gaze_queue, scene_queue):
        self.ip = ip
        self.port = port
        self.gaze_queue = gaze_queue
        self.scene_queue = scene_queue
        self.stop_event = threading.Event()
        self._scene_active = threading.Event()  # Controls scene stream on/off
        self._scene_active.set()  # Active by default
        self._worker_thread = None  # Set after thread creation for join on stop

    def pause_scene_stream(self):
        """Pause the scene (video) stream to avoid Sol SDK native crashes during idle."""
        self._scene_active.clear()
        print("[SolConnector] Scene stream paused")

    def resume_scene_stream(self):
        """Resume the scene (video) stream for ArUco detection."""
        self._scene_active.set()
        print("[SolConnector] Scene stream resumed")

    async def _gaze_stream_loop(self, ac):
        print("[SolConnector] Gaze stream loop started.")
        gaze_count = 0
        try:
            async for data in recv_gaze(ac):
                if self.stop_event.is_set(): break
                gaze_count += 1
                if gaze_count <= 3 or gaze_count % 500 == 0:
                    print(f"[SolConnector] Gaze #{gaze_count} received: {type(data).__name__}")
                try:
                    self.gaze_queue.put_nowait(data)
                except queue.Full:
                    # Drop oldest item and add new one
                    try:
                        self.gaze_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.gaze_queue.put_nowait(data)
                    except queue.Full:
                        pass
        except asyncio.CancelledError:
            pass  # Normal cancellation
        except Exception as e:
            print(f"[SolConnector] Gaze stream error: {e}")
            import traceback; traceback.print_exc()
        print(f"[SolConnector] Gaze stream loop finished. Total received: {gaze_count}")

    async def _scene_stream_loop(self, ac):
        print("[SolConnector] Scene stream loop started.")
        while not self.stop_event.is_set():
            # Wait until scene stream is activated
            while not self._scene_active.is_set():
                if self.stop_event.is_set():
                    print("[SolConnector] Scene stream loop finished (stop during pause).")
                    return
                await asyncio.sleep(0.2)

            # Run scene stream until paused or stopped
            try:
                async for frame in recv_video(ac, Camera.SCENE):
                    if self.stop_event.is_set() or not self._scene_active.is_set():
                        break
                    try: self.scene_queue.put_nowait(frame)
                    except queue.Full: pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[SolConnector] Scene stream error: {e}")
                if self.stop_event.is_set():
                    break
                await asyncio.sleep(0.5)  # Brief pause before retry

        print("[SolConnector] Scene stream loop finished.")

    async def run_session(self, on_connect=None, on_fail=None):
        """
        Main async session loop.
        on_connect: callback(message, camera_params, time_offset_ms)
        on_fail: callback(error_message)
        """
        if not SDK_AVAILABLE:
            if on_fail: on_fail("SDK未安裝。")
            return

        try:
            async with AsyncClient(self.ip, self.port) as ac:
                print("[SolConnector] 正在獲取設備狀態與相機參數...")
                status_task = ac.get_status()
                params_task = ac.get_scene_camera_param()
                time_sync_task = ac.run_time_sync(10)
                
                results = await asyncio.gather(status_task, params_task, time_sync_task, return_exceptions=True)

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        raise Exception(f"初始化任務 {i} 失敗: {result}")
                    
                status_resp, params_resp, time_sync_resp = results
                camera_params = {
                    'cam_matrix': np.array(params_resp.result.camera_param.intrinsic),
                    'dist_coeffs': np.array(params_resp.result.camera_param.distort)
                }
                time_offset_ms = time_sync_resp.time_offset.mean
                
                print(f"相機參數獲取成功。時間偏移量: {time_offset_ms:.2f} ms")
                message = f"連線成功 | 設備: {status_resp.device_name}，時間差 {time_offset_ms} ms"

                if on_connect:
                    on_connect(message, camera_params, int(time_offset_ms))

                streaming_tasks = asyncio.gather(
                    self._gaze_stream_loop(ac),
                    self._scene_stream_loop(ac)
                )
                
                while not self.stop_event.is_set():
                    await asyncio.sleep(0.1)
                
                print("[SolConnector] 收到停止信號，正在取消串流任務...")
                streaming_tasks.cancel()
                try:
                    await streaming_tasks
                except asyncio.CancelledError:
                    print("[SolConnector] 串流任務已成功取消。")

        except Exception as e:
            print(f"[SolConnector] Session failed: {e}")
            if on_fail:
                on_fail(f"操作失敗: {e}")
            # Ensure we don't crash the loop, but maybe re-raise if needed. 
            # For now, just logging and callback.

    def stop(self):
        print("[SolConnector] 正在發送停止信號...")
        self.stop_event.set()
        # Wait for the worker thread to finish so native resources are fully released
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3.0)
            if self._worker_thread.is_alive():
                print("[SolConnector] Warning: worker thread did not exit within timeout")
            else:
                print("[SolConnector] Worker thread finished cleanly.")
