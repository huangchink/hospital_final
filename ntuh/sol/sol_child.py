"""Isolated Sol scene worker - runs in a CHILD process (multiprocessing 'spawn').

Owns the single Sol connection + the only ArUco-running ScreenProjector3D, so the SDK's
uncatchable native H.264 scene-decode crash kills ONLY this child; the parent supervises and
respawns it. Streams gaze + homography + pose to the parent over IPC queues.

Top level is stdlib-only ON PURPOSE: the parent imports run_child to use it as the Process
target and must NOT load the Sol SDK / native av into its own address space. The heavy imports
(ntuh.sol.connector -> ganzin SDK + the [SolPatch] decode monkeypatch; ntuh.sol.projector -> cv2)
happen INSIDE run_child, i.e. only in the child.
"""
import asyncio
import queue
import sys
import threading
import time

from ntuh.sol import sol_ipc


def _install_child_excepthook():
    """Minimal stderr-only excepthook - the child must NOT touch tk/messagebox (headless)."""
    def _hook(exctype, value, tb):
        import traceback
        traceback.print_exception(exctype, value, tb, file=sys.stderr)
    sys.excepthook = _hook


def _extract_img(frame, np):
    """Scene FrameData -> (1200,1328,3) uint8 numpy, mirroring preview_sol_gaze extraction."""
    try:
        if hasattr(frame, "img") and frame.img is not None:
            return frame.img
        if hasattr(frame, "get_buffer"):
            buf = frame.get_buffer()
            arr = np.frombuffer(buf, dtype=np.uint8)
            return arr.reshape((1200, 1328, 3))
        if isinstance(frame, np.ndarray):
            return frame
    except Exception:
        return None
    return None


def run_child(params, gaze_q, msg_q, cmd_q, frame_q=None):
    """Child process entry (multiprocessing target). `params` is a plain picklable dict:
    ip, port, aruco_dict_id, screen_w/h/x/y, marker_k/marker_n/marker_pattern_size,
    marker_container_size, screen_width_m, pose_smooth, seed_homography(list|None),
    stream_frames(bool). When params['stream_frames'] is set and frame_q is provided, the
    already-decoded scene frames are JPEG-encoded and streamed to the parent's tester view
    (newest-only). The native H.264 decode stays here in the child; only compressed bytes cross."""
    _install_child_excepthook()

    # --- Deferred heavy imports: CHILD ONLY ---
    # Importing ntuh.sol.connector applies the [SolPatch] single-threaded-decode monkeypatch to
    # THIS child's freshly-imported SDK objects.
    import numpy as np
    import cv2
    from ntuh.sol.connector import SolConnector, SDK_AVAILABLE
    from ntuh.sol.projector import ScreenProjector3D, create_calibration_assets

    # Startup diagnostic: confirm the child did NOT inherit the GUI stack + the patch is active.
    leaked = [m for m in ("pygame", "tkinter", "gazefollower", "mediapipe") if m in sys.modules]
    try:
        from ganzin.sol_sdk.streaming.video_mixin import VideoMixin
        patched = VideoMixin._create_video_codec.__name__
    except Exception:
        patched = "?"
    try:
        msg_q.put({"type": sol_ipc.ERROR, "where": "startup",
                   "error": f"sys.modules leak={leaked or 'clean'}; decode_patch={patched}"})
    except Exception:
        pass

    # TEST HOOK: set SOL_CHILD_CRASH_AFTER=<seconds> to simulate the native decode crash
    # (hard process exit) so the parent's crash/respawn path can be exercised without the SDK.
    import os
    _crash_after = os.environ.get("SOL_CHILD_CRASH_AFTER")
    if _crash_after:
        def _suicide():
            try:
                delay = float(_crash_after)
            except ValueError:
                return
            time.sleep(delay)
            print("[sol_child] TEST: simulating native crash via os._exit(1)", file=sys.stderr)
            os._exit(1)
        threading.Thread(target=_suicide, daemon=True).start()

    if not SDK_AVAILABLE:
        try:
            msg_q.put({"type": sol_ipc.CONNECT_FAILED, "error": "Sol SDK not available in child"})
        except Exception:
            pass
        return

    adict = cv2.aruco.getPredefinedDictionary(int(params["aruco_dict_id"]))
    internal_gaze_q = queue.Queue(maxsize=120)
    internal_scene_q = queue.Queue(maxsize=2)
    connector = SolConnector(params["ip"], params["port"], internal_gaze_q, internal_scene_q)

    state = {"proj": None, "running": True, "connect_failed": False}

    def on_connect(message, cam_params, time_offset_ms):
        try:
            cam_matrix = cam_params["cam_matrix"]
            dist = cam_params["dist_coeffs"]
            proj = ScreenProjector3D(cam_matrix, dist, adict, smoothing_factor=float(params["pose_smooth"]))
            seed = params.get("seed_homography")
            if seed is not None:
                proj.set_homography(np.array(seed, dtype=float), valid=False)
            cfg_assets = {"marker_k": params["marker_k"], "marker_n": params["marker_n"],
                          "marker_pattern_size": params["marker_pattern_size"]}
            markers_px, _imgs = create_calibration_assets(params["screen_w"], params["screen_h"], adict, cfg_assets)
            marker_phys_m = params["marker_pattern_size"] / params["screen_w"] * params["screen_width_m"]
            proj.start_background_detection(marker_phys_m, markers_px, params["marker_container_size"],
                                            params["screen_w"], params["screen_h"], params["screen_width_m"])
            state["proj"] = proj
            msg_q.put({"type": sol_ipc.CONNECTED,
                       "cam_matrix": np.asarray(cam_matrix, dtype=float).tolist(),
                       "dist_coeffs": np.asarray(dist, dtype=float).reshape(-1).tolist(),
                       "time_offset_ms": int(time_offset_ms), "message": str(message)})
        except Exception as e:
            try:
                msg_q.put({"type": sol_ipc.ERROR, "where": "on_connect", "error": str(e)})
            except Exception:
                pass

    def on_fail(err):
        state["connect_failed"] = True
        try:
            msg_q.put({"type": sol_ipc.CONNECT_FAILED, "error": str(err)})
        except Exception:
            pass

    def gaze_forward():
        while state["running"]:
            try:
                g = internal_gaze_q.get(timeout=0.2)
            except queue.Empty:
                continue
            d = sol_ipc.gaze_to_dict(g)
            if d is None:
                continue
            try:
                gaze_q.put_nowait(d)
            except queue.Full:  # drop-oldest, mirroring connector gaze backpressure
                try:
                    gaze_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    gaze_q.put_nowait(d)
                except queue.Full:
                    pass
            except Exception:
                pass

    def scene_publish():
        EMIT = 1.0 / 15.0
        FRAME_EMIT = 1.0 / 15.0  # stream the tester-view frame at the same cadence as the homography
        stream_frames = bool(params.get("stream_frames")) and frame_q is not None
        last_emit = 0.0
        last_frame_emit = 0.0
        while state["running"]:
            proj = state["proj"]
            if proj is None:
                time.sleep(0.05)
                continue
            img = None
            for _ in range(5):
                try:
                    frame = internal_scene_q.get(timeout=0.1)
                except queue.Empty:
                    break
                ex = _extract_img(frame, np)
                if ex is not None:
                    img = ex
            if img is not None:
                proj.submit_frame_for_pose(img)
                # Stream the (already-decoded) scene frame to the parent's tester view, opt-in and
                # throttled. Only compressed JPEG bytes cross the queue - the native H.264 decode
                # that the isolation contains has already happened, so this is crash-safe. Newest
                # frame wins (drop-oldest) so display latency stays bounded to ~1 frame.
                if stream_frames:
                    now_f = time.time()
                    if now_f - last_frame_emit >= FRAME_EMIT:
                        last_frame_emit = now_f
                        try:
                            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                            if ok:
                                data = enc.tobytes()
                                try:
                                    frame_q.put_nowait(data)
                                except queue.Full:
                                    try:
                                        frame_q.get_nowait()
                                    except queue.Empty:
                                        pass
                                    try:
                                        frame_q.put_nowait(data)
                                    except queue.Full:
                                        pass
                        except Exception:
                            pass
            now = time.time()
            if now - last_emit >= EMIT:
                last_emit = now
                try:
                    H = proj.get_homography()
                    msg_q.put({"type": sol_ipc.HOMOGRAPHY,
                               "H": (H.tolist() if H is not None else None),
                               "valid": bool(proj.is_homography_valid(strict=True)),
                               "ids": list(proj.get_detected_marker_ids())})
                    r, t = proj.get_current_pose()
                    msg_q.put({"type": sol_ipc.POSE,
                               "rvec": (r.tolist() if r is not None else None),
                               "tvec": (t.tolist() if t is not None else None),
                               "valid": bool(proj.is_pose_valid)})
                except Exception as e:
                    try:
                        msg_q.put({"type": sol_ipc.ERROR, "where": "scene_publish", "error": str(e)})
                    except Exception:
                        pass

    def cmd_poller():
        while state["running"]:
            try:
                c = cmd_q.get(timeout=0.2)
            except (queue.Empty, EOFError, OSError):
                continue
            cmd = c.get("cmd") if isinstance(c, dict) else None
            if cmd == sol_ipc.RESUME_SCENE:
                connector.resume_scene_stream()
            elif cmd == sol_ipc.PAUSE_SCENE:
                connector.pause_scene_stream()
            elif cmd == sol_ipc.STOP:
                state["running"] = False
                connector.stop_event.set()
                break

    threads = [threading.Thread(target=f, daemon=True, name=n)
               for f, n in ((gaze_forward, "sol_gaze_fwd"),
                            (scene_publish, "sol_scene_pub"),
                            (cmd_poller, "sol_cmd_poll"))]
    for th in threads:
        th.start()

    # Run the Sol session on the child's main thread, with connect-retry for device-busy on respawn.
    attempts = 3
    try:
        for i in range(attempts):
            if connector.stop_event.is_set():
                break
            state["connect_failed"] = False
            try:
                asyncio.run(connector.run_session(on_connect, on_fail))
            except Exception as e:
                print(f"[sol_child] run_session error: {e}", file=sys.stderr)
            if connector.stop_event.is_set() or not state["connect_failed"]:
                break
            time.sleep(0.8)  # connect failed (device busy after a crash) -> retry
    finally:
        state["running"] = False
        proj = state["proj"]
        if proj is not None:
            try:
                proj.stop_background_detection()
            except Exception:
                pass
        try:
            msg_q.put({"type": sol_ipc.STOPPING})
        except Exception:
            pass
