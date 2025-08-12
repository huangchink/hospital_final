# _*_ coding: utf-8 _*_
# Author: GC Zhu (patched for robustness)

import pathlib
from typing import Tuple, Any
import time
import numpy as np
from numpy import ndarray
import cv2 as cv

from .Calibration import Calibration
from ..logger import Log


class SVRCalibration(Calibration):
    def __init__(self, model_save_path: str = ""):
        """
        Two SVMs (x/y). Also persist valid feature indices so prediction matches training cols.
        """
        super().__init__()

        try:
            cv.setNumThreads(1)  # reduce deadlock risk on some machines
        except Exception:
            pass

        if model_save_path == "":
            self.workplace_calibration_dir = pathlib.Path.home().joinpath("GazeFollower", "calibration")
            self.workplace_calibration_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.workplace_calibration_dir = pathlib.Path(model_save_path)

        self.svr_x_path = self.workplace_calibration_dir.joinpath("svr_x.xml")
        self.svr_y_path = self.workplace_calibration_dir.joinpath("svr_y.xml")
        self.valid_idx_path = self.workplace_calibration_dir.joinpath("valid_feat_idx.npy")

        self.svr_x = cv.ml.SVM.create()
        self.svr_y = cv.ml.SVM.create()

        # default
        self._valid_feat_idx: np.ndarray | None = None

        if self.svr_y_path.exists() and self.svr_x_path.exists():
            self.svr_x = cv.ml.SVM_load(str(self.svr_x_path))
            self.svr_y = cv.ml.SVM_load(str(self.svr_y_path))
            if self.valid_idx_path.exists():
                try:
                    self._valid_feat_idx = np.load(self.valid_idx_path).astype(int)
                    self.has_calibrated = True
                    Log.d(f"Loaded calibration from: {self.workplace_calibration_dir}")
                except Exception as e:
                    self.has_calibrated = False
                    Log.e(f"Failed to load valid_feat_idx.npy: {e}. Recalibration required.")
            else:
                # 沒有特徵索引就當作未完成校正（否則 predict 會用錯維度）
                self.has_calibrated = False
                Log.w("valid_feat_idx.npy missing. Will require one-time recalibration to regenerate.")
        else:
            self._set_svm_params(self.svr_x)
            self._set_svm_params(self.svr_y)
            self.has_calibrated = False

    @staticmethod
    def _set_svm_params(svr):
        svr.setType(cv.ml.SVM_EPS_SVR)
        svr.setKernel(cv.ml.SVM_RBF)
        svr.setC(1.0)
        svr.setGamma(0.01)
        svr.setP(0.001)
        svr.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 3000, 1e-3))
        try:
            svr.setShrinking(True)
        except Exception:
            pass

    @staticmethod
    def _sanitize(features: np.ndarray, labels: np.ndarray):
        """Remove non-finite samples; drop zero-variance columns."""
        mask_samples = np.isfinite(features).all(axis=1) & np.isfinite(labels).all(axis=1)
        features = features[mask_samples]
        labels = labels[mask_samples]

        if features.size > 0:
            std = features.std(axis=0)
            valid_idx = np.where(std > 1e-8)[0]
            features = features[:, valid_idx]
        else:
            valid_idx = np.array([], dtype=int)

        return features.astype(np.float32), labels.astype(np.float32), valid_idx

    def _predict_with_models(self, features: np.ndarray) -> np.ndarray:
        px = self.svr_x.predict(features)[1]
        py = self.svr_y.predict(features)[1]
        return np.concatenate((px, py), axis=1)

    def predict(self, features, estimated_coordinate) -> Tuple:
        """Return calibrated (x,y); fall back to estimated when unsafe."""
        feats = np.array(features, dtype=np.float32).reshape(1, -1)

        if not self.has_calibrated or self._valid_feat_idx is None or len(self._valid_feat_idx) == 0:
            # 沒有有效的特徵索引 → 視為未校正，直接回傳 estimated
            Log.d("Calibration incomplete (no valid feature idx). Using estimated coordinate.")
            return False, estimated_coordinate

        try:
            feats = feats[:, self._valid_feat_idx]
        except Exception as e:
            Log.e(f"Feature alignment failed: {e}. Using estimated coordinate.")
            return False, estimated_coordinate

        try:
            pred = self._predict_with_models(feats)
            x, y = float(pred[0, 0]), float(pred[0, 1])
            # 基本健檢：非有限值直接回退
            if not (np.isfinite(x) and np.isfinite(y)):
                raise ValueError(f"Non-finite pred: {(x, y)}")
            return True, (x, y)
        except Exception as e:
            Log.e(f"Predict failed: {e}. Using estimated coordinate.")
            return False, estimated_coordinate

    def calibrate(self, features, labels, ids=None) \
            -> Tuple[bool, float, ndarray] | Tuple[bool, float, Any]:

        features, labels, valid_idx = self._sanitize(features, labels)
        self._valid_feat_idx = valid_idx

        n = features.shape[0]
        if n < 4:
            self.has_calibrated = False
            Log.e(f"Too few valid samples after sanitizing: {n}")
            return self.has_calibrated, float('inf'), None

        labels_x = labels[:, 0].reshape(-1, 1)
        labels_y = labels[:, 1].reshape(-1, 1)

        # Train RBF → fallback LINEAR
        try:
            t0 = time.time()
            self._set_svm_params(self.svr_x)
            self._set_svm_params(self.svr_y)
            self.svr_x.train(features, cv.ml.ROW_SAMPLE, labels_x)
            self.svr_y.train(features, cv.ml.ROW_SAMPLE, labels_y)
            dt = time.time() - t0
            Log.d(f"SVM(RBF) training finished in {dt:.3f}s with {n} samples, {features.shape[1]} feats.")
            self.has_calibrated = True
            if dt > 5.0:
                raise RuntimeError(f"SVM RBF training took too long: {dt:.2f}s")
        except Exception as e:
            Log.e(f"SVM RBF training failed or too slow: {e}. Fallback to LINEAR.")
            try:
                self.svr_x = cv.ml.SVM.create(); self.svr_y = cv.ml.SVM.create()
                self.svr_x.setType(cv.ml.SVM_EPS_SVR); self.svr_y.setType(cv.ml.SVM_EPS_SVR)
                self.svr_x.setKernel(cv.ml.SVM_LINEAR); self.svr_y.setKernel(cv.ml.SVM_LINEAR)
                self.svr_x.setC(1.0); self.svr_y.setC(1.0)
                self.svr_x.setP(0.001); self.svr_y.setP(0.001)
                self.svr_x.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 2000, 1e-3))
                self.svr_y.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 2000, 1e-3))
                t1 = time.time()
                self.svr_x.train(features, cv.ml.ROW_SAMPLE, labels_x)
                self.svr_y.train(features, cv.ml.ROW_SAMPLE, labels_y)
                dt2 = time.time() - t1
                Log.d(f"SVM(LINEAR) training finished in {dt2:.3f}s.")
                self.has_calibrated = True
            except Exception as e2:
                self.has_calibrated = False
                Log.e(f"Fallback LINEAR SVM failed: {e2}")
                # Clean up wrong files
                for p in (self.svr_x_path, self.svr_y_path, self.valid_idx_path):
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
                return self.has_calibrated, float('inf'), None

        # Evaluate
        try:
            preds = self._predict_with_models(features)
            euclidean_distances = np.sqrt((labels_x - preds[:, [0]]) ** 2 + (labels_y - preds[:, [1]]) ** 2)
            mean_euclidean_error = float(np.mean(euclidean_distances))
            Log.d(f"Calibration completed. Mean Euclidean error: {mean_euclidean_error:.4f}")
            return self.has_calibrated, mean_euclidean_error, preds
        except Exception as e:
            self.has_calibrated = False
            Log.e(f"Post-training prediction failed: {e}")
            return self.has_calibrated, float('inf'), None

    def save_model(self) -> bool:
        """Save SVMs + valid feature idx."""
        if self.svr_x.isTrained() and self.svr_y.isTrained() and \
           (self._valid_feat_idx is not None) and (len(self._valid_feat_idx) > 0):
            self.svr_x.save(str(self.svr_x_path))
            self.svr_y.save(str(self.svr_y_path))
            try:
                np.save(self.valid_idx_path, self._valid_feat_idx)
            except Exception as e:
                Log.e(f"Save valid_feat_idx failed: {e}")
                return False
            Log.d(f"SVR model for x saved at: {self.svr_x_path}")
            Log.d(f"SVR model for y saved at: {self.svr_y_path}")
            Log.d(f"Valid feature idx saved at: {self.valid_idx_path}")
            return True
        else:
            Log.d("SVR model not trained or no valid features; skip saving.")
            return False

    def release(self):
        pass
