#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : gui.py
# Time       : 2025/11/10 16:05
# Author     : Jago
# Email      : huwl@hku.hk
# Description：
GUI tool to monitor SerialEM nav/montage outputs, split montages into tiles,
run YOLO inference on tiles, allow interactive label inspection/editing, and
export collected coordinates back to a nav-style file.

Main components:
 - SettingsPanel: set project paths and parameters; export predictions to existing nav file
 - StatusPanel: lists montages and per-montage statuses, starts processing
 - ViewerPanel: display selected montage/tile with predicted boxes; edit boxes
 - Background workers: splitting + prediction + finding-center + deduplication pipeline; filesystem watcher
"""
from __future__ import annotations
import sys
import threading
import time
import warnings
from queue import Queue, Empty
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

import cv2
import mrcfile
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ultralytics import YOLO

from src.utils.data_models import Montage, Tile, Detection, STATUS_COLORS
from src.utils.pp_io import load_nav_and_montages, ensure_project_dirs, write_detections, read_detections, \
    add_predictions_to_nav, TILE_NAME_TEMPLATE, PRED_NAME_TEMPLATE, append_to_manual_list, read_manual_list, match_name, \
    _TILE_RE, remove_from_manual_list, _PRED_RE, update_montage_if_map_generated, check_overlap, \
    collect_and_map_points_for_montage, deduplicate_global_points, preview_nav_montages
from src.utils.row_reorder_table import RowReorderTable

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)
default_brightness = 100


def get_resource_path(relative_path: str) -> Path:
    """兼容 PyInstaller 打包与源码运行"""
    if hasattr(sys, "_MEIPASS"):  # 运行在 PyInstaller 打包环境
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent.parent  # src 的上级目录

    return (base_path / relative_path).resolve()


# -----------------------------
# Filesystem Watcher
# -----------------------------
class MontageWatcher(FileSystemEventHandler):
    """Monitors directory for new stable MRC files and triggers processing."""

    def __init__(self, nav_path: Path, ui_queue: Queue, job_queue: Queue, montages: Dict[str, Montage]):
        super().__init__()
        self.nav_path = nav_path
        self.ui_queue = ui_queue
        self.job_queue = job_queue
        self.montages = montages
        self.processed_files = set()                        # 记录已处理的文件，避免重复处理
        # self._scan_existing_files()  直接使用会导致在主线程运行，使得UI卡死，job_queue后台疯狂运行

    def scan_existing_files_async(self):
        """专门提供一个方法，供外部在独立线程中调用"""
        logger.info(f"Starting initial scan for existing files in {self.nav_path.parent}...")
        candidate_files = sorted(p.resolve() for p in self.nav_path.parent.glob("*") if p.is_file())
        for p in candidate_files:
            if p not in self.processed_files and self._is_file_stable(p, check_interval=0.5):
                # 如果文件稳定，则提交到处理队列
                self._process_mrc_file(p)
        logger.info("Initial scan finished.")

    def _is_file_stable(self, file_path: Path, check_interval: float = 5.0) -> bool:
        """检查文件是否稳定（不再被修改）"""
        try:
            if not file_path.exists():
                return False
            initial_size = file_path.stat().st_size             # 获取初始文件大小
            time.sleep(check_interval)
            current_size = file_path.stat().st_size
            return initial_size == current_size and initial_size > 0
        except (OSError, IOError):
            logger.error(f"Montage monitoring failed: {file_path}")
            return False

    def _process_mrc_file(self, file_path: Path):
        """处理稳定的montage文件"""
        mont = self.montages.get(file_path.name)
        if not mont:
            return

        if mont.status == "to be validated":
            mont.status = "queuing"
            mont.map_file = file_path
            self.ui_queue.put(("update_montage_status", (mont, None)))
            self.job_queue.put(mont)
        elif mont.status == "to be shot":
            info = update_montage_if_map_generated(self.nav_path, mont)
            if info == "Updated":
                mont.status = "queuing"
                self.ui_queue.put(("update_montage_status", (mont, None)))
                self.job_queue.put(mont)
            else:
                mont.status = "error"
                self.ui_queue.put(("update_montage_status", (mont, None)))
                logger.error(f"Updated {mont.name} failed: {info}")
        # elif "processed" or "excluded" or "error":

        self.processed_files.add(file_path.resolve())       # 标记为已处理

    def on_created(self, event):
        p = Path(event.src_path).resolve()
        if self._is_file_stable(p):
            self._process_mrc_file(p)

    def on_moved(self, event):
        p = Path(event.dest_path).resolve()                     # 移动事件的目标路径在dest_path
        if self._is_file_stable(p):
            self._process_mrc_file(p)

    def on_modified(self, event):
        # 对于大文件，修改事件可能在写入过程中多次触发
        p = Path(event.src_path).resolve()
        if p not in self.processed_files and self._is_file_stable(p, check_interval=10.0):    # 对于修改事件，使用更长的检查间隔
            self._process_mrc_file(p)


# -----------------------------
# YOLO Wrapper
# -----------------------------
class YoLoWrapper:
    """Wraps YOLO model interactions."""

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict_image(self, img_path: str, cfg: dict) -> Tuple[Optional[List[Detection]], float, float, float]:
        # cfg keys expected: model_path, img_size, box_size, max_detection, conf, iou, device (either 'cpu' or [GPU index list])
        dets: List[Detection] = []
        s_pre, s_i, s_post = 0.0, 0.0, 0.0
        try:
            results = self.model.predict(source=img_path, conf=cfg["conf"], iou=cfg["iou"], imgsz=cfg["img_size"],
                                         device=cfg["device"], max_det=cfg["max_detection"], save_conf=True, verbose=False)
            for r in results:
                for b in r.boxes:
                    xywh = b.xywh[0].cpu().numpy()  # xywh: tensor([[3094.2964, 3053.0522,  263.7949,  265.7412]])
                    conf = float(b.conf[0].cpu().numpy())
                    cls = int(b.cls[0].cpu().numpy())
                    x, y, w, h = xywh
                    w, h = float(cfg["box_size"]), float(cfg["box_size"])
                    dets.append(Detection(cls, x, y, w, h, conf, "active"))
                s_pre = round(r.speed['preprocess'] / 1000, 2)
                s_i = round(r.speed['inference'] / 1000, 2)
                s_post= round(r.speed['postprocess'] / 1000, 2)
        except Exception as e:
            logger.error(f"Inference error on {img_path}: {e}")

        return dets, s_pre, s_i, s_post


# -----------------------------
# Pipeline worker classes
# -----------------------------
class MontageProcessor(QtCore.QThread):
    """Worker: Splits MRC -> Tiles, Runs YOLO, Finds Center Point, Saves Results."""

    def __init__(self, ui_queue: Queue, job_queue: Queue, predictor: YoLoWrapper, cfg: dict):
        super().__init__()
        self.ui_queue = ui_queue
        self.job_queue = job_queue
        self.predictor = predictor
        self.cfg = cfg
        self._stopping = False
        self.project_root = self.cfg["nav_path"].parent / self.cfg["project_name"]

    def stop(self):
        self._stopping = True

    def run(self):
        images_dir, preds_dir = ensure_project_dirs(self.cfg["nav_path"].parent, self.cfg["project_name"])

        while not self._stopping:
            try:
                # Use a timeout to allow checking self._stopping flag periodically
                montage = self.job_queue.get(timeout=0.5)
            except Empty:
                continue

            # If we got a job, process it fully even if stop was requested during get()
            try:
                self._process_single_montage(montage, images_dir, preds_dir)
            except Exception as e:
                montage.status = "error"
                msg = f"Processing failed for {montage.name}: {e}"
                logger.error(msg)
                self.ui_queue.put(("update_montage_status", (montage, msg)))
            finally:
                self.job_queue.task_done()

    def _process_single_montage(self, montage: Montage, images_dir: Path, preds_dir: Path):
        """Handle splitting, prediction, finding center, deduplication for one montage."""
        montage.status = "processing"
        self.ui_queue.put(("update_montage_status", (montage, None)))

        nx, ny = montage.map_frames
        try:
            with mrcfile.mmap(montage.map_file, mode='r+') as mrc:  # Open the file in memory-mapped mode
                Z = mrc.data.shape[0]  # z,y,x
                if Z != nx * ny:
                    raise ValueError(f"Z-dim {Z} mismatch with map_frames {nx}*{ny}")

                for z in range(Z):
                    # 1. Prepare Paths
                    tile_name = TILE_NAME_TEMPLATE.format(montage=montage.map_file.stem, z=z)
                    pred_name = PRED_NAME_TEMPLATE.format(montage=montage.map_file.stem, z=z)
                    tile_path = images_dir / tile_name
                    pred_path = preds_dir / pred_name

                    # 2. Save Image (Normalized)
                    splitting_time = time.time()
                    img_data = mrc.data[z].astype(np.uint16).astype(np.float16)  # To avoid transforming to float64 to compute img_norm
                    img_norm = ((img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255).astype(np.uint8)  # Ultralytics only accept int8 images to be processed
                    if not tile_path.exists() or self.cfg.get('overwrite', False):  # write png if not existing or overwrite
                        cv2.imwrite(str(tile_path), img_norm)
                    tile = Tile(name=tile_name, tile_sec=z, tile_file=tile_path)
                    splitting_time = time.time() - splitting_time

                    # 3. Prediction & Find center
                    if not pred_path.exists() or self.cfg.get('overwrite', False):
                        # A. Run YOLO
                        dets, s_pre, s_infer, s_post = self.predictor.predict_image(str(tile_path), self.cfg)
                        centering_time = time.time()
                        if len(dets) == 0:
                            append_to_manual_list(self.project_root, tile_name)
                            msg = f"For {tile_name}: cannot find any points, added to manual confirmation list."
                            logger.info(msg)
                            self.ui_queue.put(("add_manual_item", (tile_name, msg)))
                        else:
                            # B. Find Center Point
                            # center_det = self._find_center_point(img_norm, dets)  # TOO BAD
                            # center_det = self._find_center_point_kmeans(img_norm, dets)  # 15/31 -> 12/32
                            # center_det = self._find_center_point_dbscan(img_norm, dets)  # TOO SLOW
                            center_det = self._find_center_point_fft(img_norm, dets)  # FASTEST 6/32 -> 7/32 -> 9/32 -> 10/32 -> 8/32 -> 8/32 -> 6/32 -> 8/32
                            if center_det:  # Add to front of list
                                dets.insert(0, center_det)
                            else:  # Not found: Signal UI to add to manual list
                                append_to_manual_list(self.project_root, tile_name)
                                msg = f"For {tile_name}: cannot find the tracking point, added to manual confirmation list."
                                logger.info(msg)
                                self.ui_queue.put(("add_manual_item", (tile_name, msg)))

                        centering_time = time.time() - centering_time
                        write_detections(pred_path, dets)
                    montage.tiles[tile_name] = tile
                    msg = (f"For {tile_name}: picked {len(dets)} points, {round(s_pre + splitting_time, 2)}s preprocess, "
                           f"{s_infer}s inference, {round(s_post + centering_time, 2)}s postprocess.")
                    logger.info(msg)
                    self.ui_queue.put(("add_tile_item", (montage, msg)))

            # 4. Deduplication Logic
            kept_num, removed_num = self._deduplicate_montage(montage, preds_dir)
            montage.status = "processed"
            msg = f"For {montage.name}: removed {removed_num} points for collision, kept {kept_num} points."
            logger.info(msg)
            self.ui_queue.put(("update_montage_status", (montage, msg)))
        except Exception as e:
            logger.error(f"Error reading MRC {montage.map_file}: {e}")
            raise e

    def _find_center_point_fft(self, img: np.ndarray, existing_dets: List[Detection]) -> Optional[Detection]:
        """
        Identify Carbon areas using FFT High-Pass Filtering.
        Carbon has texture (medium high-freq energy), Holes are smooth (low energy), Edges are sharp (very high energy).
        """
        h, w = img.shape
        box_size = self.cfg["box_size"]

        # 1. Downsample for speed (Processing at 512px is sufficient for texture analysis)
        scale_factor = 512.0 / max(h, w)
        if scale_factor < 1.0:
            small_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        else:
            small_img = img.copy()
            scale_factor = 1.0

        sh, sw = small_img.shape
        small_img = cv2.normalize(small_img, None, 0, 255, cv2.NORM_MINMAX)
        # 2. FFT High Pass Filter to extract "Structure/Texture" map
        f = np.fft.fft2(small_img.astype(np.float32))
        fshift = np.fft.fftshift(f)
        # Mask out center (Low Frequencies) - removing illumination/gradients
        crow, ccol = sh // 2, sw // 2
        # Mask radius: ~2% of image size is usually enough to remove "flat" components
        mask_rad = int(min(sh, sw) * 0.02)
        fshift[crow - mask_rad:crow + mask_rad, ccol - mask_rad:ccol + mask_rad] = 0
        # Inverse FFT to get structure image
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # 3. Analyze Texture Energy
        # Smooth the structure map to get regional texture estimates
        texture_map = cv2.GaussianBlur(img_back, (9, 9), 0)
        # Normalize to 0-255 for thresholding
        texture_map = cv2.normalize(texture_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 4. Thresholding to find "Carbon"
        # - Low Texture (< T1) -> Holes / Flat Ice (Too smooth)
        # - High Texture (> T2) -> Edges / Grid Bars / Contaminants (Too sharp)
        # - Medium Texture -> Carbon
        # Dynamic Thresholding based on image statistics is more robust than fixed values
        mean_val = np.mean(texture_map)
        std_val = np.std(texture_map)
        # Carbon typically lies around the mean noise level
        # Holes are significantly below mean; Edges are significantly above.
        lower_thresh = mean_val - 0.5 * std_val  # Cut off very smooth areas
        upper_thresh = mean_val + 2.0 * std_val  # Cut off strong edges
        # Create Binary Mask
        # lower < texture < upper)
        texture_mask = cv2.inRange(texture_map, lower_thresh, upper_thresh)

        # 5. Clean up Mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_OPEN, kernel)  # Remove noise specks
        texture_mask = cv2.erode(texture_mask, kernel, iterations=2)  # Shrink away from edges
        intensity_mask = cv2.inRange(small_img, 25, 200)
        masked_by_intensity = cv2.bitwise_and(texture_mask, intensity_mask)

        # 6. Apply Central 1/4 ROI Constraint
        roi_mask = np.zeros_like(masked_by_intensity)
        roi_x1, roi_x2 = int(sw * 0.25), int(sw * 0.75)
        roi_y1, roi_y2 = int(sh * 0.25), int(sh * 0.75)
        roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255
        # Combine Carbon Mask and ROI
        final_mask = cv2.bitwise_and(masked_by_intensity, roi_mask)
        if cv2.countNonZero(final_mask) == 0:
            logger.warning(f"FFT: No valid carbon area found in center.")
            return None

        # 7. Distance Transform to find the "deepest" point inside the valid carbon area
        dist_transform = cv2.distanceTransform(final_mask, cv2.DIST_L2, 5)
        # Add a bias towards image center
        center_x_small, center_y_small = sw // 2, sh // 2
        y_grid, x_grid = np.indices((sh, sw))
        # Distance from image center (normalized)
        dist_from_center = np.sqrt((x_grid - center_x_small) ** 2 + (y_grid - center_y_small) ** 2)
        max_dist_center = np.sqrt(center_x_small ** 2 + center_y_small ** 2)
        # Centrality Score: 1.0 at center, 0.0 at corners
        centrality_map = 1.0 - (dist_from_center / max_dist_center)
        # Final Score = Safety (Distance from edges) * Weight + Centrality * Weight
        # We value Safety more than Centrality to avoid edges
        score_map = dist_transform + (centrality_map * (box_size * scale_factor * 0.2))

        # Get candidates
        # Only look at points with valid distance > buffer (2 pixels in small_img)
        # Must be within final_mask to ensure texture and intensity constraints are met
        ys, xs = np.nonzero(cv2.bitwise_and(final_mask, (dist_transform > 2.0).astype(np.uint8)))
        if len(xs) == 0:
            return None

        candidates = []
        for x, y in zip(xs, ys):
            score = score_map[y, x]
            candidates.append((score, x, y))

        candidates.sort(key=lambda item: item[0], reverse=True)

        # 8. Check Overlap
        final_cx, final_cy = None, None
        for _, cx_small, cy_small in candidates[:50]:
            cx_orig = cx_small / scale_factor
            cy_orig = cy_small / scale_factor
            if not check_overlap(cx_orig, cy_orig, box_size * 4, existing_dets):
                half_box = int(box_size * scale_factor // 2)
                x1 = max(0, int(cx_small) - half_box)
                y1 = max(0, int(cy_small) - half_box)
                x2 = min(w, int(cx_small) + half_box)
                y2 = min(h, int(cy_small) + half_box)
                # Extract the candidate box from original image
                candidate_box = small_img[y1:y2, x1:x2]
                # Check if box contains very bright pixels (holes)
                # Using a threshold close to 255 to detect holes
                bright_pixels = np.sum(candidate_box > 200)
                # If too many bright pixels are found, skip this candidate (it has holes)
                max_allowed_bright_pixels = candidate_box.size * 0.01  # Allow up to 1% bright pixels
                if bright_pixels > max_allowed_bright_pixels:
                    continue  # Skip this candidate, it has holes

                final_cx, final_cy = cx_orig, cy_orig
                break

        if final_cx is not None:
            return Detection(cls=0, x=final_cx, y=final_cy, w=float(box_size), h=float(box_size), conf=2.0, status="active")

        return None

    def _deduplicate_montage(self, montage: Montage, preds_dir: Path) -> Tuple[int, int]:
        """Detects and removes duplicate detections in overlapping regions between tiles. Returns the number of removed detections.
        """
        # 1. Get PieceSpacing
        nav_folder = self.cfg["nav_path"].parent
        box_size = self.cfg["box_size"]

        all_points, tile_dets_map = collect_and_map_points_for_montage(montage.map_file.name, montage.map_frames, preds_dir, nav_folder)
        if not all_points:
            logger.debug(f"Skipped deduplication for {montage.map_file}.")
            return 0, 0

        # 2. 找到重复项
        final_points, removals = deduplicate_global_points(all_points, box_size)

        # 3. 应用移除结果到内存和磁盘 (此部分仍由 Worker 负责)
        if removals:
            # 标记 detections 在 in-memory map 中为 "deleted"
            for (z, idx) in removals:
                tile_dets_map[z][idx].status = "deleted"

            nx, ny = montage.map_frames
            for z in range(nx * ny):
                # 应用更改到磁盘 (写入过滤后的 valid_dets)
                pred_name = PRED_NAME_TEMPLATE.format(montage=montage.map_file.stem, z=z)
                p_path = preds_dir / pred_name
                # 筛选出未被删除的点
                valid_dets = [d for d in tile_dets_map[z] if d.status != "deleted"]
                if len(valid_dets) == 1 and valid_dets[0].conf > 1.0:  # this only one might be the tracking point
                    valid_dets = []
                write_detections(p_path, valid_dets)
                # 更新内存中的 montage 对象
                tile_name = TILE_NAME_TEMPLATE.format(montage=montage.map_file.stem, z=z)
                montage.tiles[tile_name].detections = valid_dets
        return len(final_points), len(removals)


# -----------------------------
# GUI Components
# -----------------------------
class SettingsPanel(QtWidgets.QWidget):
    """Configuration panel for paths and detection parameters."""

    # 定义自定义信号 - Qt框架中组件间通信的核心机制
    start_requested = QtCore.pyqtSignal(dict)           # 点击 RUN 时发射
    stop_requested = QtCore.pyqtSignal()                # 点击 STOP 时发射
    export_requested = QtCore.pyqtSignal(dict)          # 点击export时发射
    nav_changed = QtCore.pyqtSignal(Path)  # 当 nav 路径变为一个有效 .nav 文件时发射

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gpus = self._detect_gpu()                  # 仅需初始运行一次
        self._build_ui()

    @staticmethod
    def _detect_gpu() -> List[Dict]:
        """GPU检测"""
        gpus = []
        try:
            import torch
            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                for i in range(n):
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    mem_gb = round(props.total_memory / 1024 ** 3)
                    gpus.append({"id": i, "name": name, "mem_total": f"{mem_gb}GB"})
                return gpus
        except ImportError as e:
            logger.debug(f"GPU detection failed: {e}")
        return gpus

    def _build_ui(self):
        """UI构建: project_name, model_path, nav_path, box_size, max_detection, conf, iou, device_combo"""
        # 使用表单布局管理器 - 适合标签-字段对的排列
        layout = QtWidgets.QFormLayout(self)

        self.project_name = QtWidgets.QLineEdit()       # 项目名称文本输入框
        self.project_name.setText("Point-Picker")
        layout.addRow("Project name:", self.project_name)

        # Model line with Browse
        self.model_path = QtWidgets.QLineEdit()         # 模型文件路径文本输入框
        default_model = get_resource_path("data/md2_pm2_best.pt")
        self.model_path.setText(str(default_model))
        self.model_path_btn = QtWidgets.QPushButton("Browse")

        model_h = QtWidgets.QHBoxLayout()
        model_h.addWidget(self.model_path)
        model_h.addWidget(self.model_path_btn)
        layout.addRow("Model path:", model_h)

        # Nav line with Browse
        self.nav_path = QtWidgets.QLineEdit()            # 导航文件路径文本输入框
        default_nav = get_resource_path("test/1211_nav001.nav")         # for test
        self.nav_path.setText(str(default_nav))
        self.nav_path_btn = QtWidgets.QPushButton("Browse")

        nav_h = QtWidgets.QHBoxLayout()
        nav_h.addWidget(self.nav_path)
        nav_h.addWidget(self.nav_path_btn)
        layout.addRow("Nav file:", nav_h)

        self.img_size = QtWidgets.QSpinBox()
        self.img_size.setRange(256, 4096)
        self.img_size.setValue(2048)
        layout.addRow("Img size:", self.img_size)

        # micrograph在中倍地图上占据的box_size，如4800X pixel size为26.66 Å，数据收集pixel size为0.9557 Å，
        # 则有box_size = 4096 / (26.66 / 0.9557) = 147 pixels
        self.box_size = QtWidgets.QSpinBox()
        self.box_size.setRange(50, 500)
        self.box_size.setValue(150)
        layout.addRow("Box size:", self.box_size)

        # 最大检测数量选择器 - 限制每张图像的最大检测目标数
        self.max_detection = QtWidgets.QSpinBox()
        self.max_detection.setRange(1, 400)
        self.max_detection.setValue(50)
        layout.addRow("Max detections:", self.max_detection)

        # Confidence (0.0 - 1.0)
        self.conf = QtWidgets.QDoubleSpinBox()
        self.conf.setRange(0.0, 1.0)
        self.conf.setSingleStep(0.01)
        self.conf.setValue(0.25)
        layout.addRow("conf:", self.conf)

        # iou (0.0 - 1.0) 较低的数值可以消除重叠的方框，从而减少检测次数，这对减少重复检测非常有用。i.e.数值越大，挑的越多，但会重复。
        self.iou = QtWidgets.QDoubleSpinBox()
        self.iou.setRange(0.0, 1.0)
        self.iou.setSingleStep(0.01)
        self.iou.setValue(0.7)
        layout.addRow("iou:", self.iou)

        # Device row: label + checkbox（右侧）
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItem("CPU")
        if self.gpus:
            for g in self.gpus:
                self.device_combo.addItem(f"cuda:{g['id']} ({g['name']}, {g['mem_total']})")
        layout.addRow("Device:", self.device_combo)

        self.overwrite = QtWidgets.QCheckBox("Overwrite")
        self.run_btn = QtWidgets.QPushButton("Process Selected")
        self.stop_btn = QtWidgets.QPushButton("STOP")
        self.export_btn = QtWidgets.QPushButton("Export Selected")
        self.stop_btn.setEnabled(False)                     # Stop initially disabled; enabled once running 
        btn_h = QtWidgets.QHBoxLayout()
        btn_h.addWidget(self.overwrite)
        btn_h.addWidget(self.run_btn)
        btn_h.addWidget(self.stop_btn)
        btn_h.addWidget(self.export_btn)
        layout.addRow(btn_h)

        # 连接按钮事件
        self.model_path_btn.clicked.connect(self.on_browse_model)
        self.nav_path_btn.clicked.connect(self.on_browse_nav)
        # 当用户用 Browse 选择文件或手工输入并完成（editingFinished）时，触发读取 nav 的逻辑
        self.nav_path.editingFinished.connect(self.on_nav_path_set)
        self.run_btn.clicked.connect(self.on_run)
        self.stop_btn.clicked.connect(self.on_stop)
        self.export_btn.clicked.connect(self.on_export)

    def set_inputs_enabled(self, enabled: bool):
        """Enable or disable all configuration inputs."""
        self.project_name.setEnabled(enabled)
        self.model_path.setEnabled(enabled)
        self.model_path_btn.setEnabled(enabled)
        self.nav_path.setEnabled(enabled)
        self.nav_path_btn.setEnabled(enabled)
        self.img_size.setEnabled(enabled)
        self.box_size.setEnabled(enabled)
        self.max_detection.setEnabled(enabled)
        self.conf.setEnabled(enabled)
        self.iou.setEnabled(enabled)
        self.device_combo.setEnabled(enabled)
        self.overwrite.setEnabled(enabled)

    def on_browse_model(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select model file", filter="*.pt *.yaml *.pth")
        if f:
            self.model_path.setText(f)

    def on_browse_nav(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select nav file", filter="*.nav")
        if f:
            self.nav_path.setText(f)
            # 立即触发 nav 设置处理（等同于用户完成手工输入）
            self.on_nav_path_set()

    def on_nav_path_set(self):
        """当 nav_path 输入完成（editingFinished 或 Browse 后）调用。仅在文件名以 .nav 结尾且文件存在时发出 nav_changed(Path) 信号。"""
        txt = self.nav_path.text().strip()
        if not txt:
            return
        p = Path(txt)
        if p.suffix == ".nav" and p.exists():
            self.nav_changed.emit(p)
        # 非 .nav 文件或者不存在，不触发读取，保持静默。

    def _validate_cfg(self):
        cfg = {
            "project_name": self.project_name.text().strip(),
            "model_path": Path(self.model_path.text().strip()) if self.model_path.text().strip() else None,
            "nav_path": Path(self.nav_path.text().strip()) if self.nav_path.text().strip() else None,
            "img_size": self.img_size.value(),
            "box_size": self.box_size.value(),
            "max_detection": self.max_detection.value(),
            "conf": self.conf.value(),
            "iou": self.iou.value(),
            "overwrite": self.overwrite.isChecked(),
        }

        device_text = self.device_combo.currentText()
        if device_text == "CPU":
            device = "cpu"
        else:
            name = device_text.split()[0]  # 取"cuda:0"
            device = [int(name.split(":")[1])]  # 取[0]
        cfg["device"] = device

        if not (cfg["project_name"] and cfg["model_path"] and cfg["nav_path"]):
            QtWidgets.QMessageBox.warning(self, "Missing content", "Please specify project name / model path / nav path.")
            return None

        if not (cfg["model_path"].exists() and cfg["nav_path"].exists()):
            QtWidgets.QMessageBox.warning(self, "Illegal content", "Please specify right model path / nav path.")
            return None

        return cfg

    def on_run(self):
        cfg = self._validate_cfg()
        if cfg:
            # 所有验证通过，发射开始请求信号，携带配置字典
            self.start_requested.emit(cfg)
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.set_inputs_enabled(False)  # Disable inputs

    def on_stop(self):
        self.stop_requested.emit()
        self.stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        self.set_inputs_enabled(True)  # Re-enable inputs

    def on_export(self):
        """发射export请求"""
        cfg = {
            "project_name": self.project_name.text().strip(),
            "nav_path": Path(self.nav_path.text().strip()) if self.nav_path.text().strip() else None,
            "box_size": self.box_size.value()
        }

        if not (cfg["project_name"] and cfg["nav_path"]):
            QtWidgets.QMessageBox.warning(self, "Missing content","Please specify project name / nav path.")
        else:
            outp, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export points to nav", filter="*.nav")
            if outp:
                cfg["save_path"] = Path(outp)
                self.export_requested.emit(cfg)


class StatusPanel(QtWidgets.QWidget):
    """Displays list of montages and their processing status."""

    montage_selected = QtCore.pyqtSignal(str)       # 选择某行时发射montage name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.montages: Dict[str, Montage] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        # 创建表格部件，设置3列：名称、状态、复选（右侧）
        # self.table_widget = QtWidgets.QTableWidget(0, 3)  # 改为3列
        self.table_widget = RowReorderTable(0, 3)
        self.table_widget.setHorizontalHeaderLabels(["Montage Name", "Status", "Selected"])
        self.table_widget.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)  # 第一列拉伸
        self.table_widget.setColumnWidth(1, 160)
        self.table_widget.setColumnWidth(2, 80)
        self.table_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)  # 整行选择
        self.table_widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)  # 只能选一行
        self.table_widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)   # 不可编辑
        self.table_widget.itemSelectionChanged.connect(self.on_selection_changed)
        self.table_widget.rowsReordered.connect(self._on_rows_reordered)
        layout.addWidget(self.table_widget)

    def _on_rows_reordered(self, rows: list):
        """
        在用户通过拖动改变行顺序后调用：根据表格中第一列（name）的顺序，
        重新构建 self.montages（保持原有 Montage 对象、不复制）。
        """
        # 备份旧映射，按新顺序重建字典（保留 Montage 对象）
        old = self.montages
        new = {}
        for name in rows:
            if name in old:
                new[name] = old[name]
        # 如果有旧字典中存在但表格中未出现的项，也追加到末尾
        for k, v in old.items():
            if k not in new:
                new[k] = v
        self.refresh(new, True)

    def on_selection_changed(self):
        """处理表格选择变化"""
        selected_items = self.table_widget.selectedItems()
        if selected_items:
            name = selected_items[0].text()                                 # 第一列是名称
            self.montage_selected.emit(name)

    def refresh(self, montages: Dict[str, Montage], delete_previous: bool):
        """刷新表格显示"""
        self.montages = montages
        if delete_previous:
            self.table_widget.setRowCount(0)                                    # 清空所有行
        for m in self.montages.values():
            self.update_montage(m)

    def update_montage(self, m: Montage):
        """更新单个蒙太奇的状态"""
        for row in range(self.table_widget.rowCount()):
            if self.table_widget.item(row, 0).text() == m.name:
                self._set_row_data(row, m)
                return
        self.table_widget.insertRow(self.table_widget.rowCount())
        self._set_row_data(self.table_widget.rowCount() - 1, m)

    def _set_row_data(self, row: int, m: Montage):
        """将蒙太奇添加到表格中row行"""
        name_item = QtWidgets.QTableWidgetItem(m.name)
        stat_item = QtWidgets.QTableWidgetItem(m.status)
        color = QtGui.QColor(STATUS_COLORS.get(m.status))
        name_item.setBackground(color)
        stat_item.setBackground(color)
        self.table_widget.setItem(row, 0, name_item)
        self.table_widget.setItem(row, 1, stat_item)

        # 保留已有 checkbox 状态（如果存在），否则创建并默认勾选
        existing_widget = self.table_widget.cellWidget(row, 2)
        if existing_widget is None:
            chk = QtWidgets.QCheckBox()
            chk.setChecked(True)
            chk.setTristate(False)
            # 居中显示
            w = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(w)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setAlignment(QtCore.Qt.AlignCenter)
            hl.addWidget(chk)
            self.table_widget.setCellWidget(row, 2, w)

    def get_checked_montages(self) -> List[str]:
        """返回当前表格中被勾选的 montage 名称列表（按行检查第 2 列的 checkbox）。"""
        res = []
        for row in range(self.table_widget.rowCount()):
            cell_w = self.table_widget.cellWidget(row, 2)
            if cell_w is None:
                continue
            # wrapper widget -> layout -> checkbox
            cb = None
            # attempt to find QCheckBox inside wrapper
            for i in range(cell_w.layout().count()):
                w = cell_w.layout().itemAt(i).widget()
                if isinstance(w, QtWidgets.QCheckBox):
                    cb = w
                    break
            if cb and cb.isChecked():
                name_it = self.table_widget.item(row, 0)
                if name_it:
                    res.append(name_it.text())
        return res

    def set_selection_enabled(self, enabled: bool):
        """启用/禁用复选框交互。"""
        for row in range(self.table_widget.rowCount()):
            cell_w = self.table_widget.cellWidget(row, 2)
            if cell_w:
                # cell_w is a container widget, layout inside has the checkbox
                layout = cell_w.layout()
                for i in range(layout.count()):
                    w = layout.itemAt(i).widget()
                    if isinstance(w, QtWidgets.QCheckBox):
                        w.setEnabled(enabled)


class ViewerPanel(QtWidgets.QWidget):
    """Main interactive area: 3-column layout: Tile List | Canvas | Confirmation List"""

    tile_selected = QtCore.pyqtSignal(str)      # 选择Manual Confirmation List某行时发射mont_stem

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_montage: Optional[Montage] = None
        self.current_tile_name: Optional[str] = None
        self.selected_det: Optional[Detection] = None

        self.images_dir: Optional[Path] = None
        self.preds_dir: Optional[Path] = None
        self.project_root: Optional[Path] = None
        self.box_size: Optional[float] = None                   # 只会在新增box时被用到

        # 渲染状态变量
        self.base_image: Optional[np.ndarray] = None            # 原始灰度numpy图像数据（高度,宽度）
        self.display_scale: float = 0.2                         # 当前显示缩放倍数（相对于原始图像） 0.15 - 3
        # self._undo_stack: List[Dict] = []                       # 撤销操作栈：存储操作记录的字典列表

        self.pan_offset = QtCore.QPoint(0, 0)                   # display像素单位的平移偏移（用于绘制）
        self._panning = False
        self._pan_start = None
        self._pan_initial_offset = QtCore.QPoint(0, 0)          # 记录拖动开始时的初始偏移
        self._image_draw_pos: Tuple[int, int, int, int] = (0, 0, 0, 0)  # 记录上次渲染的 image 在 label 上的位置（x, y, w, h） 用于坐标转换
        self._brightness: float = 1.0                           # 亮度控制（0.0 - 4.0）
        self._auto_brightness_done: bool = False                # 标志位：是否已对当前Tile进行过自动亮度调节
        self.navigation_context = "tile_list"                   # Context Awareness("tile_list" or "manual_list")

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)

        # --- Left Column: Tile List & Manual List ---
        left_widget = QtWidgets.QWidget()
        left_vbox = QtWidgets.QVBoxLayout(left_widget)
        self.tile_list = QtWidgets.QListWidget()                    # 左侧切片列表
        self.tile_list.itemClicked.connect(self.on_tile_list_clicked)   # 连接切片选择信号
        self.manual_list = QtWidgets.QListWidget()
        self.manual_list.itemClicked.connect(self.on_manual_list_clicked)

        left_vbox.addWidget(QtWidgets.QLabel("Tile List"))
        left_vbox.addWidget(self.tile_list)
        left_vbox.addWidget(QtWidgets.QLabel("Manual Confirmation List"))
        left_vbox.addWidget(self.manual_list)
        left_widget.setMaximumWidth(240)

        # --- Center Column: Canvas & Basic Controls ---
        center_widget = QtWidgets.QWidget()
        center_vbox = QtWidgets.QVBoxLayout(center_widget)
        self.canvas_label = QtWidgets.QLabel()                       # 中部画布
        self.canvas_label.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas_label.setMinimumSize(512, 512)
        self.canvas_label.setMouseTracking(True)                    # 启用鼠标跟踪（即使不按下按钮也接收移动事件）
        self.canvas_label.setFocusPolicy(QtCore.Qt.StrongFocus)     # 设置强焦点策略，可以接收键盘事件
        self.canvas_label.installEventFilter(self)                  # 安装事件过滤器，拦截画布上的事件

        bottom = QtWidgets.QVBoxLayout()  # 改为垂直布局，包含两行

        row1 = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("◀")
        self.next_btn = QtWidgets.QPushButton("▶")
        row1.addWidget(self.prev_btn)
        row1.addWidget(self.next_btn)

        row1.addWidget(QtWidgets.QLabel("Brightness:"))
        row1.addWidget(QtWidgets.QLabel("40%"))
        self.brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brightness_slider.setRange(40, 400)
        self.brightness_slider.setValue(100)
        row1.addWidget(self.brightness_slider)
        row1.addWidget(QtWidgets.QLabel("400%"))

        row2 = QtWidgets.QHBoxLayout()
        self.spin_filter_conf = QtWidgets.QDoubleSpinBox()
        self.spin_filter_conf.setRange(0.0, 1.0)
        self.spin_filter_conf.setValue(0.1)
        self.spin_filter_conf.setSingleStep(0.01)
        row2.addWidget(QtWidgets.QLabel("Exclude conf <"))
        row2.addWidget(self.spin_filter_conf)

        self.filter_curr_btn = QtWidgets.QPushButton("Apply to Curr Tile")
        self.filter_all_btn = QtWidgets.QPushButton("Apply to All Tiles")
        row2.addWidget(self.filter_curr_btn)
        row2.addWidget(self.filter_all_btn)

        self.delete_btn = QtWidgets.QPushButton("Delete Sel Point")
        self.delete_all_btn = QtWidgets.QPushButton("Delete Curr Tile")
        self.save_btn = QtWidgets.QPushButton("Save")

        row2.addWidget(self.delete_btn)
        row2.addWidget(self.delete_all_btn)
        row2.addWidget(self.save_btn)

        self.log_box = QtWidgets.QTextEdit()                        # 最下层日志框
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("log")
        self.log_box.setMaximumHeight(60)

        bottom.addLayout(row1)
        bottom.addLayout(row2)
        bottom.addWidget(self.log_box)
        center_vbox.addWidget(self.canvas_label, 1)
        center_vbox.addLayout(bottom)
        layout.addWidget(left_widget)
        layout.addWidget(center_widget, 1)  # Canvas gets most space

        self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
        self.prev_btn.clicked.connect(self.go_prev)
        self.next_btn.clicked.connect(self.go_next)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.delete_all_btn.clicked.connect(self.delete_all_for_tile)
        self.save_btn.clicked.connect(self.save_and_confirm)
        self.filter_curr_btn.clicked.connect(lambda: self.apply_conf_filter(current_only=True))
        self.filter_all_btn.clicked.connect(lambda: self.apply_conf_filter(current_only=False))

    def set_dirs(self, nav_folder: Path, cfg: dict):
        """由 MainWindow 在 on_start 时设置 project 子目录路径"""
        self.project_root = nav_folder / cfg["project_name"]
        self.box_size = cfg["box_size"]
        self.images_dir, self.preds_dir = ensure_project_dirs(nav_folder, cfg["project_name"])
        self.refresh_manual_list()

    def refresh_manual_list(self, tile_name=None):
        """Reload manual confirmation list from file. Or just add tile_name"""
        if not self.project_root:
            return
        if tile_name is None:
            items = read_manual_list(self.project_root)
            self.manual_list.clear()
            self.manual_list.addItems(items)
        else:
            self.manual_list.addItem(tile_name)

    def on_tile_list_clicked(self, item: QtWidgets.QListWidgetItem):
        self.navigation_context = "tile_list"
        self.set_current_tile(item.text())
        self._update_another_list(self.tile_list.currentItem().text(), "manual_list")

    def on_manual_list_clicked(self, item: QtWidgets.QListWidgetItem):  # 当点击时，已经有了currentRow
        self.navigation_context = "manual_list"
        mont_stem, idx = match_name(item.text(), _TILE_RE)
        if idx != -1:
            self.tile_selected.emit(mont_stem)  # 更新self.current_montage
            self.set_current_tile(item.text())
            self._update_another_list(self.manual_list.currentItem().text(), "tile_list")

    def load_montage(self, montage: Montage, loadlast=True):
        """载入 montage：刷新 tile 列表，自动打开 tile第一项。"""
        self.current_montage = montage
        self.tile_list.clear()
        if not montage.tiles:
            self.clear_canvas()
            return

        for tile_name in sorted(montage.tiles.keys()):
            li = QtWidgets.QListWidgetItem(tile_name)
            self.tile_list.addItem(li)

        self._update_tile_item_colors()

        # Default select first
        if loadlast and self.tile_list.count() > 0:  # 因为没有点击事件，所以必须手动设置currentRow
            self.tile_list.setCurrentRow(self.tile_list.count() - 1)
            self.on_tile_list_clicked(self.tile_list.currentItem())

    def set_current_tile(self, tile_name: str):
        self.current_tile_name = tile_name
        tile = self.current_montage.tiles.get(tile_name)
        if not tile:
            self.clear_canvas()
            return

        img = cv2.imread(str(tile.tile_file), cv2.IMREAD_UNCHANGED)  # HxW int8
        if img is not None:  # Numpy array is not bool
            self.base_image = img[:, :, 0] if img.ndim == 3 else img  # only read 2D array
            self._auto_brightness_done = False  # Reset auto brightness flag for new image

        mont_name, idx = match_name(tile.name, _TILE_RE)
        pred_name = PRED_NAME_TEMPLATE.format(montage=mont_name, z=idx)
        pred_path = self.preds_dir / pred_name
        tile.detections = read_detections(pred_path)  # 同一时间，只会读一个tile的detections，避免占用内存太大

        self.pan_offset = QtCore.QPoint(0, 0)  # 重置平移（切片切换时重置为居中）
        self.selected_det = None        # 重置选择
        self._render_tile(tile)         # 渲染切片
        self.canvas_label.setFocus() #  确保画布获得焦点，以便接收键盘事件

    def log(self, msg: str):
        """向 log_box 追加一行带时间戳的消息"""
        ts = QtCore.QTime.currentTime().toString("HH:mm:ss")
        self.log_box.append(f"[{ts}] {msg}")

    def clear_canvas(self):
        self.base_image = None
        self.canvas_label.clear()

    def _render_tile(self, tile: Tile):
        """绘制 pixmap 并在其上绘制 detection 框，按 display_scale 缩放显示在 canvas_label 上"""
        if self.base_image is None:
            return

        # Auto Brightness
        if not self._auto_brightness_done:
            mean_val = np.mean(self.base_image)
            if mean_val > 1:  # Avoid division by zero
                factor = default_brightness / mean_val
                # Clamp factor to slider range 0.4 - 4.0
                factor = max(0.4, min(factor, 4.0))
                # Update internal brightness and Slider UI
                self._brightness = factor
                self.brightness_slider.blockSignals(True)  # Prevent recursive call
                self.brightness_slider.setValue(int(factor * 100))
                self.brightness_slider.blockSignals(False)
            self._auto_brightness_done = True

        # 如果没有修改亮度，直接复用已加载的 pixmap
        if self._brightness != 1.0:
            arr = self.base_image.astype(np.float32) * self._brightness
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            h, w = arr.shape
            qimg = QtGui.QImage(arr.data, w, h, w, QtGui.QImage.Format_Grayscale8)
            pix = QtGui.QPixmap.fromImage(qimg)
        else:
            pix = QtGui.QPixmap(str(tile.tile_file))

        # pix.size() 返回一个 QSize 对象，自动转换为int；KeepAspectRatio 会保持长宽比缩放并留空白
        scaled = pix.scaled(pix.size() * self.display_scale, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        label_w, label_h = self.canvas_label.width(), self.canvas_label.height()
        scaled_w, scaled_h = scaled.width(), scaled.height()
        # 计算放置位置：默认居中 + pan_offset
        x = int((label_w - scaled_w) // 2 + self.pan_offset.x())
        y = int((label_h - scaled_h) // 2 + self.pan_offset.y())
        self._image_draw_pos = (x, y, scaled_w, scaled_h)  # 保存当前图像绘制区域位置，方便坐标映射
        # 创建一个与 label 大小一致的画布 pixmap，在其上绘制缩放后的图像（并随后绘制框）
        canvas_pix = QtGui.QPixmap(self.canvas_label.size())
        canvas_pix.fill(QtCore.Qt.white)  # 背景填充
        painter = QtGui.QPainter(canvas_pix)
        painter.drawPixmap(x, y, scaled)
        font = QtGui.QFont("Arial")
        # 使用像素尺寸设定，避免点(size in points)引发跨设备/缩放不一致
        font.setPixelSize(max(11, int(11 * self.display_scale)))
        painter.setFont(font)

        first = True
        # 遍历所有检测框进行绘制；仅有两种状态：active和filtered；仅绘制active
        for det in tile.detections:
            if det.status == "active":
                x0 = int((det.x - det.w / 2) * self.display_scale) + x  # 左上角x坐标
                y0 = int((det.y - det.h / 2) * self.display_scale) + y  # 左上角y坐标
                w = int(det.w * self.display_scale)
                h = int(det.h * self.display_scale)

                color_name = STATUS_COLORS.get("active")
                if self.selected_det and det == self.selected_det:
                    color_name = STATUS_COLORS.get("processing")
                painter.setPen(QtGui.QPen(QtGui.QColor(color_name), 2))
                painter.drawRect(x0, y0, w, h)
                painter.drawText(x0, y0 - 6, f"conf: {det.conf:.2f}")  # Draw Conf (Top-Left)
                if first:
                    first = False
                    painter.drawText(x0, y0 + h + 11, f"1st point for tracking")

        painter.end()
        self.canvas_label.setPixmap(canvas_pix)             # 把最终画布放到 label（pixmap 大小等于 label 大小）
        self.canvas_label.repaint()                         # 强制重绘

    def go_prev(self):
        if self.current_montage is None or self.current_tile_name is None:
            return
        if self.navigation_context == "manual_list" and self.manual_list.count() > 0:
            row = self.manual_list.currentRow()
            prev_row = (row - 1) % self.manual_list.count()
            self.manual_list.setCurrentRow(prev_row)
            self.on_manual_list_clicked(self.manual_list.currentItem())
        elif self.navigation_context == "tile_list" and self.tile_list.count() > 0:
            row = self.tile_list.currentRow()
            prev_row = (row - 1) % self.tile_list.count()
            self.tile_list.setCurrentRow(prev_row)
            self.on_tile_list_clicked(self.tile_list.currentItem())


    def go_next(self):
        if self.current_montage is None or self.current_tile_name is None:
            return
        if self.navigation_context == "manual_list" and self.manual_list.count() > 0:
            row = self.manual_list.currentRow()
            next_row = (row + 1) % self.manual_list.count()
            self.manual_list.setCurrentRow(next_row)
            self.on_manual_list_clicked(self.manual_list.currentItem())
        elif self.navigation_context == "tile_list" and self.tile_list.count() > 0:
            row = self.tile_list.currentRow()
            next_row = (row + 1) % self.tile_list.count()
            self.tile_list.setCurrentRow(next_row)
            self.on_tile_list_clicked(self.tile_list.currentItem())

    def _update_another_list(self, query_text: str, another_context: str):
        if another_context == "tile_list":
            for i in range(self.tile_list.count()):
                it_text = self.tile_list.item(i).text()
                if it_text == query_text:
                    self.tile_list.setCurrentRow(i)
                    break
        elif another_context == "manual_list":
            for i in range(self.manual_list.count()):
                it_text = self.manual_list.item(i).text()
                if it_text == query_text:
                    self.manual_list.setCurrentRow(i)
                    break

    def on_brightness_changed(self, val: int):
        """slider 回调：val 0-400 映射到 0.0-4.0"""
        if self.current_montage is None or self.current_tile_name is None:
            return
        self._brightness = val / 100.0
        tile = self.current_montage.tiles.get(self.current_tile_name)
        self._render_tile(tile)

    def delete_selected(self):
        if self.selected_det is None or self.current_montage is None or self.current_tile_name is None:
            return
        tile = self.current_montage.tiles.get(self.current_tile_name)
        for d in tile.detections:
            if d == self.selected_det:
                tile.detections.remove(d)
                break
        self.selected_det = None
        self._render_tile(tile)

    def delete_all_for_tile(self):
        if self.current_montage is None or self.current_tile_name is None:
            return
        tile = self.current_montage.tiles.get(self.current_tile_name)
        tile.detections = []
        self.selected_det = None
        self._render_tile(tile)

    def move_selected_detection(self, tile, img_x, img_y):
        # 更新检测框位置
        self.selected_det.x = img_x
        self.selected_det.y = img_y
        self._render_tile(tile)

    def add_new_detection(self, tile, img_x, img_y):
        cls, w, h = 0, self.box_size, self.box_size
        new_det = Detection(cls, img_x, img_y, w, h, 2.0, "active")
        tile.detections.insert(0, new_det)  # 最后一个会被放到最开始作为对中点
        self.selected_det = new_det
        self._render_tile(tile)

    def apply_conf_filter(self, current_only: bool):
        """Mark dets with conf < threshold as filtered."""
        if self.current_montage is None or self.current_tile_name is None:
            return
        thresh = self.spin_filter_conf.value()
        tile_path_to_process = []
        if current_only:
            mont, idx = match_name(self.current_tile_name, _TILE_RE)
            pred_file = PRED_NAME_TEMPLATE.format(montage=mont, z=idx)
            tile_path_to_process.append(self.preds_dir / pred_file)
        else:
            for p in sorted(self.preds_dir.glob("*.txt")):
                mont, idx = match_name(p.name, _PRED_RE)
                if idx > -1:
                    tile_path_to_process.append(p)

        del_count = 0
        recover_count = 0
        for tp in tile_path_to_process:
            dets = read_detections(tp)
            for d in dets:
                if d.conf < thresh:
                    d.status = "filtered"
                    del_count += 1
                elif d.conf >= thresh and d.status == "filtered":  # 对于先前被高阈值删除的，则恢复它！
                    d.status = "active"
                    recover_count += 1
            write_detections(tp, dets)
        if current_only:
            pre = f"For {tile_path_to_process[0]}: "
        else:
            pre = "For all available files: "
        msg = pre + f"Filtered {del_count} detections with conf < {thresh:.2f}. Restored {recover_count} detections. Auto-Saved."
        self.log(msg)
        logger.info(msg)
        self.set_current_tile(self.current_tile_name)

    def save_and_confirm(self):
        if self.current_montage is None or self.current_tile_name is None:
            return

        tile = self.current_montage.tiles.get(self.current_tile_name)
        try:
            mont_name, idx = match_name(tile.name, _TILE_RE)
            pred_name = PRED_NAME_TEMPLATE.format(montage=mont_name, z=idx)
            pred_path = self.preds_dir / pred_name
            write_detections(pred_path, tile.detections)
            msg = f"Saved predictions: {pred_path}"

            curr_row = -1
            for i in range(self.manual_list.count()):
                item_text = self.manual_list.item(i).text()
                if item_text == tile.name:
                    remove_from_manual_list(self.project_root, item_text)
                    self.manual_list.takeItem(i)
                    curr_row = i
                    break

            if 0 <= curr_row < self.manual_list.count():  # Move to next in list
                self.manual_list.setCurrentRow(curr_row)
                self.on_manual_list_clicked(self.manual_list.currentItem())

            self._update_tile_item_colors()

        except Exception as e:
            msg = f"Failed to save predictions: {e}"
            logger.error(msg)
        self.log(msg)

    def _update_tile_item_colors(self):
        """遍历 tile_list，读取每个 tile 的 pred 文件；如果没有任何 status == 'active' 的 detection 就把该 item 背景设为浅红色"""
        if not self.current_montage:
            return

        for i in range(self.tile_list.count()):
            item = self.tile_list.item(i)
            tile_name = item.text()
            mont_name, idx = match_name(tile_name, _TILE_RE)
            if idx == -1:
                continue
            pred_name = PRED_NAME_TEMPLATE.format(montage=mont_name, z=idx)
            pred_path = self.preds_dir / pred_name

            has_active = False
            dets = read_detections(pred_path)
            for d in dets:
                if d.status == "active" and d.conf <= 1.0:
                    has_active = True
                    break

            if not has_active:
                item.setBackground(QtGui.QBrush(QtGui.QColor(STATUS_COLORS.get("deleted"))))
            else:
                # Reset background
                item.setBackground(QtGui.QBrush(QtCore.Qt.NoBrush))

    def _screen_to_image_coords(self, screen_pos: QtCore.QPoint) -> Tuple[float, float]:
        """将屏幕坐标（相对于 canvas_label）转换为图像像素坐标（原始图像坐标，非显示坐标）"""
        if not self.canvas_label.pixmap():
            return 0.0, 0.0

        x, y, scaled_w, scaled_h = self._image_draw_pos
        # 计算 adjusted 在显示（scaled）像素坐标系内
        adjusted_x = screen_pos.x() - x
        adjusted_y = screen_pos.y() - y

        # 转换为原始图像坐标（除以 display_scale）
        img_x = adjusted_x / self.display_scale
        img_y = adjusted_y / self.display_scale

        # 边界检查
        if self.base_image is not None:
            h, w = self.base_image.shape
            img_x = max(0.0, min(img_x, float(w)))
            img_y = max(0.0, min(img_y, float(h)))

        return img_x, img_y

    def eventFilter(self, obj, event):
        """keyboard navigation and canvas events"""
        if obj is self.canvas_label:
            # 添加安全检查：如果没有当前tile或没有基础图像，不处理事件
            if self.base_image is None or self.current_montage is None or self.current_tile_name is None:
                return super().eventFilter(obj, event)

            if event.type() == QtCore.QEvent.KeyPress:  # 键盘按键事件
                if event.key() == QtCore.Qt.Key_Left:
                    self.go_prev()      # 左箭头 -> 上一个切片
                if event.key() == QtCore.Qt.Key_Right:
                    self.go_next()      # 右箭头 -> 下一个切片
                if event.key() == QtCore.Qt.Key_S:
                    self.save_and_confirm()  # S -> 按 S 或 s 更新manual_list并保存predictions
                if event.key() == QtCore.Qt.Key_Delete:
                    if event.modifiers() & QtCore.Qt.ShiftModifier:
                        self.delete_all_for_tile()      # 处理Shift+Delete：删除所有检测框
                    else:
                        self.delete_selected()          # 处理Delete：删除选中的检测框
                return True

            if event.type() == QtCore.QEvent.Wheel:     # 鼠标滚轮事件 - 缩放功能
                delta = event.angleDelta().y()  # 获取滚轮增量
                if delta > 0:
                    self.display_scale *= 1.15  # 向上滚动 -> 放大
                else:
                    self.display_scale /= 1.15  # 向下滚动 -> 缩小

                self.display_scale = min(max(self.display_scale, 0.15), 3.0)    # 限制缩放范围在0.1到3.0之间
                tile = self.current_montage.tiles.get(self.current_tile_name)
                self._render_tile(tile)
                return True

            if event.type() == QtCore.QEvent.MouseButtonPress:          # 鼠标按下事件
                pos = event.pos()                                       # 获取鼠标位置
                tile = self.current_montage.tiles.get(self.current_tile_name)

                if event.button() == QtCore.Qt.LeftButton:              # 【左键按下】：初始化平移状态，暂时不执行点击逻辑
                    self._panning = False                               # 先设为False，只有拖动距离超过阈值才视为平移
                    self._pan_start = pos
                    self._pan_initial_offset = self.pan_offset          # 记录当前偏移量作为起点
                    return True
                elif event.button() == QtCore.Qt.RightButton:           # 【右键点击】：移动已选中的框到当前位置
                    if self.selected_det:                               # 有选中框：移动到点击位置
                        img_x, img_y = self._screen_to_image_coords(pos)
                        self.move_selected_detection(tile, img_x, img_y)
                    return True

            if event.type() == QtCore.QEvent.MouseMove:                 # 只有左键按住时才处理平移
                if event.buttons() & QtCore.Qt.LeftButton:
                    if self._pan_start:
                        # 计算移动距离，设置一个小阈值(5像素)防止点击抖动被误判为拖拽
                        if not self._panning:
                            dist = (event.pos() - self._pan_start).manhattanLength()
                            if dist > 5:
                                self._panning = True
                        # 如果确认为平移模式，则更新画布
                        if self._panning:
                            self.pan_offset = self._pan_initial_offset + (event.pos() - self._pan_start)
                            tile = self.current_montage.tiles.get(self.current_tile_name)
                            self._render_tile(tile)
                    return True

            if event.type() == QtCore.QEvent.MouseButtonRelease:    # 鼠标释放
                if event.button() == QtCore.Qt.LeftButton:
                    if self._panning:                               # 如果刚刚是平移操作，释放时只需重置状态
                        self._panning = False
                        self._pan_start = None
                    else:
                        pos = event.pos()  # 获取鼠标位置
                        img_x, img_y = self._screen_to_image_coords(pos)
                        tile = self.current_montage.tiles.get(self.current_tile_name)
                        clicked = None
                        for det in tile.detections:                 # 碰撞检测：检查是否点在某个框上
                            if det.x - det.w / 2 <= img_x <= det.x + det.w / 2 and det.y - det.h / 2 <= img_y <= det.y + det.h / 2:
                                clicked = det
                                break
                        if clicked:                                 # 点击了检测框：选中它
                            self.selected_det = clicked
                        else:                                       # 点击了空白区域，新建框
                            self.add_new_detection(tile, img_x, img_y)
                        self._render_tile(tile)
                    return True

        return super().eventFilter(obj, event)

# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Picker")
        icon = get_resource_path("data/pp_gemini.ico")
        self.setWindowIcon(QtGui.QIcon(str(icon)))
        self.resize(1200, 800)

        # 初始化内部处理管道的数据队列 - 生产者-消费者模式的关键组件
        self.job_queue: Queue = Queue()         # 作业队列：主线程向工作线程提交处理任务
        self.ui_queue: Queue = Queue()          # UI更新队列：工作线程向主线程发送UI更新请求
        self.montages: Dict[str, Montage] = {}  # name -> Montage

        # 初始化三个主要的面板组件
        self.settings = SettingsPanel()
        self.status = StatusPanel()
        self.viewer = ViewerPanel()

        # 创建中央窗口部件和主布局 - Qt应用程序的标准模式
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.settings)
        left.addWidget(self.status)

        layout.addLayout(left, 0)           # 拉伸因子为0（固定宽度）
        layout.addWidget(self.viewer, 1)    # 拉伸因子为1（占据剩余空间）

        # 建立信号-槽连接 - Qt框架的事件处理机制
        self.settings.start_requested.connect(self.on_start)                # 设置面板的开始信号连接到处理函数
        self.settings.stop_requested.connect(self.on_stop)
        self.settings.export_requested.connect(self.on_export)
        self.settings.nav_changed.connect(self.on_nav_set)
        self.status.montage_selected.connect(self.on_montage_selected)      # 状态面板的选择信号
        self.viewer.tile_selected.connect(self.on_tile_selected)  # viewer面板的Manual List的选择tile信号

        self.worker: Optional[MontageProcessor] = None                      # 工作进程
        self.observer: Optional[Observer] = None                            # 文件系统观察者，用于监控新文件

        self.timer = QtCore.QTimer()                                        # 创建定时器用于定期从UI队列中处理更新
        self.timer.timeout.connect(self.process_ui_queue)
        self.timer.start(200)                                               # 启动定时器，每200毫秒触发一次

    def on_nav_set(self, nav_path: Path):
        """仅用于在未运行时预览并刷新 StatusPanel（不启动 worker）。"""
        try:
            project_name = self.settings.project_name.text().strip()
            if not project_name:
                self.viewer.log("Empty project name, skipped montages preview.")
                self.montages = preview_nav_montages(nav_path)
            else:
                self.montages = load_nav_and_montages(nav_path, project_name, False)
                self.viewer.set_dirs(nav_path.parent,{"project_name": project_name, "box_size": self.settings.box_size.value()})
            self.status.refresh(self.montages, delete_previous=True)
        except Exception as e:
            msg = f"Failed to preview {nav_path}: {e}"
            self.viewer.log(msg)
            logger.error(msg)

    def on_start(self, cfg: dict):
        """Initialize workers and load data."""
        try:
            nav_folder = cfg["nav_path"].parent
            self.montages = load_nav_and_montages(cfg["nav_path"], cfg["project_name"], cfg["overwrite"])
            self.status.refresh(self.montages, delete_previous=False)

            self.viewer.set_dirs(nav_folder, cfg)
            self.status.set_selection_enabled(False)  # 禁用 UI 交互

            for m in self.montages.values():
                if m.name not in self.status.get_checked_montages():
                    m.status = "excluded"
                    self.ui_queue.put(("update_montage_status", (m, None)))

            # Processing Worker
            detector = YoLoWrapper(str(cfg["model_path"]))
            self.worker = MontageProcessor(self.ui_queue, self.job_queue, detector, cfg)
            self.worker.start()  # 启动工作线程，在后台持续监听 job_queue

            if self.observer:
                self.observer.stop()
            self.observer = Observer()
            handler = MontageWatcher(cfg["nav_path"], self.ui_queue, self.job_queue, self.montages)
            self.observer.schedule(handler, str(nav_folder), recursive=False)
            self.observer.start()

            # daemon=True 表示如果主程序关闭，这个扫描线程也会自动随之关闭
            scan_thread = threading.Thread(target=handler.scan_existing_files_async, daemon=True)
            scan_thread.start()

            msg = f"Processing started. Monitoring directory {nav_folder}..."
            logger.info(msg)
            self.viewer.log(msg)

        except Exception as e:
            logger.error(f"Startup failed: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            # If error occurs, re-enable controls
            self.settings.on_stop()  # Reset settings buttons
            self.status.set_selection_enabled(True)

    def on_stop(self):
        # Enable controls in Status Panel
        self.status.set_selection_enabled(True)

        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=2.0)
            self.observer = None

        if self.worker and self.worker.isRunning():
            self.worker.stop()
            # Wait a bit for it to finish current tile, but don't freeze UI forever
            self.worker.wait(10000)

        logger.info("Processing stopped.")

    def on_export(self, cfg: dict):
        # reordered export
        msg = add_predictions_to_nav(cfg["nav_path"], cfg["project_name"], cfg["save_path"], self.status.get_checked_montages(), cfg["box_size"])
        self.viewer.log(msg)
        logger.info(msg)

    def on_montage_selected(self, name: str):  # load montage 选第0个
        if name in self.montages:
            self.viewer.load_montage(self.montages[name], loadlast=True)

    def on_tile_selected(self, mont_stem: str):
        last_key = list(self.montages.keys())[-1]
        name = mont_stem + str(self.montages[last_key].map_file.suffix)
        if name in self.montages:
            self.viewer.load_montage(self.montages[name], loadlast=False)

    def process_ui_queue(self):
        """Called from main thread to process updates from worker threads"""
        while not self.ui_queue.empty():
            try:
                cmd = self.ui_queue.get_nowait()
                if cmd[0] == "update_montage_status":
                    mont, msg = cmd[1]
                    if msg:
                        self.viewer.log(msg)
                    self.status.update_montage(mont)
                elif cmd[0] == "add_tile_item":
                    mont, msg = cmd[1]
                    self.viewer.log(msg)
                    if self.viewer.current_montage and self.viewer.current_montage == mont:
                        self.viewer.load_montage(mont, loadlast=True)
                elif cmd[0] == "add_manual_item":
                    tile_name, msg = cmd[1]
                    self.viewer.log(msg)
                    self.viewer.refresh_manual_list(tile_name)
            except Empty:
                break

    def closeEvent(self, event):
        # Stop logic same as on_stop
        self.on_stop()

        # Stop timer last to ensure any final UI updates from worker shutdown are processed
        if self.timer and self.timer.isActive():
            self.timer.stop()

        super().closeEvent(event)


def main():
    # logging 模块提供了六个标准的日志信息级别，用于表示事件的严重程度，从低到高排列：DEBUG, INFO, WARNING, ERROR, CRITICAL, NOTEST
    logging.basicConfig(
        level=logging.DEBUG,                                            # 显示 debug 及以上信息
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[
            logging.FileHandler("pp.log", encoding="utf-8"),    # 写入文件
            logging.StreamHandler()                                     # 同时打印到控制台
        ]
    )
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
