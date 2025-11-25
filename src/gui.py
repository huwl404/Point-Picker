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
 - Background workers: splitting + prediction pipeline; filesystem watcher
"""
from __future__ import annotations
import sys
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
    _TILE_RE, remove_from_manual_list

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -----------------------------
# Filesystem Watcher
# -----------------------------
class MontageWatcher(FileSystemEventHandler):
    """Monitors directory for new stable MRC files and triggers processing."""

    def __init__(self, monitored_path: Path, ui_queue: Queue, job_queue: Queue, montages: Dict[str, Montage]):
        super().__init__()
        self.monitored_path = monitored_path
        self.ui_queue = ui_queue
        self.job_queue = job_queue
        self.montages = montages
        self.processed_files = set()                        # 记录已处理的文件，避免重复处理
        self._scan_existing_files()

    def _scan_existing_files(self):
        """启动监控前扫描已有的文件并处理"""
        mont = self.montages
        for p in self.monitored_path.glob("*"):
            p = p.resolve()
            if p not in self.processed_files and self._is_file_stable(p, check_interval=0):
                self._process_mrc_file(p)

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
            logger.debug(f"Montage monitoring failed: {file_path}")
            return False

    def _process_mrc_file(self, file_path: Path):
        """处理稳定的montage文件"""
        mont = self.montages.get(file_path.name)
        if not mont:
            return

        if mont.status == "not generated":
            mont.status = "queuing"
            mont.map_file = file_path
            self.ui_queue.put(("update_montage", mont))
            self.job_queue.put(mont)
            self.processed_files.add(file_path.resolve())
        elif mont.status == "processed":
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
        self._total_id = 0

    def predict_image(self, img_path: str, cfg: dict) -> Optional[List[Detection]]:
        # cfg keys expected: model_path, img_size, box_size, max_detection, conf, iou, device (either 'cpu' or [GPU index list])
        dets: List[Detection] = []
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
                    self._total_id += 1
                    dets.append(Detection(self._total_id, cls, x, y, w, h, conf, "active"))

                total_num = len(dets)
                s_pre = round(r.speed['preprocess'], 1)
                s_i = round(r.speed['inference'], 1)
                s_post= round(r.speed['postprocess'], 1)
                logger.info(f"Predicted {total_num} targets: "
                            f"{s_pre}ms preprocess, {s_i}ms inference, {s_post}ms postprocess for {img_path}")
        except Exception as e:
            logger.error(f"Inference error on {img_path}: {e}")

        return dets


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
                self.ui_queue.put(("update_montage", montage))
                logger.error(f"Processing failed for {montage.name}: {e}")
            finally:
                self.job_queue.task_done()

    def _process_single_montage(self, montage: Montage, images_dir: Path, preds_dir: Path):
        """Handle splitting, prediction, and feature logic for one montage."""
        montage.status = "processing"
        self.ui_queue.put(("update_montage", montage))

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
                    img_data = mrc.data[z].astype(np.uint16).astype(np.float16)  # To avoid transforming to float64 to compute img_norm
                    img_norm = ((img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255).astype(np.uint8)  # Ultralytics only accept int8 images to be processed
                    if not tile_path.exists() or self.cfg.get('overwrite', False):  # write png if not existing or overwrite
                        cv2.imwrite(str(tile_path), img_norm)
                    tile = Tile(name=tile_name, tile_id=z, tile_file=tile_path)

                    # 3. Prediction & Find center
                    if not pred_path.exists() or self.cfg.get('overwrite', False):
                        # A. Run YOLO
                        dets = self.predictor.predict_image(str(tile_path), self.cfg)
                        # B. Find Center Point
                        start = time.time()
                        center_det = self._find_center_point(img_norm, dets)

                        if center_det:  # Add to front of list
                            dets.insert(0, center_det)
                        else:  # Not found: Signal UI to add to manual list
                            append_to_manual_list(self.project_root, tile_name)
                            self.ui_queue.put(("add_manual_item", tile_name))
                        period = time.time() - start
                        logger.info(f"_find_center_points takes {period} seconds.")

                        write_detections(pred_path, dets)

                    montage.tiles[z] = tile
                    self.ui_queue.put(("refresh_tile_list", montage.name))
            montage.status = "processed"
            self.ui_queue.put(("update_montage", montage))
        except Exception as e:
            logger.error(f"Error reading MRC {montage.map_file}: {e}")
            raise e

    def _find_center_point(self, img: np.ndarray, existing_dets: List[Detection]) -> Optional[Detection]:
        """
        Algorithm to find a suitable carbon film spot in the center 1/16th area.
        Constraints: No overlap with YOLO boxes, pixels not close to 0 or 255, darker than surroundings.
        """
        h, w = img.shape
        box_size = int(self.cfg["box_size"])
        half_box = box_size // 2

        # Define Center 1/16th area (1/4 width * 1/4 height centered)
        # Range x: [3w/8, 5w/8], Range y: [3h/8, 5h/8]
        x_start, x_end = int(3 * w / 8), int(5 * w / 8)
        y_start, y_end = int(3 * h / 8), int(5 * h / 8)

        # Heuristic: Search this area with a stride
        stride = max(10, box_size // 4)
        candidates = []

        global_mean = np.mean(img)
        for y in range(y_start, y_end - box_size, stride):
            for x in range(x_start, x_end - box_size, stride):
                # Crop the potential box area
                roi = img[y: y + box_size, x: x + box_size]

                # Check 1: Pixel value extremes (Ice vs Vacuum)
                # "Not too close to 0 (ice), not too close to 255 (vacuum)"
                if np.min(roi) < 20 or np.max(roi) > 230:
                    continue

                mean_val = np.mean(roi)
                # Check 2: Darker than surroundings (Carbon is darker than ice/vacuum usually in normalized)
                if mean_val >= global_mean * 0.95:
                    continue

                # Check 3: Overlap with existing YOLO boxes
                cx, cy = x + half_box, y + half_box
                if self._check_overlap(cx, cy, box_size, existing_dets):
                    continue

                candidates.append((mean_val, cx, cy))

        if not candidates:
            return None

        # pick the one closest to absolute center of image to be safe
        img_cx, img_cy = w / 2, h / 2
        candidates.sort(key=lambda c: (c[1] - img_cx) ** 2 + (c[2] - img_cy) ** 2)
        best = candidates[0]
        return Detection(
            id=0, cls=0,
            x=best[1], y=best[2], w=float(box_size), h=float(box_size),
            conf=2.0, status="active")  # conf 2.0 indicates generated point

    @staticmethod
    def _check_overlap(cx, cy, size, dets) -> bool:
        """Simple box overlap check."""
        r1_x1, r1_y1 = cx - size / 2, cy - size / 2
        r1_x2, r1_y2 = cx + size / 2, cy + size / 2

        for d in dets:
            r2_x1, r2_y1 = d.x - d.w / 2, d.y - d.h / 2
            r2_x2, r2_y2 = d.x + d.w / 2, d.y + d.h / 2

            # Intersection
            dx = min(r1_x2, r2_x2) - max(r1_x1, r2_x1)
            dy = min(r1_y2, r2_y2) - max(r1_y1, r2_y1)
            if dx > 0 and dy > 0:
                return True
        return False


# -----------------------------
# GUI Components
# -----------------------------
class SettingsPanel(QtWidgets.QWidget):
    """Configuration panel for paths and detection parameters."""

    # 定义自定义信号 - Qt框架中组件间通信的核心机制
    # settings_changed = QtCore.pyqtSignal(dict)        # 当运行时，任何设置改变时发射
    start_requested = QtCore.pyqtSignal(dict)           # 点击 RUN 时发射
    stop_requested = QtCore.pyqtSignal()                # 点击 STOP 时发射
    export_requested = QtCore.pyqtSignal(dict)          # 点击export时发射

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gpus = self._detect_gpu()                  # 仅需初始运行一次
        self._build_ui()
        # self._connect_change_signals()                # 连接控件变化到 settings_changed

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

    @staticmethod
    def get_resource_path(relative_path: str) -> Path:
        """兼容 PyInstaller 打包与源码运行"""
        if hasattr(sys, "_MEIPASS"):                            # 运行在 PyInstaller 打包环境
            base_path = Path(sys._MEIPASS)
        else:
            base_path = Path(__file__).resolve().parent.parent  # src 的上级目录

        return (base_path / relative_path).resolve()

    def _build_ui(self):
        """UI构建: project_name, model_path, nav_path, box_size, max_detection, conf, iou, device_combo"""
        # 使用表单布局管理器 - 适合标签-字段对的排列
        layout = QtWidgets.QFormLayout(self)

        self.project_name = QtWidgets.QLineEdit()       # 项目名称文本输入框
        self.project_name.setText("Point-Picker")
        layout.addRow("Project name:", self.project_name)

        # Model line with Browse
        self.model_path = QtWidgets.QLineEdit()         # 模型文件路径文本输入框
        default_model = self.get_resource_path("data/md2_pm2_best.pt")
        self.model_path.setText(str(default_model))
        self.model_path_btn = QtWidgets.QPushButton("Browse")

        model_h = QtWidgets.QHBoxLayout()
        model_h.addWidget(self.model_path)
        model_h.addWidget(self.model_path_btn)
        layout.addRow("Model path:", model_h)

        # Nav line with Browse
        self.nav_path = QtWidgets.QLineEdit()            # 导航文件路径文本输入框
        default_nav = self.get_resource_path("test/nav001.nav")         # for test
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

        # iou (0.0 - 1.0)
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
        self.run_btn = QtWidgets.QPushButton("RUN")
        self.stop_btn = QtWidgets.QPushButton("STOP")
        self.export_btn = QtWidgets.QPushButton("Export")
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
        self.run_btn.clicked.connect(self.on_run)
        self.stop_btn.clicked.connect(self.on_stop)
        self.export_btn.clicked.connect(self.on_export)

    def on_browse_model(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select model file", filter="*.pt *.yaml *.pth")
        if f:
            self.model_path.setText(f)

    def on_browse_nav(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select nav file", filter="*.nav")
        if f:
            self.nav_path.setText(f)

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

    def on_stop(self):
        self.stop_requested.emit()
        self.stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)

    def on_export(self):
        """发射export请求"""
        cfg = {
            "project_name": self.project_name.text().strip(),
            "nav_path": Path(self.nav_path.text().strip()) if self.nav_path.text().strip() else None,
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

        # 创建表格部件，设置2列：名称和状态
        self.table_widget = QtWidgets.QTableWidget(0, 2)                                # 2列：名称和状态
        self.table_widget.setHorizontalHeaderLabels(["Montage Name", "Status"])         # 设置列标题
        # 设置表格属性
        self.table_widget.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)  # 第一列拉伸
        self.table_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)  # 整行选择
        self.table_widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)   # 不可编辑
        self.table_widget.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.table_widget)

    def on_selection_changed(self):
        """处理表格选择变化"""
        selected_items = self.table_widget.selectedItems()
        if selected_items:
            name = selected_items[0].text()                                 # 第一列是名称
            self.montage_selected.emit(name)

    def refresh(self):
        """刷新表格显示"""
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


class ViewerPanel(QtWidgets.QWidget):
    """Main interactive area: 3-column layout: Tile List | Canvas | Confirmation List"""

    tile_selected = QtCore.pyqtSignal(str)      # 选择Manual Confirmation List某行时发射tile name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_montage: Optional[Montage] = None
        self.current_tile_id: Optional[int] = None              # 只知道id，不知道归属于哪个montage
        self.selected_det: Optional[Detection] = None

        self.images_dir: Optional[Path] = None
        self.preds_dir: Optional[Path] = None
        self.project_root: Optional[Path] = None

        # 渲染状态变量
        self.base_image: Optional[np.ndarray] = None            # 原始灰度numpy图像数据（高度,宽度）
        self.display_scale: float = 0.2                         # 当前显示缩放倍数（相对于原始图像）
        self._undo_stack: List[Dict] = []                       # 撤销操作栈：存储操作记录的字典列表

        self.pan_offset = QtCore.QPoint(0, 0)                   # display像素单位的平移偏移（用于绘制）
        self._panning = False
        self._pan_start = None
        self._pan_start_offset = QtCore.QPoint(0, 0)
        self._image_draw_pos: Tuple[int, int, int, int] = (0, 0, 0, 0)  # 记录上次渲染的 image 在 label 上的位置（x, y, w, h） 用于坐标转换
        self._brightness: float = 1.0                           # 亮度控制（0.0 - 4.0）
        self.navigation_context = "all_tiles"                   # Context Awareness("all_tiles" or "manual_list")

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)

        # --- Left Column: Tile List & Log ---
        left_widget = QtWidgets.QWidget()
        left_vbox = QtWidgets.QVBoxLayout(left_widget)
        self.tile_list = QtWidgets.QListWidget()                    # 左侧切片列表
        self.tile_list.itemClicked.connect(self.on_tile_list_clicked)   # 连接切片选择信号

        self.log_box = QtWidgets.QTextEdit()                        # 左侧日志框
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(150)

        left_vbox.addWidget(QtWidgets.QLabel("Tile List"))
        left_vbox.addWidget(self.tile_list)
        left_vbox.addWidget(QtWidgets.QLabel("Log"))
        left_vbox.addWidget(self.log_box)
        left_widget.setMaximumWidth(240)

        # --- Center Column: Canvas & Basic Controls ---
        center_widget = QtWidgets.QWidget()
        center_vbox = QtWidgets.QVBoxLayout(center_widget)
        self.canvas_label = QtWidgets.QLabel()                       # 中部画布
        self.canvas_label.setMinimumSize(512, 512)
        self.canvas_label.setAlignment(QtCore.Qt.AlignCenter)
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

        self.filter_curr_btn = QtWidgets.QPushButton("Apply to Curr")
        self.filter_all_btn = QtWidgets.QPushButton("Apply to All")
        row2.addWidget(self.filter_curr_btn)
        row2.addWidget(self.filter_all_btn)

        self.delete_btn = QtWidgets.QPushButton("Delete Sel")
        self.restore_btn = QtWidgets.QPushButton("Restore Sel")
        self.delete_all_btn = QtWidgets.QPushButton("Delete Curr")
        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.save_btn = QtWidgets.QPushButton("Save")

        row2.addWidget(self.delete_btn)
        row2.addWidget(self.restore_btn)
        row2.addWidget(self.delete_all_btn)
        row2.addWidget(self.undo_btn)
        row2.addWidget(self.save_btn)

        bottom.addLayout(row1)
        bottom.addLayout(row2)
        center_vbox.addWidget(self.canvas_label, 1)
        center_vbox.addLayout(bottom)

        # --- Right Column: Manual List & Operation Hint---
        right_widget = QtWidgets.QWidget()
        right_vbox = QtWidgets.QVBoxLayout(right_widget)
        self.manual_list_widget = QtWidgets.QListWidget()
        self.manual_list_widget.itemClicked.connect(self.on_manual_list_clicked)
        right_vbox.addWidget(QtWidgets.QLabel("Manual Confirmation List"))
        right_vbox.addWidget(self.manual_list_widget)
        right_widget.setMaximumWidth(240)

        layout.addWidget(left_widget)
        layout.addWidget(center_widget, 1)  # Canvas gets most space
        layout.addWidget(right_widget)

        self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
        self.prev_btn.clicked.connect(self.go_prev)
        self.next_btn.clicked.connect(self.go_next)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.restore_btn.clicked.connect(self.restore_selected)
        self.delete_all_btn.clicked.connect(self.delete_all_for_tile)
        self.undo_btn.clicked.connect(self.undo_last)
        self.save_btn.clicked.connect(self.confirm_and_save)
        self.filter_curr_btn.clicked.connect(lambda: self.apply_conf_filter(current_only=True))
        self.filter_all_btn.clicked.connect(lambda: self.apply_conf_filter(current_only=False))

    def set_dirs(self, nav_folder: Path, project_name: str):
        """由 MainWindow 在 on_start 时设置 project 子目录路径"""
        self.project_root = nav_folder / project_name
        self.images_dir, self.preds_dir = ensure_project_dirs(nav_folder, project_name)
        self.refresh_manual_list()

    def refresh_manual_list(self, tile_name=None):
        """Reload manual confirmation list from file. Or just add tile_name"""
        if not self.project_root:
            return
        if tile_name is None:
            items = read_manual_list(self.project_root)
            self.manual_list_widget.clear()
            self.manual_list_widget.addItems(items)
        else:
            self.manual_list_widget.addItem(tile_name)

    def on_tile_list_clicked(self, item: QtWidgets.QListWidgetItem):
        """Handle click on main tile list (Left). Sets context to 'all_tiles'."""
        self.navigation_context = "all_tiles"
        idx = item.data(QtCore.Qt.UserRole)
        self.set_current_tile(idx)

    def on_manual_list_clicked(self, item: QtWidgets.QListWidgetItem):
        self.navigation_context = "manual_list"
        self.tile_selected.emit(item.text())

    def load_montage(self, montage: Montage):
        """载入 montage：刷新 tile 列表，自动打开 tile第一项。"""
        self.current_montage = montage
        self.tile_list.clear()
        if not montage.tiles:
            self.clear_canvas()
            return

        for idx in sorted(montage.tiles.keys()):
            tile = montage.tiles[idx]
            li = QtWidgets.QListWidgetItem(f"{tile.name}")
            li.setBackground(QtGui.QColor(STATUS_COLORS.get(tile.status)))
            li.setData(QtCore.Qt.UserRole, idx)         # 将切片索引存储在用户数据中，便于后续检索
            self.tile_list.addItem(li)

        # Default select first
        if self.tile_list.count() > 0:
            self.set_current_tile(self.tile_list.item(0).data(QtCore.Qt.UserRole))

    def set_current_tile(self, idx: int):
        self.current_tile_id = idx
        tile = self.current_montage.tiles.get(idx)
        if not tile:
            self.clear_canvas()
            return

        img = cv2.imread(str(tile.tile_file), cv2.IMREAD_UNCHANGED)  # HxW int8
        if img is not None:  # Numpy array is not bool
            self.base_image = img[:, :, 0] if img.ndim == 3 else img  # only read 2D array

        mont_name, idx = match_name(tile.name, _TILE_RE)
        pred_name = PRED_NAME_TEMPLATE.format(montage=mont_name, z=idx)
        pred_path = self.preds_dir / pred_name
        tile.detections = read_detections(pred_path)

        self.pan_offset = QtCore.QPoint(0, 0)  # 重置平移（切片切换时重置为居中）
        self.selected_det = tile.detections[0]        # 默认选第一个
        self._undo_stack.clear()        # 重置撤销stack
        self._render_tile(tile)         # 渲染切片
        self.canvas_label.setFocus() #  确保画布获得焦点，以便接收键盘事件

        # 更新切片列表中的选中高亮
        if self.navigation_context == "all_tiles":
            for i in range(self.tile_list.count()):
                it = self.tile_list.item(i)
                if it.data(QtCore.Qt.UserRole) == idx:
                    self.tile_list.setCurrentItem(it)
                    break

    def log(self, msg: str):
        """向 log_box 追加一行带时间戳的消息"""
        ts = QtCore.QTime.currentTime().toString("HH:mm:ss")
        self.log_box.append(f"[{ts}] {msg}\n")

    def clear_canvas(self):
        self.base_image = None
        self.canvas_label.clear()

    def _render_tile(self, tile: Tile):
        """绘制 pixmap 并在其上绘制 detection 框，按 display_scale 缩放显示在 canvas_label 上"""
        if self.base_image is None:
            return

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
        center_x = (label_w - scaled_w) // 2
        center_y = (label_h - scaled_h) // 2
        x = int(center_x + self.pan_offset.x())
        y = int(center_y + self.pan_offset.y())

        # 创建一个与 label 大小一致的画布 pixmap，在其上绘制缩放后的图像（并随后绘制框）
        canvas_pix = QtGui.QPixmap(label_w, label_h)
        canvas_pix.fill(QtCore.Qt.white)  # 背景填充
        painter = QtGui.QPainter(canvas_pix)
        painter.drawPixmap(x, y, scaled)
        font = QtGui.QFont("Arial")
        # 使用像素尺寸设定，避免点(size in points)引发跨设备/缩放不一致
        font.setPixelSize(max(9, int(6 * self.display_scale)))
        painter.setFont(font)

        # 遍历所有检测框进行绘制
        for det in tile.detections:
            # 根据检测框状态和是否被选中选择颜色
            color_name = STATUS_COLORS.get(det.status)
            if self.selected_det and det.id == self.selected_det.id:
                color_name = STATUS_COLORS.get("processing")

            x0 = int((det.x - det.w / 2) * self.display_scale) + x  # 左上角x坐标
            y0 = int((det.y - det.h / 2) * self.display_scale) + y  # 左上角y坐标
            w = int(det.w * self.display_scale)
            h = int(det.h * self.display_scale)

            painter.setPen(QtGui.QPen(QtGui.QColor(color_name), 2))
            painter.drawRect(x0, y0, w, h)

            painter.setPen(QtCore.Qt.white)
            painter.drawText(x0, y0 - 6, f"conf: {det.conf:.2f}")  # Draw Conf (Top-Left)
            painter.drawText(x0, y0 + h + 9, f"ID: {det.id}")       # Draw ID (Bottom-Left)

        painter.end()
        self.canvas_label.setPixmap(canvas_pix)             # 把最终画布放到 label（pixmap 大小等于 label 大小）
        self.canvas_label.repaint()  # 强制重绘
        self._image_draw_pos = (x, y, scaled_w, scaled_h)   # 保存当前图像绘制区域位置，方便坐标映射

    def eventFilter(self, obj, event):
        # keyboard navigation and canvas events
        if obj is self.canvas_label:
            # 添加安全检查：如果没有当前tile或没有基础图像，不处理事件
            if self.base_image is None or self.current_montage is None or self.current_tile_id is None:
                return super().eventFilter(obj, event)

            if event.type() == QtCore.QEvent.KeyPress:  # 键盘按键事件
                if event.key() == QtCore.Qt.Key_Left:
                    self.go_prev()      # 左箭头 -> 上一个切片
                if event.key() == QtCore.Qt.Key_Right:
                    self.go_next()      # 右箭头 -> 下一个切片
                if event.matches(QtGui.QKeySequence.Undo):
                    self.undo_last()    # Ctrl+Z -> 撤销操作
                if event.key() == QtCore.Qt.Key_R:
                    self.restore_selected()  # R -> 按 R 或 r 恢复选中框
                if event.key() == QtCore.Qt.Key_S:
                    self.confirm_and_save()  # S -> 按 S 或 s 恢复选中框
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

                # 限制缩放范围在0.2到4.0之间
                self.display_scale = min(max(self.display_scale, 0.2), 4.0)
                # 重新渲染当前切片
                tile = self.current_montage.tiles.get(self.current_tile_id)
                if tile:
                    self._render_tile(tile)
                return True

            if event.type() == QtCore.QEvent.MouseButtonPress:          # 鼠标按下事件
                pos = event.pos()                                       # 获取鼠标位置
                tile = self.current_montage.tiles.get(self.current_tile_id)
                if not tile:
                    return True

                img_x, img_y = self._screen_to_image_coords(pos)
                if event.button() == QtCore.Qt.LeftButton:              # 处理左键点击
                    clicked = None
                    for det in tile.detections:
                        if det.x - det.w/2 <= img_x <= det.x + det.w/2 and det.y - det.h/2 <= img_y <= det.y + det.h/2:
                            clicked = det
                            break

                    if clicked:                     # 点击了检测框：选中它
                        self.selected_det = clicked
                        self._render_tile(tile)     # 重新渲染以显示选中状态
                    else:                           # 点击了空白区域
                        if self.selected_det:       # 有选中框：移动到点击位置
                            self.move_selected_detection(tile, img_x, img_y)
                        else:                       # 没有选中框：创建新框
                            self.add_new_detection(tile, img_x, img_y)
                elif event.button() == QtCore.Qt.RightButton:           # 处理右键点击 - 取消选择
                    self.selected_det = None
                    self._render_tile(tile)  # 重新渲染以更新显示
                elif event.button() == QtCore.Qt.MiddleButton:          # 处理中键点击 - 启动平移
                    self._panning = True
                    self._pan_start = pos
                return True

            if event.type() == QtCore.QEvent.MouseMove:              # 鼠标移动：如果在平移模式，更新 pan_offset 并重绘；否则忽略
                if self._panning:
                    assert self._pan_start is not None
                    delta = event.pos() - self._pan_start
                    self.pan_offset = self._pan_start_offset + delta
                    self._clamp_pan()
                    tile = self.current_montage.tiles.get(self.current_tile_id)
                    if tile:
                        self._render_tile(tile)
                    return True

            if event.type() == QtCore.QEvent.MouseButtonRelease:    # 鼠标释放：结束平移
                if self._panning:
                    self._panning = False
                    self._pan_start = None
                    self._pan_start_offset = QtCore.QPoint(0, 0)
                    return True

        return super().eventFilter(obj, event)

    def _clamp_pan(self):
        """限制 pan_offset 的范围，避免图像被拖出太远（保持至少 20px 可见）"""
        label_size = self.canvas_label.size()
        label_w, label_h = label_size.width(), label_size.height()
        x, y, scaled_w, scaled_h = self._image_draw_pos
        # self._image_draw_pos 是基于上次渲染时 center+pan_offset 的值。为了在改变 pan_offset 后获得正确约束，
        # 我们重新计算 center（未含 pan_offset）
        center_x = (label_w - scaled_w) // 2
        center_y = (label_h - scaled_h) // 2

        # 允许的 x(top-left) 范围： [20 - scaled_w, label_w - 20]
        min_x = 20 - scaled_w
        max_x = label_w - 20
        # 转换到 pan_offset 的范围： pan = x - center
        pan_min = min_x - center_x
        pan_max = max_x - center_x

        min_y = 20 - scaled_h
        max_y = label_h - 20
        pan_min_y = min_y - center_y
        pan_max_y = max_y - center_y

        # clamp
        px = max(int(pan_min), min(int(self.pan_offset.x()), int(pan_max)))
        py = max(int(pan_min_y), min(int(self.pan_offset.y()), int(pan_max_y)))
        self.pan_offset = QtCore.QPoint(px, py)

    def go_prev(self):
        if self.current_montage is None or self.current_tile_id is None:
            return
        if self.navigation_context == "manual_list":
            row = self.manual_list_widget.currentRow()
            next_row = (row - 1) % self.manual_list_widget.count()
            self.manual_list_widget.setCurrentRow(next_row)
            self.on_manual_list_clicked(self.manual_list_widget.currentItem())
        else:
            keys = sorted(self.current_montage.tiles.keys())
            curr_idx_in_keys = keys.index(self.current_tile_id)
            next_id = keys[(curr_idx_in_keys - 1) % len(keys)]
            self.set_current_tile(next_id)

    def go_next(self):
        if self.current_montage is None or self.current_tile_id is None:
            return
        if self.navigation_context == "manual_list":
            row = self.manual_list_widget.currentRow()
            next_row = (row + 1) % self.manual_list_widget.count()
            self.manual_list_widget.setCurrentRow(next_row)
            self.on_manual_list_clicked(self.manual_list_widget.currentItem())
        else:
            keys = sorted(self.current_montage.tiles.keys())
            curr_idx_in_keys = keys.index(self.current_tile_id)
            next_id = keys[(curr_idx_in_keys + 1) % len(keys)]
            self.set_current_tile(next_id)

    def on_brightness_changed(self, val: int):
        """slider 回调：val 0-400 映射到 0.0-4.0"""
        if self.current_montage is None or self.current_tile_id is None:
            return
        self._brightness = val / 100.0
        tile = self.current_montage.tiles.get(self.current_tile_id)
        self._render_tile(tile)

    def apply_conf_filter(self, current_only: bool):
        """Mark dets with conf < threshold as deleted."""
        if self.current_montage is None or self.current_tile_id is None:
            return
        thresh = self.spin_filter_conf.value()
        tiles_to_process = []
        t = self.current_montage.tiles.get(self.current_tile_id)
        if current_only:
            tiles_to_process.append(t)
        else:
            # TODO现在这样显然不对！需要好好想想如何能对所有文件操作
            tiles_to_process = list(self.current_montage.tiles.values())

        count = 0
        for tile in tiles_to_process:
            dirty = False
            for d in tile.detections:
                if d.status != "deleted" and d.conf < thresh:
                    d.status = "deleted"
                    dirty = True
                    count += 1
            if dirty:
                mont_name, idx = match_name(tile.name, _TILE_RE)
                pred_name = PRED_NAME_TEMPLATE.format(montage=mont_name, z=idx)
                pred_path = self.preds_dir / pred_name
                write_detections(pred_path, tile.detections)
                tile._dirty = True

        self._render_tile(t)
        self._update_tile_list_item_color(self.current_tile_id, t)
        self.log(f"Filtered {count} detections < {thresh:.2f}")

    def delete_selected(self):
        if self.selected_det is None or self.current_montage is None or self.current_tile_id is None:
            return
        tile = self.current_montage.tiles.get(self.current_tile_id)
        prev_status = self.selected_det.status
        action = {"type": "status_change",
                  "tile": self.current_tile_id,
                  "det_id": self.selected_det.id,
                  "prev_status": prev_status,
                  "new_status": "deleted"}
        self._undo_stack.append(action)

        for d in tile.detections:
            if d.id == self.selected_det.id:
                d.status = "deleted"
                break
        self.selected_det = None
        tile._dirty = True
        self._render_tile(tile)
        self._update_tile_list_item_color(self.current_tile_id, tile)

    def restore_selected(self):
        """把选中的 detection（如果为 deleted）恢复为 active（并加入 undo 栈）"""
        if self.current_montage is None or self.current_tile_id is None or self.selected_det is None:
            return
        tile = self.current_montage.tiles.get(self.current_tile_id)
        prev_status = self.selected_det.status
        self.selected_det.status = "active"
        action = {"type": "status_change",
                  "tile": self.current_tile_id,
                  "det_id": self.selected_det.id,
                  "prev_status": prev_status,
                  "new_status": "active"}
        self._undo_stack.append(action)
        tile._dirty = True
        self._render_tile(tile)

    def delete_all_for_tile(self):
        if self.current_montage is None or self.current_tile_id is None:
            return
        tile = self.current_montage.tiles.get(self.current_tile_id)
        prev = [(d.id, d.status) for d in tile.detections]
        action = {"type": "bulk_status_change",
                  "tile": self.current_tile_id,
                  "prev": prev,
                  "new_status": "deleted"}
        self._undo_stack.append(action)
        for d in tile.detections:
            d.status = "deleted"
        tile._dirty = True
        self._render_tile(tile)
        self._update_tile_list_item_color(self.current_tile_id, tile)

    def move_selected_detection(self, tile, img_x, img_y):
        prev_xy = (self.selected_det.x, self.selected_det.y)
        # 更新检测框位置
        self.selected_det.x = img_x
        self.selected_det.y = img_y
        tile._dirty = True

        action = {"type": "move",
                  "tile": self.current_tile_id,
                  "det_id": self.selected_det.id,
                  "prev_xy": prev_xy,
                  "new_xy": (img_x, img_y)}
        self._undo_stack.append(action)
        self._render_tile(tile)

    def add_new_detection(self, tile, img_x, img_y):
        # 计算新检测框的参数
        if tile.detections:
            last_id = tile.detections[-1].id
            last_cls = tile.detections[-1].cls
            last_w = tile.detections[-1].w
            last_h = tile.detections[-1].h
        else:
            last_id = 0
            last_cls = 0
            last_w = 150.0
            last_h = 150.0

        new_id = last_id + 1
        new_det = Detection(new_id, last_cls, float(img_x), float(img_y), float(last_w), float(last_h), 0.0, "active")
        tile.detections.append(new_det)
        tile._dirty = True

        action = {"type": "add",
                  "tile": self.current_tile_id,
                  "det_id": new_id}
        self._undo_stack.append(action)
        self.selected_det = new_det
        self._render_tile(tile)

    def undo_last(self):
        if not self._undo_stack:
            return
        action = self._undo_stack.pop()
        t_id = action.get("tile")
        tile = self.current_montage.tiles.get(t_id)
        if not tile:
            return

        if action["type"] == "status_change":
            det_id = action["det_id"]
            for d in tile.detections:
                if d.id == det_id:
                    d.status = action["prev_status"]
                    break
        elif action["type"] == "bulk_status_change":
            for det_id, prev_status in action["prev"]:
                for d in tile.detections:
                    if d.id == det_id:
                        d.status = prev_status
                        break
            self._update_tile_list_item_color(t_id, tile)
        elif action["type"] == "move":
            det_id = action["det_id"]
            prev_xy = action["prev_xy"]
            for d in tile.detections:
                if d.id == det_id:
                    d.x, d.y = prev_xy
                    break
        elif action["type"] == "add":
            det_id = action["det_id"]
            new_list = [d for d in tile.detections if d.id != det_id]
            tile.detections = new_list
            # 如果当前 selected_det 指向被删除的 detection，清除 selection
            if self.selected_det and self.selected_det.id == det_id:
                self.selected_det = None

        tile._dirty = True
        if t_id == self.current_tile_id:        # 如果我们 undo 的正是当前显示的 tile，则重渲染
            self._render_tile(tile)

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

    def _update_tile_list_item_color(self, idx: int, tile: Tile):
        """主要是更新切片列表项的背景色为红色或者撤销"""
        if tile.detections:
            all_deleted = all(det.status == "deleted" for det in tile.detections)
            if all_deleted:
                tile.status = "deleted"
            else:
                tile.status = "processed"

        color = QtGui.QColor(STATUS_COLORS.get(tile.status))
        for i in range(self.tile_list.count()):
            it = self.tile_list.item(i)
            if it.data(QtCore.Qt.UserRole) == idx:
                it.setBackground(color)
                break

    def confirm_and_save(self):
        if self.current_montage is None or self.current_tile_id is None:
            return

        curr_row = self.manual_list_widget.currentRow()
        if curr_row >= 0:
            try:
                item_text = self.manual_list_widget.item(curr_row).text()
                remove_from_manual_list(self.project_root, item_text)
                self.manual_list_widget.takeItem(curr_row)

                tile = self.current_montage.tiles.get(self.current_tile_id)
                mont_name, idx = match_name(tile.name, _TILE_RE)
                pred_name = PRED_NAME_TEMPLATE.format(montage=mont_name, z=idx)
                pred_path = self.preds_dir / pred_name
                write_detections(pred_path, tile.detections)
                tile._dirty = False  # 清除修改标记
                msg = f"Saved predictions: {pred_path}"

                # Move to next in list
                if curr_row < self.manual_list_widget.count():
                    self.manual_list_widget.setCurrentRow(curr_row)
                    self.on_manual_list_clicked(self.manual_list_widget.currentItem())
                elif self.manual_list_widget.count() > 0:
                    self.manual_list_widget.setCurrentRow(0)
                    self.on_manual_list_clicked(self.manual_list_widget.currentItem())
            except Exception as e:
                msg = f"Failed to save predictions: {e}"
                logger.error(msg)
            self.log(msg)


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Picker")
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
        self.status.montage_selected.connect(self.on_montage_selected)      # 状态面板的选择信号
        self.viewer.tile_selected.connect(self.on_tile_selected)            # viewer面板的Manual List的选择tile信号

        self.worker: Optional[MontageProcessor] = None                      # 工作进程
        self.observer: Optional[Observer] = None                            # 文件系统观察者，用于监控新文件

        self.timer = QtCore.QTimer()                                        # 创建定时器用于定期从UI队列中处理更新
        self.timer.timeout.connect(self.process_ui_queue)
        self.timer.start(200)                                               # 启动定时器，每200毫秒触发一次

    def on_start(self, cfg: dict):
        """Initialize workers and load data."""
        try:
            nav_folder = cfg["nav_path"].parent
            self.montages = load_nav_and_montages(cfg["nav_path"], cfg["project_name"], cfg["overwrite"])
            self.status.montages = self.montages
            self.status.refresh()
            self.viewer.set_dirs(nav_folder, cfg["project_name"])

            # Watcher
            if self.observer:
                self.observer.stop()
            self.observer = Observer()
            handler = MontageWatcher(nav_folder, self.ui_queue, self.job_queue, self.montages)
            self.observer.schedule(handler, str(nav_folder), recursive=False)
            self.observer.start()

            # Processing Worker
            detector = YoLoWrapper(str(cfg["model_path"]))
            self.worker = MontageProcessor(self.ui_queue, self.job_queue, detector, cfg)
            self.worker.start()         # 启动工作线程，在后台持续监听 job_queue

            # Enqueue existing tasks
            for m in self.montages.values():
                if m.status == "not generated" and m.map_file.exists():
                    m.status = "queuing"
                    self.ui_queue.put(("update_montage", m))
                    self.job_queue.put(m)

        except Exception as e:
            logger.error(f"Startup failed: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def on_stop(self):
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=2.0)
            self.observer = None

        if self.worker and self.worker.isRunning():
            self.worker.stop()
            # Wait a bit for it to finish current tile, but don't freeze UI forever
            self.worker.wait(10000)

        # NOTE: Do NOT clear queues forcefully. Let them drain naturally or just exist.
        # Clearing them risks deleting data that the worker just finished but hasn't been saved/displayed.
        logger.info("Processing stopped.")

    def on_export(self, cfg: dict):
        msg = add_predictions_to_nav(cfg["nav_path"], cfg["project_name"], cfg["save_path"])
        QtWidgets.QMessageBox.information(self, "Export", msg)
        logger.info(msg)

    def on_montage_selected(self, name: str):
        if name in self.montages:
            self.viewer.load_montage(self.montages[name])

    def on_tile_selected(self, tile_name: str):
        mont_stem, idx = match_name(tile_name, _TILE_RE)
        last_key = list(self.montages.keys())[-1]
        name = mont_stem + str(self.montages[last_key].map_file.suffix)
        if idx != -1 and name in self.montages:
            self.viewer.load_montage(self.montages[name])
            self.viewer.set_current_tile(idx)

    def process_ui_queue(self):
        """Called from main thread to process updates from worker threads"""
        while not self.ui_queue.empty():
            try:
                cmd = self.ui_queue.get_nowait()
                if cmd[0] == "update_montage":
                    mont = cmd[1]
                    self.status.update_montage(mont)
                elif cmd[0] == "refresh_tile_list":
                    name = cmd[1]
                    if self.viewer.current_montage and self.viewer.current_montage.name == name:
                        self.viewer.load_montage(self.montages[name])
                elif cmd[0] == "add_manual_item":
                    tile_name = cmd[1]
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
