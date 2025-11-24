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
 - SettingsPanel: set project paths and parameters
 - StatusPanel: lists montages and per-montage statuses, starts processing
 - ViewerPanel: display selected montage/tile with predicted boxes; edit boxes
 - Background workers: splitting + prediction pipeline; filesystem watcher
 - Export interface: export_annotations(output_path)
"""
from __future__ import annotations
import sys
import os
import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import random
from venv import logger

from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
from PIL import Image, ImageQt
import mrcfile
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ultralytics import YOLO


# -----------------------------
# Simple data models
# -----------------------------
@dataclass
class Detection:
    id: int
    cls: int
    x: float  # x_center (pixel)
    y: float  # y_center (pixel)
    w: float  # width (pixel)
    h: float  # height (pixel)
    conf: float
    state: str = "active"  # active, deleted
    color_state: str = "green"  # green default, yellow selected, red deleted


@dataclass
class Tile:
    tile_index: int
    tile_path: Path
    image: Optional[np.ndarray] = None  # HxW numpy array
    detections: List[Detection] = field(default_factory=list)


@dataclass
class Montage:
    name: str
    mrc_path: Path
    status: str = "not_generated"  # not_generated, not_processed, processing, processed, error
    tiles: Dict[int, Tile] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


# -----------------------------
# Utility functions
# -----------------------------
def read_mrc_first_slice(p: Path) -> np.ndarray:
    """Read an MRC; if stacked for tiles, we will not call this; but useful."""
    with mrcfile.open(p, permissive=True) as m:
        arr = np.asarray(m.data)
        if arr.ndim == 3:
            # Many montages are (Z, Y, X) where Z might be tiles; here we return first slice
            arr = arr[0]
        return arr.astype(np.float32)


def read_mrc_slices(p: Path) -> np.ndarray:
    """Return full array as (Z, Y, X)"""
    with mrcfile.open(p, permissive=True) as m:
        arr = np.asarray(m.data)
        return arr


# -----------------------------
# YOLO predictor
# -----------------------------
class YoloPredictor:
    """Wrapper with .predict(tile_path) -> List[Detection]"""

    def __init__(self, model_path: Optional[str] = None, conf_thresh: float = 0.3):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self._counter = 0
        if model_path:
            try:
                self.model = YOLO(model_path)
            except Exception:
                self.model = None
        else:
            self.model = None

    def predict(self, tile_array: np.ndarray) -> List[Detection]:
        """
        Accepts tile_image array (H,W) or (H,W,3).
        If ultralytics is available and model loaded, runs real prediction.
        Otherwise returns deterministic dummy boxes for UI testing.
        """
        if self.model is not None:
            # convert numpy to suitable image format (RGB)
            try:
                results = self.model.predict(source=tile_array, conf=self.conf_thresh, verbose=False)
                dets = []
                for r in results:
                    if hasattr(r, 'boxes'):
                        boxes = r.boxes
                        for b in boxes:
                            xyxy = b.xyxy[0].cpu().numpy()
                            conf = float(b.conf.cpu().numpy()[0])
                            cls = int(b.cls.cpu().numpy()[0])
                            x0, y0, x1, y1 = xyxy
                            xc = float((x0 + x1) / 2.0)
                            yc = float((y0 + y1) / 2.0)
                            w = float(x1 - x0)
                            h = float(y1 - y0)
                            self._counter += 1
                            dets.append(Detection(self._counter, cls, xc, yc, w, h, conf))
                return dets
            except Exception as e:
                print("[WARN] real model prediction failed:", e)
                # fall through to dummy
        # dummy deterministic boxes
        H, W = tile_array.shape[:2]
        rng = random.Random((int(H) * 31 + int(W)))
        dets = []
        for i in range(rng.randint(0, 4)):
            self._counter += 1
            w = W * rng.uniform(0.05, 0.25)
            h = H * rng.uniform(0.05, 0.25)
            xc = rng.uniform(w / 2, W - w / 2)
            yc = rng.uniform(h / 2, H - h / 2)
            conf = rng.uniform(0.4, 0.99)
            dets.append(Detection(self._counter, 0, xc, yc, w, h, conf))
        return dets


# -----------------------------
# Pipeline worker classes
# -----------------------------
class SplitAndPredictWorker(threading.Thread):
    """
    Worker thread that receives montage jobs: split montage into tiles (assumes
    mrc is stacked with Z = n_tiles), load each tile, run predictor, and save results
    into Montage object.
    """

    def __init__(self, job_queue: Queue, predictor: YoloPredictor, ui_queue: Queue):
        super().__init__(daemon=True)
        self.job_queue = job_queue
        self.predictor = predictor
        self.ui_queue = ui_queue
        self._stopping = False

    def stop(self):
        self._stopping = True

    def run(self):
        while not self._stopping:
            try:
                montage: Montage = self.job_queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                montage.status = "processing"
                self.ui_queue.put(("update_montage", montage.name))
                arr = read_mrc_slices(montage.mrc_path)  # (Z, Y, X)
                Z, Y, X = arr.shape
                # store tiles
                for z in range(Z):
                    tile = Tile(tile_index=z,
                                tile_path=montage.mrc_path.with_name(f"{montage.mrc_path.stem}_tile{z:03d}.mrc"))
                    tile.image = arr[z].astype(np.float32)
                    # run predictor
                    dets = self.predictor.predict(tile.image)
                    tile.detections = dets
                    montage.tiles[z] = tile
                    # notify UI that this tile is available
                    self.ui_queue.put(("tile_ready", montage.name, z))
                montage.status = "processed"
                self.ui_queue.put(("update_montage", montage.name))
            except Exception as e:
                montage.status = "error"
                montage.metadata['error'] = str(e)
                self.ui_queue.put(("update_montage", montage.name))
            finally:
                self.job_queue.task_done()


# -----------------------------
# Filesystem watcher
# -----------------------------
class MontageWatcher(FileSystemEventHandler):
    """Watch a directory and signal when new .mrc files appear."""

    def __init__(self, monitored_paths: List[Path], ui_queue: Queue, nav_montages: Dict[str, Montage]):
        super().__init__()
        self.monitored_paths = monitored_paths
        self.ui_queue = ui_queue
        self.nav_montages = nav_montages

    def on_created(self, event):
        p = Path(event.src_path)
        if p.suffix.lower() in ('.mrc',):
            # check if this file corresponds to any expected montage by name
            for name, montage in self.nav_montages.items():
                if p.name == montage.mrc_path.name:
                    montage.status = "not_processed"
                    montage.mrc_path = p
                    self.ui_queue.put(("update_montage", name))
                    break

    def on_moved(self, event):
        self.on_created(event)

    def on_modified(self, event):
        # some writers create then modify; treat as create
        self.on_created(event)


# -----------------------------
# GUI Components
# -----------------------------
class SettingsPanel(QtWidgets.QWidget):
    # 定义自定义信号 - Qt框架中组件间通信的核心机制
    settings_changed = QtCore.pyqtSignal(dict)  # 当设置改变时发射，携带配置字典
    start_requested = QtCore.pyqtSignal(dict)   # 当用户点击开始按钮时发射，携带验证后的配置

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gpus = self._detect_gpu()          # 仅需初始运行一次
        self._build_ui()
        self._connect_change_signals()          # 连接变化信号以便发射 settings_changed

    @staticmethod
    def _detect_gpu() -> List[Dict[str, Any]]:
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
        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
            pass

        return gpus

    @staticmethod
    def get_resource_path(relative_path: str) -> Path:
        """兼容 PyInstaller 打包与源码运行"""
        if hasattr(sys, "_MEIPASS"):  # 运行在 PyInstaller 打包环境
            base_path = Path(sys._MEIPASS)
        else:
            base_path = Path(__file__).resolve().parent.parent  # src 的上级目录

        return (base_path / relative_path).resolve()

    def _build_ui(self):
        # 使用表单布局管理器 - 适合标签-字段对的排列
        layout = QtWidgets.QFormLayout()
        self.setLayout(layout)

        # 创建各种输入字段组件
        self.project_name = QtWidgets.QLineEdit()       # 项目名称文本输入框
        self.model_path = QtWidgets.QLineEdit()         # 模型文件路径文本输入框
        self.nav_path = QtWidgets.QLineEdit()           # 导航文件路径文本输入框

        default_model = self.get_resource_path("data/md2_pm2_best.pt")
        self.model_path.setText(str(default_model))

        # 最大检测数量选择器 - 限制每张图像的最大检测目标数
        self.max_detection = QtWidgets.QSpinBox()
        self.max_detection.setRange(1, 1000)
        self.max_detection.setValue(50)

        self.overwrite = QtWidgets.QCheckBox("Overwrite tiles/labels on re-run")
        start_btn = QtWidgets.QPushButton("RUN")        # 开始按钮 - 触发处理流程

        layout.addRow("Project name:", self.project_name)

        # Model line with Browse
        model_h = QtWidgets.QHBoxLayout()
        model_h.setContentsMargins(0, 0, 0, 0)          # 设置布局的边距为0（上、右、下、左），消除布局与周围元素的间距
        model_h.setSpacing(4)                           # 设置布局内组件之间的间距为4像素
        self.model_path_btn = QtWidgets.QPushButton("Browse")
        model_h.addWidget(self.model_path)
        model_h.addWidget(self.model_path_btn)
        layout.addRow("Model path:", model_h)

        # Nav line with Browse
        nav_h = QtWidgets.QHBoxLayout()
        nav_h.setContentsMargins(0, 0, 0, 0)
        nav_h.setSpacing(4)
        self.nav_path_btn = QtWidgets.QPushButton("Browse")
        nav_h.addWidget(self.nav_path)
        nav_h.addWidget(self.nav_path_btn)
        layout.addRow("Nav file:", nav_h)

        # Device row: label + checkbox（右侧）
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItem("CPU")
        if self.gpus:
            for g in self.gpus:
                self.device_combo.addItem(f"cuda:{g['id']} ({g['name']}, {g['mem_total']})")
        layout.addRow("Device:", self.device_combo)

        layout.addRow("Max detections:", self.max_detection)

        run_h = QtWidgets.QHBoxLayout()
        run_h.setContentsMargins(0, 0, 0, 0)
        run_h.setSpacing(4)
        run_h.addWidget(self.overwrite)
        run_h.addWidget(start_btn)
        layout.addRow(run_h)

        # 连接按钮事件
        self.model_path_btn.clicked.connect(self.on_browse_model)
        self.nav_path_btn.clicked.connect(self.on_browse_nav)
        start_btn.clicked.connect(self.on_start)

    # ---------------- 事件 / 信号连接 ----------------
    def _connect_change_signals(self):
        # 当关键字段改变时发射 settings_changed，便于外部 UI 实时更新或预览
        self.project_name.textChanged.connect(self._emit_settings_changed)
        self.model_path.textChanged.connect(self._emit_settings_changed)
        self.nav_path.textChanged.connect(self._emit_settings_changed)
        self.max_detection.valueChanged.connect(self._emit_settings_changed)
        self.overwrite.stateChanged.connect(self._emit_settings_changed)

    def _emit_settings_changed(self):
        cfg = {
            "project_name": self.project_name.text().strip(),
            "model_path": self.model_path.text().strip() or None,
            "nav_path": Path(self.nav_path.text().strip()) if self.nav_path.text().strip() else None,
            "max_detection": int(self.max_detection.value()),
            "use_gpu": bool(self.use_gpu_cb.isChecked()),
            "overwrite": bool(self.overwrite.isChecked()),
        }
        # 发射当前配置
        self.settings_changed.emit(cfg)

    def on_browse_model(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select model file", filter="*.pt *.yaml *.pth")
        if f:
            self.model_path.setText(f)

    def on_browse_nav(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select nav file", filter="*.nav")
        if f:
            self.nav_path.setText(f)

    def on_start(self):
        cfg = {
            "project_name": self.project_name.text().strip(),
            "model_path": self.model_path.text().strip() if self.model_path.text().strip() else None,
            "nav_path": Path(self.nav_path.text().strip()) if self.nav_path.text().strip() else None,
            "max_detection": int(self.max_detection.value()),
            "overwrite": bool(self.overwrite.isChecked()),
        }

        device_text = self.device_combo.currentText()
        device = "cpu" if device_text == "CPU" else device_text.split()[0]  # 取"cuda:0"
        cfg["device"] = device

        if not cfg["nav_path"]:
            QtWidgets.QMessageBox.warning(self, "Missing nav path", "Please specify nav path.")
            return
        # 所有验证通过，发射开始请求信号，携带配置字典
        self.start_requested.emit(cfg)


class StatusPanel(QtWidgets.QWidget):
    montage_selected = QtCore.pyqtSignal(str)  # montage name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.montages: Dict[str, Montage] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        layout.addWidget(self.list_widget)
        btn_row = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton("Export")
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.export_btn)
        layout.addLayout(btn_row)
        self.refresh_btn.clicked.connect(self.refresh_view)

    def refresh_view(self):
        self.list_widget.clear()
        for name, m in sorted(self.montages.items()):
            item = QtWidgets.QListWidgetItem(f"{name} | {m.status}")
            # color by status
            if m.status in ("processing",):
                item.setBackground(QtGui.QColor("yellow"))
            elif m.status in ("not_generated", "not_processed"):
                item.setBackground(QtGui.QColor("white"))
            elif m.status == "processed":
                item.setBackground(QtGui.QColor("lightgreen"))
            elif m.status == "error":
                item.setBackground(QtGui.QColor("lightcoral"))
            self.list_widget.addItem(item)

    def update_montage(self, name: str, montage: Montage):
        self.montages[name] = montage
        self.refresh_view()

    def on_item_clicked(self, item: QtWidgets.QListWidgetItem):
        text = item.text()
        name = text.split("|", 1)[0].strip()
        self.montage_selected.emit(name)


class ViewerPanel(QtWidgets.QWidget):
    """
    Large right-hand viewer showing selected montage/tile image and overlayed detections.
    Allows selection, marking (delete/restore), and edit of detection boxes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_montage: Optional[Montage] = None
        self.current_tile_index: Optional[int] = None
        self.selected_det: Optional[Detection] = None
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        # top controls
        controls = QtWidgets.QHBoxLayout()
        self.tile_list = QtWidgets.QListWidget()
        self.tile_list.setMaximumWidth(200)
        self.tile_list.itemClicked.connect(self.on_tile_selected)
        controls.addWidget(self.tile_list)
        # main canvas
        self.canvas_label = QtWidgets.QLabel()
        self.canvas_label.setMinimumSize(600, 400)
        self.canvas_label.setAlignment(QtCore.Qt.AlignCenter)
        controls.addWidget(self.canvas_label, 1)
        layout.addLayout(controls)
        # bottom controls
        bottom = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("◀")
        self.next_btn = QtWidgets.QPushButton("▶")
        self.delete_btn = QtWidgets.QPushButton("Delete")
        self.restore_btn = QtWidgets.QPushButton("Restore")
        bottom.addWidget(self.prev_btn)
        bottom.addWidget(self.next_btn)
        bottom.addWidget(self.delete_btn)
        bottom.addWidget(self.restore_btn)
        layout.addLayout(bottom)
        # signals
        self.prev_btn.clicked.connect(self.go_prev)
        self.next_btn.clicked.connect(self.go_next)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.restore_btn.clicked.connect(self.restore_selected)

        # keyboard
        self.canvas_label.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas_label.installEventFilter(self)

    # ---------- public API ----------
    def load_montage(self, montage: Montage):
        self.current_montage = montage
        self.tile_list.clear()
        for idx in sorted(montage.tiles.keys()):
            it = QtWidgets.QListWidgetItem(f"tile {idx}")
            self.tile_list.addItem(it)
        # auto-select first tile if exists
        if montage.tiles:
            self.set_current_tile(next(iter(montage.tiles.keys())))

    def set_current_tile(self, idx: int):
        self.current_tile_index = idx
        tile = self.current_montage.tiles.get(idx)
        if tile is None:
            self.canvas_label.clear()
            return
        self._render_tile(tile)

    def _render_tile(self, tile: Tile):
        # convert numpy image to QPixmap
        arr = tile.image
        if arr is None:
            self.canvas_label.setText("No image")
            return
        # normalize for display
        a = arr.astype(np.float32)
        a = a - a.min()
        if a.max() > 0:
            a = a / a.max() * 255.0
        im = Image.fromarray(a.astype(np.uint8))
        qim = ImageQt.ImageQt(im.convert("L"))
        pix = QtGui.QPixmap.fromImage(qim)
        # draw overlay
        painter = QtGui.QPainter(pix)
        pen = QtGui.QPen(QtGui.QColor("yellow"))
        pen.setWidth(2)
        painter.setPen(pen)
        # draw detections
        for det in tile.detections:
            color = QtGui.QColor("green")
            if det.state == "deleted":
                color = QtGui.QColor("red")
            if self.selected_det and det.id == self.selected_det.id:
                color = QtGui.QColor("yellow")
            pen = QtGui.QPen(color)
            pen.setWidth(2)
            painter.setPen(pen)
            x0 = det.x - det.w / 2
            y0 = det.y - det.h / 2
            painter.drawRect(int(x0), int(y0), int(det.w), int(det.h))
        painter.end()
        self.canvas_label.setPixmap(pix.scaled(self.canvas_label.size(), QtCore.Qt.KeepAspectRatio))

    # ---------- slots ----------
    def on_tile_selected(self, item: QtWidgets.QListWidgetItem):
        txt = item.text()
        idx = int(txt.split()[1])
        self.set_current_tile(idx)

    def go_prev(self):
        if self.current_montage is None: return
        keys = sorted(self.current_montage.tiles.keys())
        if not keys: return
        if self.current_tile_index is None:
            self.set_current_tile(keys[0]);
            return
        i = keys.index(self.current_tile_index)
        if i > 0:
            self.set_current_tile(keys[i - 1])

    def go_next(self):
        if self.current_montage is None: return
        keys = sorted(self.current_montage.tiles.keys())
        if not keys: return
        if self.current_tile_index is None:
            self.set_current_tile(keys[0]);
            return
        i = keys.index(self.current_tile_index)
        if i < len(keys) - 1:
            self.set_current_tile(keys[i + 1])

    def delete_selected(self):
        tile = self.current_montage.tiles.get(self.current_tile_index)
        if not tile: return
        # naive: mark first detection as deleted (you can build UI to choose)
        if tile.detections:
            det = tile.detections[0]
            det.state = "deleted"
            det.color_state = "red"
            self.selected_det = det
            self._render_tile(tile)

    def restore_selected(self):
        if self.current_montage is None: return
        tile = self.current_montage.tiles.get(self.current_tile_index)
        if not tile: return
        if tile.detections:
            det = tile.detections[0]
            det.state = "active"
            det.color_state = "green"
            self.selected_det = det
            self._render_tile(tile)

    def eventFilter(self, obj, event):
        # space to next
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Space:
                self.go_next()
                return True
            if event.key() == QtCore.Qt.Key_Left:
                self.go_prev()
                return True
            if event.key() == QtCore.Qt.Key_Right:
                self.go_next()
                return True
        return super().eventFilter(obj, event)


# -----------------------------
# Main application
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Picker")
        self.resize(1200, 800)

        # 创建中央窗口部件和主布局 - Qt应用程序的标准模式
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout()             # 创建水平布局管理器
        central.setLayout(h)

        # 初始化三个主要的面板组件
        self.settings = SettingsPanel()
        self.status = StatusPanel()
        self.viewer = ViewerPanel()

        left_col = QtWidgets.QVBoxLayout()      # 创建垂直布局管理器
        left_col.addWidget(self.settings)
        left_col.addWidget(self.status)

        h.addLayout(left_col, 0)        # 添加左侧列布局，拉伸因子为0（固定宽度）
        h.addWidget(self.viewer, 1)     # 添加查看器面板，拉伸因子为1（占据剩余空间）

        # 初始化内部处理管道的数据队列 - 生产者-消费者模式的关键组件
        self.job_queue: Queue = Queue()         # 作业队列：主线程向工作线程提交处理任务
        self.ui_queue: Queue = Queue()          # UI更新队列：工作线程向主线程发送UI更新请求
        self.predictor = YoloPredictor()
        self.worker = SplitAndPredictWorker(self.job_queue, self.predictor, self.ui_queue)
        self.worker.start()                     # 启动工作线程

        self.montages: Dict[str, Montage] = {}  # 键：图像名称，值：Montage对象

        # 建立信号-槽连接 - Qt框架的事件处理机制
        self.settings.start_requested.connect(self.on_start)                # 设置面板的开始信号连接到处理函数
        self.status.montage_selected.connect(self.on_montage_selected)      # 状态面板的选择信号
        self.status.export_btn.clicked.connect(self.on_export_requested)    # 状态面板的导出信号

        self.observer: Optional[Observer] = None                            # 文件系统观察者，用于监控新文件
        # 创建定时器用于定期从UI队列中处理更新 - 避免线程直接操作UI
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_ui_queue)
        self.timer.start(200)                                               # 启动定时器，每200毫秒触发一次

    def on_start(self, cfg: dict):
        # read nav and create montage entries
        nav_dir: Path = cfg["nav_path"]
        # find nav file(s) inside the path; for simplicity treat every .mrc present as expected montage
        expected = list(nav_dir.glob("*.mrc"))
        # create Montage entries keyed by stem
        for p in expected:
            name = p.stem
            mont = Montage(name=name, mrc_path=p, status="not_generated")
            self.montages[name] = mont
        # update status panel
        self.status.montages = self.montages
        self.status.refresh_view()
        # start watcher to monitor the folder for new mrc files (if nav referenced remote files, you may need other logic)
        if self.observer:
            self.observer.stop()
        event_handler = MontageWatcher([nav_dir], self.ui_queue, self.montages)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(nav_dir), recursive=False)
        self.observer.start()

        # Kick off processing for any files already present
        for name, mont in self.montages.items():
            if mont.mrc_path.exists():
                # if file exists now, mark and queue job
                mont.status = "not_processed"
                self.status.update_montage(name, mont)
                # queue processing
                self.job_queue.put(mont)

    def process_ui_queue(self):
        """Called from main thread to process updates from worker threads"""
        while True:
            try:
                item = self.ui_queue.get_nowait()
            except Empty:
                break
            if not item:
                continue
            cmd = item[0]
            if cmd == "update_montage":
                name = item[1]
                self.status.update_montage(name, self.montages[name])
            elif cmd == "tile_ready":
                name, tile_idx = item[1], item[2]
                # if montage currently selected, trigger viewer refresh
                if self.viewer.current_montage and self.viewer.current_montage.name == name:
                    self.viewer.load_montage(self.montages[name])
                    self.viewer.set_current_tile(tile_idx)

    def on_montage_selected(self, name: str):
        mont = self.montages.get(name)
        if not mont:
            return
        self.viewer.load_montage(mont)

    def on_export_requested(self):
        # user presses Export; call export interface
        try:
            outp, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export annotations to nav", filter="*.nav")
            if not outp:
                return
            self.export_annotations(Path(outp))
            QtWidgets.QMessageBox.information(self, "Export", f"Exported to {outp}")
        except NotImplementedError:
            QtWidgets.QMessageBox.warning(self, "Export",
                                          "Export function is not implemented. Implement export_annotations().")

    def export_annotations(self, out_path: Path):
        """
        EXPORT INTERFACE - implement saving annotations to a nav file (or other format).
        Expected behavior to implement:
          - iterate self.montages -> montage.tiles -> tile.detections
          - write nav format with DrawnID, PieceOn, XYinPc (or write a custom CSV)
        For now we raise to indicate this is a user-implementation point.
        """
        raise NotImplementedError(
            "Please implement export_annotations(out_path) to save annotations to desired format.")

    def closeEvent(self, event):
        # cleanup threads / observer
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=1.0)
        if self.worker:
            self.worker.stop()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
