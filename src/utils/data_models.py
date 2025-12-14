#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : data_models.py
# Time       : 2025/11/22 16:14
# Author     : 14750
# Email      : huwl@hku.hk
# Description：

"""
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

@dataclass
class Detection:
    cls: int
    x: float  # x_center (pixel)
    y: float  # y_center (pixel)
    w: float  # width (pixel)
    h: float  # height (pixel)
    conf: float  # 2.0 -> externally added
    status: str = "active"  # active, filtered

    def __eq__(self, other):
        """重写相等性判断。当且仅当 self 和 other 对象的 x 和 y 坐标相等时，返回 True。"""
        if not isinstance(other, Detection):
            return NotImplemented

        epsilon = 1e-6  # 定义一个很小的容忍度
        # 检查 x 和 y 是否在容忍度范围内相等
        x_equal = math.isclose(self.x, other.x, abs_tol=epsilon)
        y_equal = math.isclose(self.y, other.y, abs_tol=epsilon)
        return x_equal and y_equal


@dataclass
class Tile:
    name: str  # tile_file.name
    tile_sec: int
    tile_file: Path
    status: str = "processed"  # processed, deleted, processing
    detections: List[Detection] = field(default_factory=list)


@dataclass
class Montage:
    name: str  # map_file.name
    map_id: int
    map_file: Path
    map_frames: List[int]
    status: str = "to be validated"  # to be validated, to be shot, queuing, processing, processed, error, excluded
    tiles: Dict[str, Tile] = field(default_factory=dict)  # slice in this montage = tile_name -> tile


STATUS_COLORS = {
    "to be validated": "cyan",
    "to be shot": "#F4F0F1",  # slight grey
    "queuing": "#ADE699",  # slight green
    "processing": "yellow",
    "processed": "#10C378",  # green
    "error": "#FF773C",  # orange
    "excluded": "grey",

    "active": "#7DF9FF",  # electric blue
    "deleted": "red"
}
