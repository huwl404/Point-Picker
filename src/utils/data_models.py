#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : data_models.py
# Time       : 2025/11/22 16:14
# Author     : 14750
# Email      : huwl@hku.hk
# Description：

"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

@dataclass
class Detection:
    id: int
    cls: int
    x: float  # x_center (pixel)
    y: float  # y_center (pixel)
    w: float  # width (pixel)
    h: float  # height (pixel)
    conf: float
    status: str = "active"  # active, processing, deleted


@dataclass
class Tile:
    name: str  # tile_file.name
    tile_id: int
    tile_file: Path
    status: str = "processed"  # processed, deleted
    detections: List[Detection] = field(default_factory=list)


@dataclass
class Montage:
    name: str  # map_file.name
    map_id: int
    map_file: Path
    map_frames: List[int]
    status: str = "not generated"  # not generated, not processed, processing, processed, error
    tiles: Dict[int, Tile] = field(default_factory=dict)  # slice in this montage = tile_id -> tile


STATUS_COLORS = {
    "not generated": "white",
    "queuing": "cyan",
    "processing": "yellow",
    "processed": "lightgreen",
    "error": "lightcoral",

    "active": "blue",
    "deleted": "red"
}
