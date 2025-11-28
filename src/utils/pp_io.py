#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : pp_io.py
# Time       : 2025/11/22 16:09
# Author     : 14750
# Email      : huwl@hku.hk
# Description：
Handles file input/output, regex parsing, and global constants.
"""
import os
import re
import shutil
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Pattern

from src.utils.data_models import Detection, Tile, Montage
from src.utils.nav_io import NavItem, read_nav_file, write_nav_file


# -----------------------------------------------------------------------------
# Global Constants & Templates
# -----------------------------------------------------------------------------
TILE_NAME_TEMPLATE = "{montage}_tile{z:03d}.png"      # Template for tile image filenames
PRED_NAME_TEMPLATE = "{montage}_tile{z:03d}.txt"      # Template for prediction filenames
MANUAL_LIST_FILENAME = "manual_confirmation.txt"      # File to store tiles needing manual review

# Regex for parsing filenames
_TILE_RE = re.compile(r"(?P<mont>.+)_tile(?P<idx>\d+)\.png$")
_PRED_RE = re.compile(r"(?P<mont>.+)_tile(?P<idx>\d+)\.txt$")


def ensure_project_dirs(nav_folder: Path, project_name: str) -> Tuple[Path, Path]:
    """Create and return images and predictions directories."""
    project_root = nav_folder / project_name
    images_dir = project_root / "images"
    preds_dir = project_root / "predictions"
    images_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, preds_dir

def match_name(name: str, patt: Pattern[str]) -> Tuple[str, int]:
    m = patt.match(name)
    if m:
        return m.group('mont'), int(m.group('idx'))
    return "Not Found, indicated by -1", -1

def load_existing_info(nav_folder: Path, project_name: str) -> Dict[str, Dict[str, Tile]]:
    """Scan project folder to load existing processed tiles and predictions."""
    images_dir, preds_dir = ensure_project_dirs(nav_folder, project_name)
    mont_tiles: Dict[str, Dict[str, Tile]] = {}

    for p in sorted(images_dir.glob("*.png")):  # 有序字典
        mont_stem, idx = match_name(p.name, _TILE_RE)
        if idx == -1:
            continue

        # only include tile if prediction file exists and is non-empty
        pred_name = PRED_NAME_TEMPLATE.format(montage=mont_stem, z=idx)
        pred_file = preds_dir / pred_name
        if p.stat().st_size == 0 or not pred_file.exists():
            continue

        tile = Tile(name=p.name, tile_sec=idx, tile_file=p)
        # setdefault确保蒙太奇名称对应的字典存在，然后设置切片索引到tile的映射
        mont_tiles.setdefault(mont_stem, {})[p.name] = tile

    return mont_tiles

def load_nav_and_montages(nav_path: Path, project_name: str, overwrite: bool) -> Dict[str, Montage]:
    """Parse .nav file to build Montage objects, optionally loading existing data."""
    nav_folder = nav_path.parent
    ensure_project_dirs(nav_folder, project_name)
    montages: Dict[str, Montage] = {}

    items = read_nav_file(str(nav_path))
    for it in items:
        type_tag = getattr(it, "Type", None)
        if type_tag == 2:  # Map
            mapid = getattr(it, "MapID", None)
            mapfile = getattr(it, "MapFile", None)
            mapframes = getattr(it, "MapFramesXY", None)
            mont = Montage(name=Path(mapfile).name, map_id=mapid, map_file=Path(mapfile), map_frames=mapframes)
            montages[mont.name] = mont
        # ignore 0 -> point & 1 -> polygon

    # Load existing state if not overwriting
    if not overwrite:
        existing = load_existing_info(nav_folder, project_name)  # key is montage stem
        for montage_name in montages.keys():
            base_name = Path(montage_name).stem             # 获取不带后缀的文件名
            if base_name in existing:
                montages[montage_name].tiles.update(existing[base_name])
                montages[montage_name].status = "processed"
    else:
        p = nav_folder / project_name / MANUAL_LIST_FILENAME
        if os.path.isfile(p):
            os.remove(p)

    return montages

def read_detections(path: Path) -> List[Detection]:
    """Parse a detection text file into a list of Detection objects."""
    dets: List[Detection] = []
    if not path.exists():
        return dets

    with path.open('r', encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.split()

            # 期望的格式：cls x y w h conf status
            if len(parts) >= 7:
                try:
                    cls = int(parts[0])     # 类别ID，转换为整数，默认为0
                    x = float(parts[1])     # 中心点x坐标，转换为浮点数
                    y = float(parts[2])     # 中心点y坐标，转换为浮点数
                    w = float(parts[3])     # 宽度，转换为浮点数
                    h = float(parts[4])     # 高度，转换为浮点数
                    conf = float(parts[5])  # 置信度，转换为浮点数
                    status = parts[6]
                    dets.append(Detection(cls, x, y, w, h, conf, status))
                except ValueError:
                    continue

    return dets

def write_detections(path: Path, dets: List[Detection]):
    """Write a list of Detection objects to a text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        fh.write("# cls center_x center_y w h conf status, conf=2 means externally added\n")
        for d in dets:
            fh.write(f"{d.cls} {d.x:.2f} {d.y:.2f} {d.w:.2f} {d.h:.2f} {d.conf:.2f} {d.status}\n")

def add_predictions_to_nav(input_nav: Path, project_name: str, output_nav: Path) -> str:
    """Export valid predictions back into a new .nav file."""
    # 1) read nav file
    items = read_nav_file(str(input_nav))

    # 2) build map lookup by MapFile stem
    map_lookup = build_map_lookup(items)
    if not map_lookup:
        return f"No Map items found in {input_nav}"

    images_dir, preds_dir = ensure_project_dirs(input_nav.parent, project_name)
    # 3) iterate prediction files
    new_items = []
    group_counter = 0       # GroupID starts from 1 (per-file group)
    # set class iterator to continue numbering
    tag_id= 1
    for p in sorted(preds_dir.glob("*.txt")):
        mont, idx = match_name(p.name, _PRED_RE)
        if idx == -1:
            continue

        map_item = map_lookup.get(mont)
        if not map_item:
            continue

        try:
            stage_z = map_item.StageXYZ[2]
        except Exception:
            continue

        coords_iter = list(parse_prediction_file(p))
        if not coords_iter:
            continue

        # All NavItems from this file get the same GroupID
        group_id = group_counter
        group_counter += 1
        first = True
        for (x, y) in coords_iter:
            d = {
                'Color': 0, 'NumPts': 0, 'Regis': 1, 'Type': 0,
                'GroupID': group_id, 'DrawnID': int(map_item.MapID),
                'CoordsInPiece': [float(x), float(y), float(stage_z)],
                'PieceOn': int(idx),
            }
            # first nav item gets Acquire=1
            if first:
                d['Acquire'] = 1
                first = False

            # create NavItem (tag will be autogenerated as Item-<number>)
            nav_item = NavItem(d, f'{project_name}-{tag_id}')
            new_items.append(nav_item)
            tag_id += 1

    if tag_id == 1:
        return f'No active predictions found: {preds_dir}.'

    # 5) copy input nav to output nav (overwrite if exists)
    shutil.copy2(str(input_nav), str(output_nav))

    # Append new items to output nav
    write_nav_file(str(output_nav), *new_items, mode='a')

    return f"Saved nav: {output_nav}. montage items: {len(map_lookup)},  total points: {tag_id - 1}."

def build_map_lookup(items) -> Dict[str, NavItem]:
    """Return dict mapping MapFile stem -> MapItem instance"""
    lookup = {}
    for it in items:
        if getattr(it, 'kind', None) == 'Map':
            stem = Path(it.MapFile).stem
            lookup[stem] = it
    return lookup

def parse_prediction_file(fn: Path):
    """Yield (x, y) for active rows in prediction file."""
    with fn.open('r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 7 and parts[6].lower() == 'active':
                try:
                    yield float(parts[1]), float(parts[2])
                except ValueError:
                    pass

def append_to_manual_list(project_root: Path, entry: str) -> Optional[str]:
    """Append tile_name to the manual confirmation file."""
    p = project_root / MANUAL_LIST_FILENAME
    try:
        with p.open("a", encoding="utf-8") as f:
            f.write(entry + "\n")
        return None
    except Exception as e:
        return f"Error appending to manual list: {e}"

def read_manual_list(project_root: Path) -> List[str]:
    """Read all entries from the manual confirmation file."""
    p = project_root / MANUAL_LIST_FILENAME
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def remove_from_manual_list(project_root: Path, entry_to_remove: str):
    """Remove a specific entry from the manual confirmation file."""
    p = project_root / MANUAL_LIST_FILENAME
    if not p.exists():
        return

    lines = read_manual_list(project_root)
    lines = [L for L in lines if L != entry_to_remove]

    with p.open("w", encoding="utf-8") as f:
        for L in lines:
            f.write(L + "\n")

def read_mdoc_spacing(mdoc_path: Path) -> Tuple[int, int]:
    """Read PieceSpacing from .mdoc file. Returns (spacing_x, spacing_y)."""
    if not mdoc_path.exists():
        return 0, 0

    try:
        with mdoc_path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip().startswith("PieceSpacing ="):
                    # Format: PieceSpacing = 3686 3686
                    parts = line.split('=')[1].strip().split()
                    if len(parts) >= 2:
                        return int(parts[0]), int(parts[1])
    except Exception:
        pass

    return 0, 0