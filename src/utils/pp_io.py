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
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Pattern, Any, Set

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


def normalize_path(path_str: str) -> Path:
    """将路径标准化为当前系统的格式，因Linux下的Path不支持解析Windows路径"""
    # 将Windows路径分隔符转换为当前系统的分隔符
    normalized = path_str.replace('\\', os.sep).replace('/', os.sep)
    return Path(normalized)

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

def preview_nav_montages(nav_path: Path) -> Dict[str, Montage]:
    """预览nav文件"""
    montages: Dict[str, Montage] = {}
    items = read_nav_file(str(nav_path))
    for it in items:
        type_tag = getattr(it, "Type", None)
        if type_tag == 2:  # Map
            mapid = getattr(it, "MapID", None)
            mapfile = getattr(it, "MapFile", None)
            mapframes = getattr(it, "MapFramesXY", None)
            status = "to be validated"
        # 0 -> point & 1 -> polygon
        # there are scripts to define maps from points or polygon, to handle it:
        elif type_tag == 0 or type_tag == 1:
            mapfile = getattr(it, "FileToOpen", None)
            if mapfile is None:  # items without this tag would be omitted
                continue
            mapid = getattr(it, "MapID", None)
            mapframes = [0, 0]  # default, when files are generated, information would be updated
            status = "to be shot"
        else:
            continue
        if normalize_path(mapfile).name in montages.keys() and mapframes == [0, 0]:  # dont let script-defined points override shot maps
            continue
        mont = Montage(name=normalize_path(mapfile).name, map_id=mapid,
                       map_file=normalize_path(mapfile), map_frames=mapframes, status=status)
        montages[mont.name] = mont
    return montages

def load_nav_and_montages(nav_path: Path, project_name: str, overwrite: bool) -> Dict[str, Montage]:
    """Parse .nav file to build Montage objects, optionally loading existing data."""
    montages = preview_nav_montages(nav_path)
    nav_folder = nav_path.parent
    ensure_project_dirs(nav_folder, project_name)
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

def update_montages_if_map_notfound(nav_path: Path, montages: Dict[str, Montage]) -> str:
    items = read_nav_file(str(nav_path))
    for it in items:
        try:
            if getattr(it, "Type", None) != 2:
                continue

            mapfile_raw = getattr(it, "MapFile")
            mapfile_norm = normalize_path(mapfile_raw)
            if mapfile_norm.name not in montages.keys():
                map_file = nav_path.parent / mapfile_norm.name
                map_id = getattr(it, "MapID")
                map_frames = getattr(it, "MapFramesXY")
                mont = Montage(name=map_file.name, map_id=map_id, map_file=map_file, map_frames=map_frames)
                montages[mont.name] = mont
                return "Updated"
        except Exception as e:
            return str(e)
    return "Not Found."

def update_montage_if_map_generated(nav_path: Path, mont: Montage) -> str:
    """
    重新读取 nav_path，查找是否存在 Type==2 且 MapFile 对应 mont.map_file 的条目。
    如果找到，更新 mont 的关键信息并返回 True；否则返回 False。
    注意：该函数只**更新传入的 montage 对象（就地修改）**，不返回新的对象。
    """
    items = read_nav_file(str(nav_path))
    for it in items:
        try:
            if getattr(it, "Type", None) != 2:
                continue
            mapfile_raw = getattr(it, "MapFile", None)
            mapfile_norm = normalize_path(mapfile_raw)
            if mapfile_norm.name != mont.name:
                continue
            # 找到匹配项 -> 更新 montage 字段
            mont.map_file = nav_path.parent / mapfile_norm.name
            mont.map_id = getattr(it, "MapID", mont.map_id)
            mont.map_frames = getattr(it, "MapFramesXY", mont.map_frames)
            return "Updated"
        except Exception as e:
            return str(e)
    return "Not Found."

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

def check_overlap(cx: float, cy: float, size: float, dets: List[Detection]) -> bool:
    r1_x1, r1_y1 = cx - size / 2, cy - size / 2  # 左上角
    r1_x2, r1_y2 = cx + size / 2, cy + size / 2  # 右下角

    for d in dets:
        r2_x1, r2_y1 = d.x - d.w / 2, d.y - d.h / 2
        r2_x2, r2_y2 = d.x + d.w / 2, d.y + d.h / 2

        # Intersection
        dx = min(r1_x2, r2_x2) - max(r1_x1, r2_x1)
        dy = min(r1_y2, r2_y2) - max(r1_y1, r2_y1)
        if dx > 0 and dy > 0:
            return True
    return False


def collect_and_map_points_for_montage(montage_name: str, map_frames: List[int], preds_dir: Path, nav_folder: Path)\
        -> Tuple[List, Dict[int, List[Detection]]]:
    """
    收集单个 Montage 的所有 detections，并映射到全局坐标。
    返回: (全局坐标点列表, z_index -> 局部 detections 列表)
    """
    # 1. 获取 PieceSpacing
    mdoc_path = nav_folder / f"{montage_name}.mdoc"
    spacing_x, spacing_y = read_mdoc_spacing(mdoc_path)
    if spacing_x == 0 or spacing_y == 0:
        return [], {}

    nx, ny = map_frames  # (columns, rows)
    all_points = []  # list of {'global_x', 'global_y', 'conf', 'mont_name', 'tile_z', 'local_idx', 'det'}
    tile_dets_map = {}  # z -> list of Detections

    mont_stem = Path(montage_name).stem
    preds_for_montage = sorted(preds_dir.glob(f"{mont_stem}_tile*.txt"))
    for p_path in preds_for_montage:
        mont, z = match_name(p_path.name, _PRED_RE)
        if z == -1:
            continue

        dets = read_detections(p_path)
        tile_dets_map[z] = dets
        # 映射局部坐标到全局坐标
        # z=0 -> row=0, col=0
        # z=1 -> row=1, col=0 (Y increases)
        # ...
        # z=ny -> row=0, col=1 (X increases, Y resets)
        col = z // ny  # X index
        row = z % ny  # Y index
        offset_x = col * spacing_x
        offset_y = row * spacing_y

        for i, d in enumerate(dets):
            gx = offset_x + d.x
            gy = offset_y + d.y
            all_points.append({'global_x': gx, 'global_y': gy, 'conf': d.conf,
                               'mont_name': montage_name, 'tile_z': z, 'local_idx': i, 'det': d})

    return all_points, tile_dets_map


def deduplicate_global_points(all_points: List, box_size: float) -> Tuple[List, Set[Tuple[int, int]]]:
    """
    对全局坐标点列表执行碰撞去重逻辑，保留置信度高的点。
    返回: (最终唯一的高置信度点列表, 被移除点的键集合)
    """
    removals = set()  # Set of (tile_z, local_idx) to remove
    # 预排序：按置信度 (conf) 降序排列，确保高置信度点优先保留
    all_points.sort(key=lambda p: p['conf'], reverse=True)

    for i in range(len(all_points)):
        p1 = all_points[i]
        # If p1 is already marked for removal, it shouldn't suppress others
        if (p1['tile_z'], p1['local_idx']) in removals:
            continue

        for j in range(i + 1, len(all_points)):
            p2 = all_points[j]
            # If p2 is already removed, skip it
            if (p2['tile_z'], p2['local_idx']) in removals:
                continue

            # 使用 check_overlap 检查碰撞；p1 作为中心点，p2 作为检测目标
            tmp_det = [Detection(cls=0, x=p2['global_x'], y=p2['global_y'], w=box_size, h=box_size, conf=p2['conf'], status="active")]
            if check_overlap(p1['global_x'], p1['global_y'], box_size, tmp_det):
                # 因为已排序，p1 的置信度总是大于或等于 p2，所以移除 p2
                removals.add((p2['tile_z'], p2['local_idx']))

    final_points = [p for p in all_points if (p['tile_z'], p['local_idx']) not in removals]
    return final_points, removals

def add_predictions_to_nav(input_nav: Path, project_name: str, output_nav: Path, selected_montage_names: List[str], box_size: float) -> str:
    """Export selected and deduplicated predictions back into a new .nav file."""
    # 读取 nav 文件并建立 Map lookup: MapFile name -> NavItem
    items = read_nav_file(str(input_nav))
    map_lookup = build_map_lookup(items)
    if not map_lookup:
        return f"Failed to save: No Map items found in {input_nav}."

    images_dir, preds_dir = ensure_project_dirs(input_nav.parent, project_name)
    removed_points = 0
    new_items = []
    group_id = 1        # GroupID starts from 1 (per-file group), serialEM cannot recognize GroupID = 0
    tag_id= 1           # set class iterator to continue numbering
    msg = "."
    for name in selected_montage_names:
        map_item = map_lookup.get(name)
        if not map_item:
            msg = f" {name}" + msg
            continue

        try:
            map_id = map_item.MapID
            map_frames = map_item.MapFramesXY
            stage_z = map_item.StageXYZ[2]
        except Exception:
            msg = f" {name}" + msg
            continue

        montage_points, tile_dets_map = collect_and_map_points_for_montage(name, map_frames, preds_dir, input_nav.parent)
        # list of {'global_x', 'global_y', 'conf', 'mont_name', 'tile_z', 'local_idx', 'det'}
        montage_final_points, removals = deduplicate_global_points(montage_points, box_size)
        removed_points += len(removals)

        grouped_dets = defaultdict(list)
        # 遍历列表，进行分组
        for point in montage_final_points:
            # 提取 tile_z 作为键
            tile_z = point['tile_z']
            # 提取 det 对象作为值，并添加到对应键的列表中
            det_object = point['det']
            grouped_dets[tile_z].append(det_object)

        # 遍历字典，更新预测文件，并且写入nav_items
        for tile_z, dets in grouped_dets.items():
            pred_name = PRED_NAME_TEMPLATE.format(montage=Path(name).stem, z=tile_z)
            p_path = preds_dir / pred_name
            write_detections(p_path, dets)

            # All NavItems from one tile get the same GroupID
            first = True
            for d in dets:
                item = {
                    'Color': 0, 'NumPts': 0, 'Regis': 1, 'Type': 0,
                    'GroupID': group_id, 'DrawnID': int(map_id),
                    'CoordsInPiece': [float(d.x), float(d.y), float(stage_z)], 'PieceOn': int(tile_z),
                }
                # first nav item gets Acquire=1
                if first:
                    item['Acquire'] = 1
                    first = False

                # create NavItem (tag will be autogenerated as Item-<number>)
                nav_item = NavItem(item, f'{project_name}-{tag_id}')
                new_items.append(nav_item)
                tag_id += 1
            group_id += 1

    # copy input nav to output nav (overwrite if exists)
    shutil.copy2(str(input_nav), str(output_nav))
    # Append new items to output nav
    write_nav_file(str(output_nav), *new_items, mode='a')

    return (f"Saved to {output_nav}. Totally saved {group_id - 1} groups, {tag_id - 1} points.\n"
            f"Found {len(map_lookup)} maps in {input_nav}; Selected {len(selected_montage_names)} maps; Deduplicated {removed_points} points. "
            f"Skipped following montages for reading issue:" + msg)

def build_map_lookup(items) -> Dict[str, NavItem]:
    """Return dict mapping MapFile name -> MapItem instance"""
    lookup = {}
    for it in items:
        if getattr(it, 'Type', None) == 2:
            name = normalize_path(it.MapFile).name
            lookup[name] = it
    return lookup

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