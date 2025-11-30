# -*- mode: python ; coding: utf-8 -*-
#
# pyinstaller build.spec
# 此配置将创建目录形式的打包 (非单个可执行文件, 即 --onedir 模式).

import sys
from PyInstaller.utils.hooks import collect_all

# --- Analysis (分析阶段) ---
a = Analysis(
    ['src/gui.py'],
    pathex=['.'], # 搜索路径，当前目录
    binaries=[],
    # datas 列表用于添加数据文件和文件夹。
    # 格式为：(源路径, 打包后的目标路径)
    datas=[
        ('data', 'data'),  # 将 'data' 文件夹及其内容包含进去，目标文件夹仍命名为 'data'
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# --- PYZ (Python 压缩包) ---
# 将 Python 模块打包成一个压缩包
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# --- EXE (可执行文件) ---
# 创建主可执行文件
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Point-Picker',  # 最终可执行文件的名称
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='data/pp_gemini.ico',  # 设置程序图标，路径是相对于 spec 文件的
)

# --- COLLECT (收集) ---
# 收集所有依赖和数据文件到最终的发布目录
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Point-Picker',
)