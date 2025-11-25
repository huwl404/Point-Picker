#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : qt5.py
# Time       : 2025/11/10 22:56
# Author     : 14750
# Email      : huwl@hku.hk
# Description：
PyQt5 版本: 5.15.3 !!! 5.15.2 does not work for me!!!
"""
import sys
print("Python 路径:", sys.prefix)
print("执行路径:", sys.executable)

# 检查 PyQt5 实际安装位置
import PyQt5
print("PyQt5 安装位置:", PyQt5.__file__)  # None


import sys
from PyQt5 import QtWidgets, QtCore, QtGui

print("PyQt5 版本:", QtCore.PYQT_VERSION_STR)
print("Qt 版本:", QtCore.QT_VERSION_STR)

# 测试基本功能
app = QtWidgets.QApplication(sys.argv)
print("QApplication 创建成功")

# 测试各个主要模块
print("QtWidgets 可用:", hasattr(QtWidgets, 'QMainWindow'))
print("QtGui 可用:", hasattr(QtGui, 'QPainter'))
print("QtCore 可用:", hasattr(QtCore, 'QTimer'))

print("所有模块导入成功！")