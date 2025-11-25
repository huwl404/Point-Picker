#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : gpu.py
# Time       : 2025/11/11 14:10
# Author     : 14750
# Email      : huwl@hku.hk
# Description：

"""
import torch
print(torch.version.cuda)  # 应显示 CUDA 版本，而不是 None

# pip uninstall torch
# pip install torch --index-url https://download.pytorch.org/whl/cu117

# torchvision 0.23.0 requires torch==2.8.0, but you have torch 2.0.1+cu117 which is incompatible.


# pip install torch --index-url https://download.pytorch.org/whl/cu126
# Successfully installed torch-2.8.0+cu126

# UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11070).
# Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx
# Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver.
# (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\c10\cuda\CUDAFunctions.cpp:109.)
# return torch._C._cuda_getDeviceCount() > 0
# 更新驱动后解决该问题。

ok = torch.cuda.is_available()
if ok:
    dev_count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if dev_count > 0 else "GPU"
    print(f"{name} (count={dev_count})")
print("torch found, cuda not available")
