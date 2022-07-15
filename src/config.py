import os
import sys
import pwd
import torch
import random
import psutil
import platform
import datetime
import numpy as np
import pandas as pd
import pynvml # pip install nvidia-ml-py

# config
# default
CONFIG = 'default'
LR = 0.001
G_READOUT = 'both'
G_FUSION = 'cat'
DROPOUT = 0.1
EMB_DIM = 64
HAZ_DIM = 64
G_HID_LAYER = 3
G_HID_DIM = 64

# 3306: 77.02±1.97
# CONFIG = '3306'
# LR = 0.001
# G_READOUT = 'avg'
# G_FUSION = 'cat'
# DROPOUT = 0.3
# EMB_DIM = 32
# HAZ_DIM = 16
# G_HID_LAYER = 3
# G_HID_DIM = 16

# 3637: 76.94±1.34
# CONFIG = '3637'
# LR = 0.0005
# G_READOUT = 'avg'
# G_FUSION = 'dot'
# DROPOUT = 0.1
# EMB_DIM = 32
# HAZ_DIM = 128
# G_HID_LAYER = 4
# G_HID_DIM = 32

# 3444:
# CONFIG = '3444'
# LR = 0.001
# G_READOUT = 'avg'
# G_FUSION = 'cat'
# DROPOUT = 0.5
# EMB_DIM = 16
# HAZ_DIM = 32
# G_HID_LAYER = 4
# G_HID_DIM = 16

GLOBALSEED = 42
TRIAL = 100
K = 5
BATCH = 32
EPOCHS = 100
SAGPOOL_RATIO = 0.5
MVPOOL_RATIO = 0.8
INTERVALS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Baseline methods
CPH_PENALTY = 1e-2
RSF_N_ESTIMATORS = 100

OUT_DIM = 1
G_INP_DIM = 1
NUM_HEADS = 2

# IBS grid number
IBS_GRID_N = 100

KEY_GENUS = {
    570: 'Klebsiella',
    1678: 'Bifidobacterium',
    102106: 'Collinsella',
    577309: 'Paraprevotella',
    375288: 'Parabacteroides',
    239759: 'Alistipes',
    816: 'Bacteroides',
    1578: 'Lactobacillus',
    1350: 'Enterococcus',
    1730: 'Eubacterium',
    216851: 'Faecalibacterium',
    33024: 'Phascolarctobacterium',
    29465: 'Veillonella',
    239934: 'Akkermansia',
    848: 'Fusobacterium'
}

# malignant melanoma, 悪性黒色腫
MagMela_GENUS = {
    570: 'Klebsiella',
    1678: 'Bifidobacterium',
    102106: 'Collinsella',
    375288: 'Parabacteroides',
    816: 'Bacteroides',
    1578: 'Lactobacillus',
    1350: 'Enterococcus',
    216851: 'Faecalibacterium',
    29465: 'Veillonella',
    239934: 'Akkermansia',
}

# NSCLC
NSCLC_GENUS = {}

# TAXID_NAME = {arr[0]:arr[1] for line in open('../data/intermediate_data/MIND_taxid_name.txt', 'r') if (arr:=line.strip().split())}
TAXID_NAME = {arr[0]:arr[1].split(';')[-1].split('__')[1] for line in open('../data/intermediate_data/b3_taxid_lineage.txt', 'r') if (arr:=line.strip().split('\t'))}

# TOP30 = ['1263', '1301', '1716', '102106', '39948', '1279', '838', '40544', '189330', '35832', '29465', '1743', '1578', '1357', '239759', '848', '1350', '239934', '1654', '33042', '216851', '572511', '1678', '841', '836', '119852', '375288', '724', '28050', '32207']

def set_metaverse_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def print_title(title):
    n = len(title) + 2
    print('┌' + '─'*n + '┐')
    print(f'│ {title} │')
    print('└' + '─'*n + '┘')

def watermark():
    print_title('Experimental Environment')
    print(f'platform: {platform.platform()}')
    print(f'node: {platform.node()}')
    print(f'time: {datetime.datetime.now()}')
    print(f'python interpreter: {sys.executable}')
    print(f'python version: '+sys.version.replace('\n',''))
    print(f'torch version: {torch.__version__}')
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    if device == 'gpu':
        pynvml.nvmlInit()
        print(f'CUDA version: {torch.version.cuda}')
        print(f'driver version: {pynvml.nvmlSystemGetDriverVersion().decode()}')
        print(f'cuDNN version: {torch.backends.cudnn.version()}')
        print(f'gpu usable count: {torch.cuda.device_count()}')
        deviceCount = pynvml.nvmlDeviceGetCount()
        print(f'gpu total count: {deviceCount}')
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode()
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memstr = f'{int(meminfo.used/1024/1024):5d}M / {int(meminfo.total/1024/1024):5d}M, {meminfo.used/meminfo.total:3.0%}'
            temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
            # fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            power_status = pynvml.nvmlDeviceGetPowerState(handle)
            print(f'    gpu {i}: {name}, [mem] {memstr}, {temp:3d}°C, 🔋 {power_status}')
        pynvml.nvmlShutdown()
        # print(torch.cuda.memory_summary())

    print(f'cpu: [logical] {psutil.cpu_count()}, [physical] {psutil.cpu_count(logical=False)}, [usage] {psutil.cpu_percent()}%')
    print(f'current dir: {os.getcwd()}')

    user_info = pwd.getpwuid(os.getuid())
    print(f'user: {user_info.pw_name}')
    print(f'shell: {user_info.pw_shell}')

