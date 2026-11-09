# -*- mode: python ; coding: utf-8 -*-
import os
import sys

from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.win32.versioninfo import (
    VSVersionInfo,
    FixedFileInfo,
    StringFileInfo,
    StringTable,
    StringStruct,
    VarFileInfo,
    VarStruct,
)

sys.path.insert(0, os.getcwd())
from version import (
    APP_COMPANY,
    APP_COPYRIGHT,
    APP_DESCRIPTION,
    APP_NAME,
    APP_VERSION,
    APP_VERSION_INFO,
)

datas = [('engine.py', '.'), ('CHANGELOG.md', '.')]
binaries = []
hiddenimports = [
    'open3d',
    'pandas',
    'numpy',
    'matplotlib',
    'matplotlib.backends.backend_tkagg',
    'appdirs',
    'plotly',
    'plotly.graph_objects',
    'dash',
    'dash.html',
    'dash.dcc',
]
excludes = [
    'torch',
    'torchvision',
    'torchaudio',
    'tensorboard',
    'cv2',
    'onnxruntime',
    'IPython',
    'ipykernel',
    'ipywidgets',
    'jupyter_client',
    'jupyter_core',
    'notebook',
    'nbformat',
    'traitlets',
    'pytest',
    'black',
    'numba',
    'llvmlite',
]
tmp_ret = collect_all('open3d')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

version_info = VSVersionInfo(
    ffi=FixedFileInfo(
        filevers=APP_VERSION_INFO,
        prodvers=APP_VERSION_INFO,
        mask=0x3F,
        flags=0x0,
        OS=0x40004,
        fileType=0x1,
        subtype=0x0,
        date=(0, 0),
    ),
    kids=[
        StringFileInfo(
            [
                StringTable(
                    '040904B0',
                    [
                        StringStruct('CompanyName', APP_COMPANY),
                        StringStruct('FileDescription', APP_DESCRIPTION),
                        StringStruct('FileVersion', APP_VERSION),
                        StringStruct('InternalName', APP_NAME),
                        StringStruct('OriginalFilename', 'PointCloudDownsampler.exe'),
                        StringStruct('ProductName', APP_NAME),
                        StringStruct('ProductVersion', APP_VERSION),
                        StringStruct('LegalCopyright', APP_COPYRIGHT),
                    ],
                )
            ]
        ),
        VarFileInfo([VarStruct('Translation', [1033, 1200])]),
    ],
)


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PointCloudDownsampler',
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
    version=version_info,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PointCloudDownsampler',
)
