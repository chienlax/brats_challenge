# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['scripts/visualize_predictions.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('apps', 'apps'),
        ('outputs/nnunet/predictions/Dataset501_BraTSPostTx/3d_fullres_all', 'outputs/nnunet/predictions/Dataset501_BraTSPostTx/3d_fullres_all'),
        ('outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx', 'outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx'),
    ],
    hiddenimports=['dash', 'dash_bootstrap_components', 'nibabel', 'numpy', 'plotly', 'scipy', 'imageio', 'matplotlib'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BraTS_Visualizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BraTS_Visualizer',
)
