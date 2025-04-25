import os
import subprocess
from pathlib import Path

BLENDER_PATH = '/scratch/duanyiyang/3D_CV/blender-2.90.0-linux64/blender'  # 你的blender可执行文件路径
RENDER_SCRIPT = '/scratch/duanyiyang/3D_CV/utils/blender.py'  # 你的blender渲染脚本
# OUTPUT_ROOT = '/your/output/folder'
OBJ_ROOT = '/scratch/duanyiyang/datasets/shapenet/ShapeNetCore.v2/ShapeNetCore.v2'

categories = [
    '02691156',  # airplane
    '02828884',  # bench
    '03001627',  # chair
    '03211117',  # display
    '04379243'   # table
]

for cat in categories:
    for model_dir in Path(OBJ_ROOT, cat).iterdir():
        obj_path = model_dir / 'models' / 'model_normalized.ply'
        if not obj_path.exists():
            continue
        output_dir = model_dir / 'screenshots'
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            BLENDER_PATH, '--background', '--python', RENDER_SCRIPT, '--',
            '--views', '24',
            '--resolution', '224',
            '--output_folder', str(output_dir),
            '--scale', '1.0',
            str(obj_path)
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)
