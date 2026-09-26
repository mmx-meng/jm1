"""直接运行问题二原认证版九面板、资源和交付绘图代码。"""
from pathlib import Path
import os
import subprocess
import sys

P = Path(__file__).resolve().parent
Q2 = P.parent
PLOTS = P / 'original_q2_plots'
cases = [
    ('省架次与总体完成优先', Q2 / '全局认证_冻结最终十九架次', '省架次优先_原版时序图'),
    ('3%时效容许差', P / '全局认证_集成列池CP_候选007', '3%时效容许差_原版时序图'),
    ('紧急硬时限余量优先', P / '全局认证_集成列池CP_候选019', '紧急箱优先_原版时序图'),
]
for label, stem, directory in cases:
    env = dict(os.environ, Q2_CERTIFIED_PREFIX=str(stem.resolve()))
    subprocess.run([sys.executable, str(PLOTS / '认证更新_图01_九面板地图.py')],
                   cwd=PLOTS, env=env, check=True)
    (PLOTS / '认证更新_图01_九面板运输地图.png').replace(
        PLOTS / f'问题二_原版九面板_{label}.png')
    (PLOTS / '认证更新_图01_绘图核验.json').replace(
        PLOTS / f'原版图源核验_{label}.json')
    subprocess.run([sys.executable, str(PLOTS / '认证更新_调度装载交付.py'),
                    '--source', str(stem), '--output', str(PLOTS / directory),
                    '--dpi', '230'], cwd=PLOTS, check=True)
print('三组原版九面板、交付与资源时序图重绘完成')
