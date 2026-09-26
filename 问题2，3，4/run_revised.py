"""Reproduce Q2's fast 22-sortie alternative, Q3's 22+4 and inherited Q4.

Requirements: Python with pandas, numpy, matplotlib, pillow, openpyxl,
ortools, rasterio/DEM dependencies present in the original model runtime.
Run: python run_revised.py
"""
from pathlib import Path
import subprocess, sys, json
from PIL import Image

ROOT=Path(__file__).resolve().parent
Q3=ROOT/'q3_source';Q4=ROOT/'q4_updated';OUT=ROOT/'figures'
Q2_SCENARIOS=ROOT/'q2_source/priority_scenarios'
Q2_FAST=ROOT/'q2_source/time_priority_3d'
Q2_OLD_PLOT=Q2_SCENARIOS/'original_q2_plots'
OUT.mkdir(exist_ok=True)

def command(*args,cwd=ROOT):
    print('RUN',*args,flush=True)
    subprocess.run([sys.executable,'-X','utf8',*map(str,args)],cwd=cwd,check=True)

command('solve_q2_fast22.py',cwd=Q2_FAST)
command('问题二_真正时间优先_原始DEM独立核验.py',cwd=Q2_FAST)
command('plot_q2_original_terrain_3d.py',cwd=Q2_FAST)
command('认证更新_调度装载交付.py','--source',Q2_FAST/'问题二_真正时间优先_22架次',
        '--output',Q2_FAST/'原版问题二_22架次时序图','--dpi','240',cwd=Q2_OLD_PLOT)
command('复核分情景选解.py',cwd=Q2_SCENARIOS)
command('问题三_V3_结果核验.py','--prefix','问题三_时间主方案',cwd=Q3)
command('compute_q4.py','--source',Q3,'--output',Q4/'data',cwd=Q4)
command('compute_q4_spatial.py','--q3-dir',Q3,'--data-dir',Q4/'data',cwd=Q4)
command('plot_time_decisions.py')
command('问题三_V4_统计绘图.py','--prefix','问题三_时间主方案',
        '--output-prefix',OUT/'问题三_时间主方案','--dpi','260',cwd=Q3)
command('问题三_V7_六面板三维山地航线.py','--prefix','问题三_时间主方案',
        '--geometry','问题三_时间主方案_精确',
        '--output-prefix',OUT/'问题三_时间主方案_三维','--dpi','230',cwd=Q3)
command('问题三_V4_地图绘图.py','--prefix','问题三_时间主方案',
        '--geometry','问题三_时间主方案_精确','--stations','问题三_V3_站点.csv',
        '--output-prefix',OUT/'问题三_时间主方案_地图','--dpi','220',cwd=Q3)
command('compute_q4_no_singletons.py',cwd=Q4)
for figure in ('plot_q4_no_singletons_maps.py','plot_q4_no_singletons_answer.py',
               'plot_q4_no_singletons_hex.py','plot_q4_no_singletons_ellipses.py',
               'plot_q4_no_singletons_timeline.py'):
    command(figure,cwd=Q4)
reports=[json.loads((Q3/'问题三_时间主方案_V3_独立核验.json').read_text(encoding='utf-8')),
         json.loads((Q2_FAST/'问题二_真正时间优先_原始DEM独立核验.json').read_text(encoding='utf-8')),
         json.loads((Q4/'data/qa.json').read_text(encoding='utf-8')),
         json.loads((Q4/'no_singletons/results.json').read_text(encoding='utf-8'))]
assert reports[0]['status']=='全部通过' and reports[0]['精确通信核验']['failed_coverage_count']==0
assert reports[1]['status']=='通过' and reports[1]['all_return_s']==6899
assert reports[2]['status']=='PASS' and reports[2]['recommendations']=={'2':'K2_02','3':None}
assert reports[3]['checks']['cp_sat_optimal'] and reports[3]['checks']['no_single_site_group']
images=[p for p in OUT.glob('*.png') if '.tmp.' not in p.name and not p.name.startswith(('问题四_','时效图04_'))]
images += list((Q4/'no_singletons/figures').glob('*.png'))
images += list(Q2_FAST.glob('问题二_原版视角三维_*.png'))
images += list((Q2_FAST/'原版问题二_22架次时序图').glob('*.png'))
assert len(images)>=13
for path in images:
    with Image.open(path) as image:image.load()
print('PASS',len(images),'valid PNGs and all Q3/Q4 independent checks')
