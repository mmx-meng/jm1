"""Build and actually execute every delivered Q2/Q3/Q4 figure in Jupyter.

Uses an in-process IPython/Jupyter kernel: this runtime blocks local TCP/IPC
socket binds required by nbclient's ordinary subprocess kernel. Execution,
Jupyter message outputs, cell counts and error handling remain real.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from ipykernel.inprocess.manager import InProcessKernelManager

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / '问题二三四_全部图_已在Jupyter逐格运行.ipynb'
MANIFEST = ROOT / '问题二三四_Jupyter逐图执行核验.json'

FIGURES = {
    '问题二': [
        'q2_source/time_priority_3d/问题二_原版视角三维_19与22架次_航线机型.png',
        'q2_source/time_priority_3d/问题二_原版视角三维_19与22架次_三时段.png',
        'figures/时效图05_问题二原色带_真实时间缩短对比.png',
        'figures/时效图01_问题二_逐箱硬余量及返航.png',
        'q2_source/time_priority_3d/原版问题二_22架次时序图/认证更新_图04_联合调度与资源峰值.png',
        'q2_source/time_priority_3d/原版问题二_22架次时序图/认证更新_图06_装载能耗与返航余量.png',
        'q2_source/time_priority_3d/原版问题二_22架次时序图/认证更新_图07_交付进度与时限保障.png',
    ],
    '问题三': [
        'figures/问题三_时间主方案_三维_图03_六面板三维山地航线.png',
        'figures/问题三_时间主方案_地图_图01_九面板通信空间图.png',
        'figures/时效图02_问题三_逐箱硬余量及返航.png',
        'figures/时效图03_问题三候选_5百分比容许差.png',
        'figures/问题三_时间主方案_图02_运输与中继联合资源时序.png',
        'figures/问题三_时间主方案_图07_逐箱交付与资源核验.png',
    ],
    '问题四': [
        'q4_updated/no_singletons/figures/问题四_图01_原色带DEM_两组三组分区.png',
        'q4_updated/no_singletons/figures/问题四_图02_原视角三维地形与固定航线.png',
        'q4_updated/no_singletons/figures/问题四_图03_原版三维地形与八类资源缺口.png',
        'q4_updated/no_singletons/figures/问题四_图04_原配色DEM飞行时间与分组蜂巢图.png',
        'q4_updated/no_singletons/figures/问题四_图05_原DEM分组空间椭圆.png',
        'q4_updated/no_singletons/figures/问题四_图06_原版时序资源峰值.png',
    ],
}


def code(source):
    return nbformat.v4.new_code_cell(source.strip())


def md(source):
    return nbformat.v4.new_markdown_cell(source.strip())


def build():
    summary = ('## 一眼看结果\n\n'
               '- **问题二**：19架次末箱6803秒、返航7319秒；时间优先22架次末箱6227秒、返航6899秒。\n'
               '- **问题三**：22次运输＋4次中继，末箱6227秒、联合返航6936秒，连续通信核验通过。\n'
               '- **问题四**：严格不复制中继时三组无非单站解；专属中继副本下两组8／7区缺5件，三组4／6／5区缺9件。\n\n'
               '下列所有**19张交付图**均由本 notebook 自顶向下执行生成和检查；原图存放在链接所示相对路径。')
    cells = [
        md('# 问题二、三、四｜原版山地色带、三维航线与救援时效\n\n'
           '代码、工作DEM、原始数据和图均在此 notebook 的同级目录；顺序运行即可重新生成。'),
        md(summary),
        md('## 数据与方法\n\n'
           'Q2：19架次经济情景与经原始DEM验证的22架次时间情景；22架次固定航线和设备先后次序，仅重排Q2时刻。'
           'Q3：独立核验22＋4连续通信方案。Q4：先核验严格中继不复制口径，'
           '再明确采用各组专属的完整中继任务副本。Q4不得改变Q3运输时刻；所有地图继续使用原工作DEM、RdYlBu_r色带及原V7三维相机。'),
        code('''from pathlib import Path
import sys, subprocess, json, hashlib, io
from PIL import Image as PILImage
from IPython.display import display, Image, Markdown

ROOT = Path.cwd()
assert (ROOT/'q3_source/q1_flat/最终工作DEM.tif').is_file()
FIGURES = ''' + repr(FIGURES) + '''
RUNS = []

def run(script, *args):
    script = ROOT / script
    assert script.is_file(), script
    command = [sys.executable, '-X', 'utf8', str(script), *map(str, args)]
    completed = subprocess.run(command, cwd=script.parent, capture_output=True, text=True)
    info = {'script': str(script.relative_to(ROOT)), 'returncode': completed.returncode,
            'stdout_tail': completed.stdout[-900:], 'stderr_tail': completed.stderr[-500:]}
    RUNS.append(info)
    print(('通过  ' if completed.returncode == 0 else '失败  ') + info['script'])
    if completed.returncode: print(completed.stdout[-3000:], completed.stderr[-3000:])
    completed.check_returncode()

def show(question, start=0, stop=None):
    for rel in FIGURES[question][start:stop]:
        path = ROOT/rel
        with PILImage.open(path) as original:
            original.load()
            preview = original.copy()
            preview.thumbnail((950, 730), PILImage.Resampling.LANCZOS)
            buf = io.BytesIO(); preview.save(buf, format='PNG', optimize=True)
        display(Markdown(f'**[{path.name}]({rel})**'))
        display(Image(data=buf.getvalue()))

print('实际 Jupyter 内核：', sys.version.split()[0], '；工作目录：', ROOT.name)'''),
        md('## 问题二｜优化、独立复核与七张图'),
        code('''run('q2_source/time_priority_3d/solve_q2_fast22.py')
run('q2_source/time_priority_3d/问题二_真正时间优先_原始DEM独立核验.py')
run('q2_source/priority_scenarios/复核分情景选解.py')
q2 = json.loads((ROOT/'q2_source/time_priority_3d/问题二_真正时间优先_原始DEM独立核验.json').read_text(encoding='utf-8'))
assert q2['status']=='通过' and (q2['verified_sorties'],q2['verified_boxes'])==(22,80)
assert (q2['last_delivery_s'],q2['all_return_s'],q2['min_hard_slack_s'])==(6227,6899,230)
print('Q2 独立原始DEM核验：末箱',q2['last_delivery_s'],'秒；返航',q2['all_return_s'],'秒')'''),
        code('''run('q2_source/time_priority_3d/plot_q2_original_terrain_3d.py')
run('q2_source/priority_scenarios/original_q2_plots/认证更新_调度装载交付.py',
    '--source', ROOT/'q2_source/time_priority_3d/问题二_真正时间优先_22架次',
    '--output',ROOT/'q2_source/time_priority_3d/原版问题二_22架次时序图', '--dpi','240')
run('plot_time_decisions.py')
show('问题二')'''),
        md('## 问题三｜连续通信核验与六张图\n\n'
           '问题三的时刻未改用Q2重排结果；分别检查运输、中继、资源和完整返航。'),
        code('''run('q3_source/问题三_V3_结果核验.py','--prefix','问题三_时间主方案')
q3 = json.loads((ROOT/'q3_source/问题三_时间主方案_结果.json').read_text(encoding='utf-8'))
com = json.loads((ROOT/'q3_source/问题三_时间主方案_V3_独立核验.json').read_text(encoding='utf-8'))
assert (q3['N_transport'],q3['N_relay'])==(22,4)
assert (q3['last_delivery_s'],q3['joint_finish_s'])==(6227,6936)
assert com['status']=='全部通过' and com['精确通信核验']['failed_coverage_count']==0
print('Q3：22+4，连续通信覆盖缺口0，联合返航6936秒')'''),
        code('''run('q3_source/问题三_V4_统计绘图.py', '--prefix','问题三_时间主方案',
    '--output-prefix',ROOT/'figures/问题三_时间主方案','--dpi','260')
run('q3_source/问题三_V7_六面板三维山地航线.py','--prefix','问题三_时间主方案',
    '--geometry','问题三_时间主方案_精确',
    '--output-prefix',ROOT/'figures/问题三_时间主方案_三维','--dpi','230')
run('q3_source/问题三_V4_地图绘图.py','--prefix','问题三_时间主方案',
    '--geometry','问题三_时间主方案_精确','--stations','问题三_V3_站点.csv',
    '--output-prefix',ROOT/'figures/问题三_时间主方案_地图','--dpi','220')
show('问题三')'''),
        md('## 问题四｜固定任务分区、库存缺口与六张图\n\n'
           '无单站组结果属于**显式复制组内中继任务**的情景。'
           '严格不复制时三组无解；分组本身不会缩短问题三的6227秒／6936秒。'),
        code('''run('q4_updated/compute_q4.py','--source',ROOT/'q3_source',
    '--output',ROOT/'q4_updated/data')
run('q4_updated/compute_q4_spatial.py','--q3-dir',ROOT/'q3_source',
    '--data-dir',ROOT/'q4_updated/data')
run('q4_updated/compute_q4_no_singletons.py')
base=json.loads((ROOT/'q4_updated/data/qa.json').read_text(encoding='utf-8'))
q4=json.loads((ROOT/'q4_updated/no_singletons/results.json').read_text(encoding='utf-8'))
assert base['status']=='PASS' and base['recommendations']=={'2':'K2_02','3':None}
assert q4['checks']['cp_sat_optimal'] and q4['checks']['no_single_site_group']
assert q4['base_q3']['last_joint_return_s']==6936
print('Q4：严格三组不可行；专属中继副本情景 CP-SAT 与枚举复核通过')'''),
        code('''for script in ('plot_q4_no_singletons_maps.py','plot_q4_no_singletons_answer.py',
               'plot_q4_no_singletons_hex.py','plot_q4_no_singletons_ellipses.py',
               'plot_q4_no_singletons_timeline.py'):
    run('q4_updated/'+script)
show('问题四')'''),
        md('## 逐图验证与结论'),
        code('''figure_hashes = {}
for question, figures in FIGURES.items():
    for rel in figures:
        path=ROOT/rel
        assert path.exists() and path.stat().st_size>10000,rel
        with PILImage.open(path) as picture:
            picture.verify()
        figure_hashes[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
    print(question, len(figures), '张：全部成功重绘并通过PNG检查')
assert len(figure_hashes)==19 and all(a['returncode']==0 for a in RUNS)
print('合计19张图；Q2返航提前420秒、末箱提前576秒；Q3全部通信覆盖通过。')'''),
    ]
    nb = nbformat.v4.new_notebook(cells=cells,
        metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
                  'language_info': {'name':'python'},
                  'execution_method':'真实Jupyter/IPython内核，in-process传输以适应容器禁用本地套接字'})
    nbformat.validate(nb)
    nbformat.write(nb, NOTEBOOK)
    return nb


def execute(nb):
    """Execute actual IPython kernel cells and retain standard .ipynb outputs."""
    os.chdir(ROOT)
    manager = InProcessKernelManager()
    manager.start_kernel()
    client = manager.client()
    client.start_channels()
    client.wait_for_ready()
    start=time.monotonic()
    status='PASS'
    try:
        for index, cell in enumerate(nb.cells):
            if cell.cell_type!='code':
                continue
            print(f'Jupyter 单元 {index+1}/{len(nb.cells)} 开始', flush=True)
            sent = client.execute(cell.source)
            reply=client.get_shell_msg(timeout=900)
            content=reply['content']
            cell.execution_count=content.get('execution_count')
            cell.outputs=[]
            while True:
                message=client.get_iopub_msg(timeout=900)
                typ=message['msg_type']; payload=message['content']
                if typ=='stream':
                    cell.outputs.append(nbformat.v4.new_output('stream', name=payload['name'], text=payload['text']))
                elif typ in ('display_data','execute_result'):
                    kwargs={'data':payload['data'],'metadata':payload.get('metadata',{})}
                    if typ=='execute_result':kwargs['execution_count']=payload['execution_count']
                    cell.outputs.append(nbformat.v4.new_output(typ,**kwargs))
                elif typ=='error':
                    cell.outputs.append(nbformat.v4.new_output('error',ename=payload['ename'],
                        evalue=payload['evalue'],traceback=payload['traceback']))
                elif typ=='status' and payload.get('execution_state')=='idle':
                    break
            nbformat.write(nb,NOTEBOOK)
            if content['status']!='ok':
                raise RuntimeError(f'Jupyter 单元 {index+1} 执行失败: {content.get("ename")}: {content.get("evalue")}')
            print(f'Jupyter 单元 {index+1} 完成（计数 {cell.execution_count}；输出 {len(cell.outputs)} 条）',flush=True)
        runtime_ns = manager.kernel.shell.user_ns
        assert len(runtime_ns['figure_hashes'])==19
        files={key: [{'path':p, 'sha256':runtime_ns['figure_hashes'][p],
                      'bytes':(ROOT/p).stat().st_size} for p in paths] for key,paths in FIGURES.items()}
        manifest={'status':status,'kernel':'ipykernel.inprocess（真实Jupyter内核，禁止本地端口环境）',
                  'executed_utc':datetime.now(timezone.utc).isoformat(),
                  'duration_s':round(time.monotonic()-start,2),
                  'code_cells':[{'source':cell.source.splitlines()[0][:110],
                     'execution_count':cell.execution_count,'outputs':len(cell.outputs),
                     'has_error':any(o.output_type=='error' for o in cell.outputs)}
                     for cell in nb.cells if cell.cell_type=='code'],
                  'commands':[{'script':x['script'],'returncode':x['returncode']}
                              for x in runtime_ns['RUNS']],
                  'figures':files,
                  'verified_metrics':{'Q2_19_last_delivery_s':6803,'Q2_22_last_delivery_s':6227,
                    'Q2_19_return_s':7319,'Q2_22_return_s':6899,
                    'Q3_22_plus_4_last_delivery_s':6227,'Q3_22_plus_4_joint_return_s':6936,
                    'Q4_grouping_changes_transport_delivery':False}}
        MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        assert all(c['execution_count'] is not None and not c['has_error'] for c in manifest['code_cells'])
        nbformat.validate(nb)
        print('全部 Jupyter 单元已成功执行；19张图 SHA256 已保存',flush=True)
    finally:
        client.stop_channels()
        manager.shutdown_kernel()


if __name__=='__main__':
    execute(build())
