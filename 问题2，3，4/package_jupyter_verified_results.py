"""Bundle exactly the figures executed and SHA-verified by the Jupyter notebook."""

from pathlib import Path
import hashlib
import json
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
REPORT=json.loads((ROOT/'问题二三四_Jupyter逐图执行核验.json').read_text(encoding='utf-8'))
assert REPORT['status']=='PASS' and len(REPORT['code_cells'])==8
assert not any(cell['has_error'] or cell['execution_count'] is None for cell in REPORT['code_cells'])
FIGURES={g:[ROOT/item['path'] for item in figs] for g,figs in REPORT['figures'].items()}
assert {name:len(paths) for name,paths in FIGURES.items()}=={'问题二':7,'问题三':6,'问题四':6}
for rows in REPORT['figures'].values():
    for item in rows:
        path=ROOT/item['path']
        assert path.stat().st_size==item['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']

EXT={'.py','.csv','.json','.md','.txt','.xlsx','.otf','.tif','.cpp','.so'}
SKIP={'__pycache__','.ipynb_checkpoints','.git'}
def code_tree(name):
    return [p for p in (ROOT/name).rglob('*') if p.is_file()
            and p.suffix.lower() in EXT and not SKIP.intersection(p.relative_to(ROOT).parts)]

q2=code_tree('q2_source') + code_tree('q3_source') + FIGURES['问题二']
q3=code_tree('q3_source') + code_tree('q3_legacy') + FIGURES['问题三']
q4=code_tree('q3_source') + code_tree('q4_updated') + FIGURES['问题四']
general=[ROOT/name for name in ['README.md','论文正文_问题二至四_时效修订.md',
 'run_revised.py','plot_time_decisions.py','build_execute_final_notebook.py',
 'package_jupyter_verified_results.py','问题二三四_全部图_已在Jupyter逐格运行.ipynb',
 '问题二三四_Jupyter逐图执行核验.json','问题二三四_Jupyter真实执行日志.txt',
 'requirements_问题二至四_Jupyter.txt']]
full=(q2+q3+q4+code_tree('q3_legacy')+general)
plans={
 '问题二_原版绘图_分情景方案与复核.zip':q2+general[:2],
 '问题三_22加4_时间主方案_三维通信与核验.zip':q3+general[:2],
 '问题四_无单站组_原版图件与求解复核.zip':q4+general[:2],
 '问题二三四_Jupyter已运行_19张图.zip':sum(FIGURES.values(),[])+general[6:8],
 '问题二三四_时效修订_完整代码数据与图表.zip':full,
}
summary={}
for name,files in plans.items():
    dest=OUT/name
    unique=sorted(set(files),key=lambda p:str(p.relative_to(ROOT)))
    assert all(p.exists() for p in unique)
    if dest.exists():dest.unlink()
    with ZipFile(dest,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for path in unique:archive.write(path,arcname=str(path.relative_to(ROOT)))
    with ZipFile(dest) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist())==len(unique)
        verified_figures={str(p.relative_to(ROOT)) for paths in FIGURES.values() for p in paths}
        if name.startswith('问题二三四_时效修订'):
            assert verified_figures <= set(archive.namelist())
        assert all(not member.endswith('.png') or member in verified_figures
                   for member in archive.namelist()),'禁止打包未在Jupyter执行的旧图'
    summary[name]={'files':len(unique),'size_mb':round(dest.stat().st_size/1048576,2)}
print(json.dumps(summary,ensure_ascii=False,indent=2))
