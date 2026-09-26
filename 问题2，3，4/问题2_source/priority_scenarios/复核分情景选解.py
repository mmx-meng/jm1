"""按题目时限复核问题二两个决策偏好；候选来自原问题二认证档案。"""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import hashlib
import io
import json
import pandas as pd

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
ARCHIVE = P.parents[2] / 'source_q2/华为杯/问题二_认证更新_代码与数据(1).zip'
SCAN = P / '候选档案筛查.csv'
VERIFIED = {
    '全局认证_冻结最终十九架次': ROOT / 'q2_source/全局认证_冻结最终十九架次',
    '全局认证_集成列池CP_候选007': P / '全局认证_集成列池CP_候选007',
    '全局认证_集成列池CP_候选019': P / '全局认证_集成列池CP_候选019',
    '问题二_真正时间优先_22架次': ROOT / 'q2_source/time_priority_3d/问题二_真正时间优先_22架次',
}

def metrics(stem: str, flights: pd.DataFrame, boxes: pd.DataFrame) -> dict:
    assert len(boxes) == 80 and boxes['货箱编号'].nunique() == 80
    assert len(flights) == flights['架次编号'].nunique()
    hard = boxes[boxes['硬截止秒'].notna()]
    urgent = hard[hard['硬截止秒'].eq(3600)]
    assert len(hard) == 31 and len(urgent) == 17
    assert (hard['硬截止秒'] - hard['送达秒']).min() >= 0
    assert abs(boxes['归一化加权延误'].sum()) < 1e-8
    return {'方案': stem, '运输架次': len(flights),
            '全部返航秒': int(flights['返回秒'].max()),
            '末箱交付秒': int(boxes['送达秒'].max()),
            '一小时紧急17箱末箱秒': int(urgent['送达秒'].max()),
            '一小时紧急17箱平均秒': float(urgent['送达秒'].mean()),
            '最小硬截止余量秒': int((hard['硬截止秒']-hard['送达秒']).min()),
            '总能耗kWh': float(flights['能耗kWh'].sum()),
            '延误': float(boxes['归一化加权延误'].sum())}

if ARCHIVE.exists():
    data = []
    with ZipFile(ARCHIVE) as z:
        names = set(z.namelist())
        for name in sorted(names):
            if not name.startswith('outputs/') or not name.endswith('_架次.csv'):
                continue
            source = name.removesuffix('_架次.csv')
            if source + '_逐箱.csv' not in names:
                continue
            flights = pd.read_csv(io.BytesIO(z.read(name)))
            boxes = pd.read_csv(io.BytesIO(z.read(source + '_逐箱.csv')))
            data.append(metrics(source.removeprefix('outputs/'), flights, boxes))
    fast = VERIFIED['问题二_真正时间优先_22架次']
    data.append(metrics(fast.name,
                        pd.read_csv(str(fast) + '_架次.csv'),
                        pd.read_csv(str(fast) + '_逐箱.csv')))
    pd.DataFrame(data).to_csv(SCAN, index=False, encoding='utf-8-sig')
else:
    assert SCAN.exists(), '需要原始问题二候选档案，或附带的冻结筛查CSV'

scan = pd.read_csv(SCAN)
baseline = sorted(scan.to_dict('records'),
                  key=lambda x: (x['运输架次'], x['全部返航秒'], x['总能耗kWh']))[0]
fastest = sorted(scan.to_dict('records'),
                 key=lambda x: (x['全部返航秒'], x['末箱交付秒'], x['最小硬截止余量秒']*-1))[0]
ref = int(scan['全部返航秒'].min())
cases = {'减少出动架次优先': baseline, '真正缩短全部完成时间': fastest}
for percent in [3, 5, 10]:
    options = scan[scan['全部返航秒'].le(ref * (1 + percent / 100))]
    chosen = sorted(options.to_dict('records'), key=lambda x: (
        -x['最小硬截止余量秒'], x['一小时紧急17箱平均秒'],
        x['运输架次'], x['总能耗kWh'], x['全部返航秒']))[0]
    cases[f'紧急余量优先_{percent}%时间容许差'] = chosen
assert baseline['方案'] == '全局认证_冻结最终十九架次'
assert fastest['方案'] == '问题二_真正时间优先_22架次'
assert cases['紧急余量优先_3%时间容许差']['方案'] == fastest['方案']
assert cases['紧急余量优先_5%时间容许差']['方案'] == fastest['方案']
assert cases['紧急余量优先_10%时间容许差']['方案'] == '全局认证_集成列池CP_候选019'

checks = {}
for name, prefix in VERIFIED.items():
    f = pd.read_csv(str(prefix) + '_架次.csv')
    d = pd.read_csv(str(prefix) + '_逐箱.csv')
    recomputed = metrics(name, f, d)
    scanned = scan[scan['方案'].eq(name)].iloc[0].to_dict()
    for col in recomputed:
        if col != '方案':
            assert abs(float(recomputed[col]) - float(scanned[col])) < 1e-7, (name, col)
    if name.endswith(('007', '019')):
        audit = P / ('优先级方案' + name[-3:] + '_原始DEM独立核验.json')
    elif name == '问题二_真正时间优先_22架次':
        audit = ROOT / 'q2_source/time_priority_3d/问题二_真正时间优先_原始DEM独立核验.json'
    else:
        audit = ROOT / 'q2_source/全局认证_冻结最终十九架次_独立核验.json'
    proof = json.loads(audit.read_text(encoding='utf-8'))
    assert proof['status'] in ['通过', '全部通过']
    dem_hash = proof.get('source_DEM_sha256', proof.get('source_sha256', {}).get('最终工作DEM.tif'))
    assert dem_hash == '02f3581304f2e52deaff44563d69d1b8f7ccfc9aedbb4886bc71eb50fee48c6d'
    assert int(proof.get('all_return_s', recomputed['全部返航秒'])) == recomputed['全部返航秒']
    checks[name] = {'原始DEM复核': proof['status'], '文件SHA256': {
        kind: hashlib.sha256(Path(str(prefix) + '_' + kind + '.csv').read_bytes()).hexdigest()
        for kind in ['架次', '逐箱', '航段']}}

result = {'status': 'PASS', '范围': '原始认证候选档案；未证明完整路线空间全局最优',
          '候选记录数': len(scan), '最快已核验候选返航秒': ref,
          '3%上限秒': ref * 1.03, '5%上限秒': ref * 1.05,
          'case_choices': cases, '原始DEM物理复核': checks}
(P / '分情景最终选解与复核.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v['方案'] for k, v in cases.items()}, ensure_ascii=False))
