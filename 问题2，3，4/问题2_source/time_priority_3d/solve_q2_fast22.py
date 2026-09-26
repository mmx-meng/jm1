"""复用已核验山地航线，在问题二约束下重新优化22架次的真实时刻。

固定每条路线的货箱、顺序、机型及实体机/电池指派；问题二不加入中继
时窗。CP-SAT依次优化返航时间、末箱交付、最小硬余量和加权送达时刻。
证明仅覆盖这套固定路线及实体指派，不宣称全局最优。
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

from ortools.sat.python import cp_model
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / 'q3_source/问题三_时间主方案_运输'
STEM = OUT / '问题二_真正时间优先_22架次'
F = pd.read_csv(str(SOURCE) + '_架次.csv')
D = pd.read_csv(str(SOURCE) + '_逐箱.csv')
L = pd.read_csv(str(SOURCE) + '_航段.csv')
assert len(F) == 22 and len(D) == D['货箱编号'].nunique() == 80
old = F.set_index('架次编号')
assert D['归一化加权延误'].sum() == 0
assert len(D[D['硬截止秒'].notna()]) == 31
HORIZON = 9000
STAGES = []


def model(*, return_cap=None, delivery_cap=None, slack_floor=164,
          optimize='return'):
    m = cp_model.CpModel()
    starts = {r.架次编号: m.NewIntVar(0, HORIZON, f'start_{r.架次编号}')
              for r in F.itertuples(index=False)}
    returns = {r.架次编号: starts[r.架次编号] + int(r.返回秒 - r.开始秒)
               for r in F.itertuples(index=False)}
    batteries = {r.架次编号: starts[r.架次编号] + int(r.电池充满秒 - r.开始秒)
                 for r in F.itertuples(index=False)}
    # 原排班给出同一实体的合法复用顺序；保持该顺序但允许任务提前。
    for entity, occupied_to in [('无人机', returns), ('电池', batteries)]:
        for _, group in F.groupby(entity):
            ids = group.sort_values('开始秒')['架次编号'].tolist()
            for first, second in zip(ids[:-1], ids[1:]):
                m.Add(starts[second] >= occupied_to[first])

    arrivals = {}
    for row in D.itertuples(index=False):
        offset = int(row.送达秒 - old.loc[row.架次编号, '开始秒'])
        arrival = starts[row.架次编号] + offset
        arrivals[row.货箱编号] = arrival
        m.Add(arrival <= int(row.期望送达秒))
        if pd.notna(row.硬截止秒):
            m.Add(arrival <= int(row.硬截止秒) - slack_floor)

    last_return = m.NewIntVar(0, 2 * HORIZON, 'last_return')
    last_box = m.NewIntVar(0, 2 * HORIZON, 'last_delivery')
    m.AddMaxEquality(last_return, list(returns.values()))
    m.AddMaxEquality(last_box, list(arrivals.values()))
    if return_cap is not None:
        m.Add(last_return <= return_cap)
    if delivery_cap is not None:
        m.Add(last_box <= delivery_cap)
    min_hard = m.NewIntVar(0, 10800, 'minimum_hard_slack')
    for row in D[D['硬截止秒'].notna()].itertuples(index=False):
        m.Add(min_hard <= int(row.硬截止秒) - arrivals[row.货箱编号])

    if optimize == 'return':
        m.Minimize(last_return)
    elif optimize == 'last_delivery':
        m.Minimize(last_box)
    elif optimize == 'slack':
        m.Maximize(min_hard)
    elif optimize == 'weighted_delivery':
        m.Minimize(sum(int(row.优先系数) * arrivals[row.货箱编号]
                       for row in D.itertuples(index=False)))
    else:
        raise ValueError(optimize)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_search_workers = 4
    status = solver.Solve(m)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(f'{optimize}：{solver.StatusName(status)}，停止输出')
    STAGES.append({'objective': optimize, 'status': solver.StatusName(status),
                   'value': solver.ObjectiveValue(),
                   'bound': solver.BestObjectiveBound()})
    return (int(solver.Value(last_return)), int(solver.Value(last_box)),
            int(solver.Value(min_hard)),
            {name: int(solver.Value(start)) for name, start in starts.items()})


first = model(optimize='return')
second = model(return_cap=first[0], optimize='last_delivery')
third = model(return_cap=first[0], delivery_cap=second[1], optimize='slack')
fourth = model(return_cap=first[0], delivery_cap=second[1],
               slack_floor=third[2], optimize='weighted_delivery')
assert (first[0], second[1], third[2]) == (6899, 6227, 230)
times = fourth[3]
f, d = F.copy(), D.copy()
shift = {rid: start - int(old.loc[rid, '开始秒']) for rid, start in times.items()}
for col in ['开始秒', '返回秒', '电池充满秒']:
    f[col] = f[col] + f['架次编号'].map(shift)
d['送达秒'] = d['送达秒'] + d['架次编号'].map(shift)
assert int(f['返回秒'].max()) == first[0]
assert int(d['送达秒'].max()) == second[1]
assert int((d['硬截止秒'] - d['送达秒']).dropna().min()) == third[2]
assert abs(d['归一化加权延误'].sum()) < 1e-10
assert f['能耗kWh'].sum() > 0

f.to_csv(str(STEM) + '_架次.csv', index=False, encoding='utf-8-sig')
d.to_csv(str(STEM) + '_逐箱.csv', index=False, encoding='utf-8-sig')
L.to_csv(str(STEM) + '_航段.csv', index=False, encoding='utf-8-sig')
metadata = {'name': '问题二真正时间优先22架次',
            'metrics': [0.0, first[0] / 3600, float(f['能耗kWh'].sum()), len(f)],
            'last_delivery_s': second[1], 'minimum_hard_slack_s': third[2],
            'time_is_transport_all_return_s': first[0],
            'fixed_route_CP_SAT': STAGES,
            'not_global_optimality': True,
            'transport_origin_sha256': {
                kind: hashlib.sha256(Path(str(SOURCE) + '_' + kind + '.csv').read_bytes()).hexdigest()
                for kind in ['架次', '逐箱', '航段']},
            'relation_to_Q3': '仅复用其22条运输路线与实体分配；本方案重新排时、不使用中继时窗。问题三的通信及问题四的输入保持原验证时间表。'}
Path(str(STEM) + '.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'返航秒': first[0], '末箱秒': second[1], '最小硬余量秒': third[2],
                  '能耗kWh': metadata['metrics'][2], '固定路线四阶段': STAGES},
                 ensure_ascii=False), flush=True)
