"""Question 4: exact grouping of the FIXED Question-3 task/communication records.

No flight, relay sortie, start time, charging interval or communication assignment
is split, copied or rescheduled.  Only service-area groups and dedicated resources
are assigned.  Half-open intervals [start, end) permit reuse exactly at end.

Usage: python compute_q4.py [--source ../q3_v10_execution] [--output data]
The copied source tables in data/sources allow later independent reruns.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREFIX = '问题三_时间主方案'
SOURCE_FILES = {
    'flights.csv': PREFIX + '_运输_架次.csv',
    'deliveries.csv': PREFIX + '_运输_逐箱.csv',
    'legs.csv': PREFIX + '_运输_航段.csv',
    'relays.csv': PREFIX + '_中继架次.csv',
    'communication.csv': PREFIX + '_精确_通信提交记录.csv',
    'transport.xlsx': 'q1_flat/运输无人机数据.xlsx',
    'relay.xlsx': '中继无人机数据.xlsx',
    'nodes.xlsx': 'q1_flat/调度中心与服务区.xlsx',
    'boxes.xlsx': 'q1_flat/物资需求与配送时限.xlsx',
    'q3_validation.json': PREFIX + '_V3_独立核验.json',
}
RESOURCE_META = {
    'transport_drone_A': ('A型运输无人机', 'transport_drone', 'A'),
    'transport_drone_B': ('B型运输无人机', 'transport_drone', 'B'),
    'transport_drone_C': ('C型运输无人机', 'transport_drone', 'C'),
    'transport_battery_A': ('A型共享电池', 'transport_battery', 'A'),
    'transport_battery_B': ('B型共享电池', 'transport_battery', 'B'),
    'transport_battery_C': ('C型共享电池', 'transport_battery', 'C'),
    'relay_drone_R': ('中继无人机', 'relay_drone', 'R'),
    'relay_component_R': ('中继能源组件', 'relay_component', 'R'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(rows, target):
    pd.DataFrame(rows).to_csv(target, index=False, encoding='utf-8-sig')


def peak(intervals):
    """Exact interval clique number; end events precede starts at ties."""
    events = []
    for a, b in intervals:
        assert 0 <= a < b
        events.extend([(float(a), 1), (float(b), -1)])
    count = maximum = 0
    for _, delta in sorted(events):
        count += delta
        assert count >= 0
        maximum = max(maximum, count)
    assert count == 0
    return maximum


def color_intervals(rows):
    """Optimal interval coloring, with concrete independent within-group IDs."""
    active = []
    free = []
    n = 0
    for row in sorted(rows, key=lambda r: (r['start_s'], r['end_s'], r['task_id'])):
        while active and active[0][0] <= row['start_s']:
            _, color = heapq.heappop(active)
            heapq.heappush(free, color)
        if free:
            color = heapq.heappop(free)
        else:
            n += 1
            color = n
        heapq.heappush(active, (row['end_s'], color))
        row['assigned_id'] = f"{row['group_id']}-{row['resource_key']}-{color:02d}"
    assert n == peak((r['start_s'], r['end_s']) for r in rows)
    by_id = {}
    for row in rows:
        by_id.setdefault(row['assigned_id'], []).append(row)
    for values in by_id.values():
        order = sorted(values, key=lambda r: r['start_s'])
        assert all(a['end_s'] <= b['start_s'] for a, b in zip(order, order[1:]))
    return n


def partitions(items, k):
    """All unlabeled k-partitions through canonical restricted-growth strings."""
    if not items:
        return
    def rec(labels):
        if len(labels) == len(items):
            if max(labels) + 1 == k:
                yield [[items[i] for i, a in enumerate(labels) if a == g] for g in range(k)]
            return
        for value in range(min(max(labels) + 1, k - 1) + 1):
            yield from rec(labels + [value])
    yield from rec([0])


def cv(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=0) / values.mean()) if values.mean() else 0.0


def gini(values):
    """Population Gini of strictly nonnegative group totals."""
    x = np.sort(np.asarray(values, dtype=float))
    assert len(x) > 0 and np.all(x >= 0) and x.sum() > 0
    return float(np.sum((2*np.arange(1,len(x)+1)-len(x)-1)*x)/(len(x)*x.sum()))


def peak_witness(rows):
    """Independent midpoint count and a concrete positive-width peak witness."""
    if not rows:
        return dict(peak_start_s=0.0, peak_end_s=0.0, peak_tasks='')
    times = sorted({r[key] for r in rows for key in ('start_s','end_s')})
    best = (-1, 0.0, 0.0, [])
    for a, b in zip(times, times[1:]):
        mid = (a+b)/2
        active = sorted(r['task_id'] for r in rows if r['start_s']<=mid<r['end_s'])
        if len(active)>best[0]:
            best=(len(active),a,b,active)
    assert best[0] == peak((r['start_s'],r['end_s'])for r in rows)
    return dict(peak_start_s=best[1],peak_end_s=best[2],peak_tasks=','.join(best[3]))


def main(source: Path, output: Path):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    cached = output / 'sources'
    cached.mkdir(exist_ok=True)
    provenance = []
    for short, long in SOURCE_FILES.items():
        src = source / long if source != cached else cached / short
        if not src.is_file():
            raise FileNotFoundError(f'缺少输入文件：{src}')
        dst = cached / short
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        provenance.append(dict(local_file='sources/' + short, original_file=long,
                               sha256=digest(dst), bytes=dst.stat().st_size))
    F = pd.read_csv(cached/'flights.csv')
    D = pd.read_csv(cached/'deliveries.csv')
    L = pd.read_csv(cached/'legs.csv')
    R = pd.read_csv(cached/'relays.csv')
    C = pd.read_csv(cached/'communication.csv')
    B = pd.read_excel(cached/'boxes.xlsx', sheet_name='逐箱货箱清单').set_index('货箱编号')
    raw_n = pd.read_excel(cached/'nodes.xlsx', header=None)
    nodes = {}
    for r in raw_n.itertuples(index=False, name=None):
        if isinstance(r[0], str) and (r[0] == 'O01' or r[0].startswith('S0')):
            nodes[r[0]] = dict(site_id=r[0], longitude=float(r[2]), latitude=float(r[3]),
                              elevation_m=float(r[4]))
    sites = sorted(s for s in nodes if s != 'O01')
    assert len(F) == 22 and len(R) == 4 and len(D) == len(B) == 80 and len(sites) == 15
    assert D['货箱编号'].is_unique and set(D['货箱编号']) == set(B.index)
    raw_u = pd.read_excel(cached/'transport.xlsx', header=None)
    raw_r = pd.read_excel(cached/'relay.xlsx', header=None)
    U = {str(r[0]): str(r[1]) for r in raw_u.iloc[8:16].itertuples(index=False, name=None)}
    inventory = {f'transport_drone_{g}': sum(v == g for v in U.values()) for g in 'ABC'}
    inventory.update({f'transport_battery_{r[0]}': int(r[1]) for r in raw_u.iloc[19:22].itertuples(index=False, name=None)})
    relay_ids = [str(raw_r.iloc[i, 0]) for i in [6, 7]]
    inventory['relay_drone_R'] = len(relay_ids)
    inventory['relay_component_R'] = int(raw_r.iloc[11, 1])
    relay_turnaround = float(raw_r.iloc[2, 11])
    assert inventory == dict(transport_drone_A=4, transport_drone_B=2, transport_drone_C=2,
                            transport_battery_A=6, transport_battery_B=4, transport_battery_C=4,
                            relay_drone_R=2, relay_component_R=6)
    assert np.allclose(R['无人机可用秒'], R['返回秒'] + relay_turnaround)
    routes = {r['架次编号']: [s for s in r['访问顺序'].split('→') if s != 'O01'] for r in F.to_dict('records')}
    parent = {s: s for s in sites}
    def find(a):
        if parent[a] != a:
            parent[a] = find(parent[a])
        return parent[a]
    links = []
    def join(values, reason, task):
        values = sorted(set(values))
        for a, b in itertools.combinations(values, 2):
            ra, rb = find(a), find(b)
            parent[rb] = ra
            links.append(dict(site_a=a, site_b=b, relation=reason, task_id=task))
    for fid, visits in routes.items():
        join(visits, '同一运输架次', fid)
    transport_components = {}
    for site in sites:
        transport_components.setdefault(find(site), []).append(site)
    transport_components = sorted(transport_components.values(), key=lambda x: x[0])
    assert len(transport_components) == 10
    assert sorted(map(len, transport_components)) == [1]*7 + [2, 2, 4]
    transport_component_rows = [dict(block_id=f'T{i:02d}',site_id=site,
                                     block_size=len(block),sites='、'.join(block))
                                for i,block in enumerate(transport_components,1) for site in block]
    relay_routes = {}
    for jid in R['中继架次']:
        subset = C[(C['保障方式'] == '中继') & (C['中继架次编号'] == jid)]
        fids = sorted(set(subset['运输架次编号']))
        assert fids, f'中继架次 {jid} 未承担保障关系'
        relay_routes[jid] = fids
        join([site for fid in fids for site in routes[fid]], '同一中继架次保障', jid)
    comp_dict = {}
    for site in sites:
        comp_dict.setdefault(find(site), []).append(site)
    comps = sorted(comp_dict.values(), key=lambda x: x[0])
    comp_ids = {s: f'C{i+1}' for i, values in enumerate(comps) for s in values}
    assert sorted(map(len, comps)) == [1, 1, 13]
    component_rows = []
    for s in sites:
        d = D[D['服务区'] == s]
        component_rows.append(dict(component_id=comp_ids[s], site_id=s,
            component_size=len(comps[int(comp_ids[s][1:])-1]), n_boxes=len(d),
            mass_kg=float(B.loc[d['货箱编号'],'单箱质量（kg）'].sum()), **{k:v for k,v in nodes[s].items() if k!='site_id'}))
    base_intervals = []
    def interval(task, kind, model, start, end):
        key = f'{kind}_{model}'
        label, _, _ = RESOURCE_META[key]
        base_intervals.append(dict(task_id=task, resource_key=key, resource_label=label,
            resource_type=kind, model=model, start_s=float(start), end_s=float(end),
            duration_s=float(end-start)))
    for row in F.to_dict('records'):
        interval(row['架次编号'], 'transport_drone', row['机型'], row['开始秒'], row['返回秒'])
        interval(row['架次编号'], 'transport_battery', row['机型'], row['开始秒'], row['电池充满秒'])
    for row in R.to_dict('records'):
        interval(row['中继架次'], 'relay_drone', 'R', row['开始秒'], row['无人机可用秒'])
        interval(row['中继架次'], 'relay_component', 'R', row['开始秒'], row['组件充满秒'])
    audit_s = max(x['end_s'] for x in base_intervals) - min(x['start_s'] for x in base_intervals)
    assert audit_s > 0
    global_peaks = {key: peak((r['start_s'],r['end_s']) for r in base_intervals if r['resource_key']==key) for key in RESOURCE_META}
    all_scenarios, all_groups, all_resources, all_scenario_resources = [], [], [], []
    all_site_groups, all_intervals, all_task_assignments = [], [], []
    for k in (2, 3):
        candidates = list(partitions(comps, k))
        assert len(candidates) == {2:3, 3:1}[k]
        for idx, parts in enumerate(candidates, 1):
            sid = f'K{k}_{idx:02d}'
            group_sites = {f'G{i+1}': sorted(s for comp in part for s in comp) for i, part in enumerate(parts)}
            assign = {s:g for g, values in group_sites.items() for s in values}
            route_groups = {}
            for fid, visits in routes.items():
                selected = {assign[s] for s in visits}
                assert len(selected) == 1
                route_groups[fid] = selected.pop()
            relay_groups = {}
            for jid, fids in relay_routes.items():
                selected = {route_groups[f] for f in fids}
                assert len(selected) == 1
                relay_groups[jid] = selected.pop()
            task_groups = {**route_groups, **relay_groups}
            rows = [dict(scenario_id=sid, group_id=task_groups[row['task_id']], **row) for row in base_intervals]
            group_rows, resource_rows = [], []
            for gid, values in group_sites.items():
                for key, (label, rtype, model) in RESOURCE_META.items():
                    selected = [r for r in rows if r['group_id']==gid and r['resource_key']==key]
                    minimum = color_intervals(selected)
                    resource_rows.append(dict(scenario_id=sid,n_groups=k,group_id=gid,
                        resource_key=key,resource_label=label,resource_type=rtype,model=model,
                        required_count=minimum,inventory=inventory[key],global_peak=global_peaks[key],
                        occupancy_s=sum(r['duration_s'] for r in selected),audit_s=audit_s,
                        utilization=(sum(r['duration_s'] for r in selected)/(minimum*audit_s)
                                     if minimum else float('nan')),**peak_witness(selected)))
                gf = F[F['架次编号'].map(route_groups)==gid]
                gr = R[R['中继架次'].map(relay_groups)==gid]
                gd = D[D['架次编号'].map(route_groups)==gid]
                gl = L[L['架次编号'].isin(gf['架次编号'])]
                source_b = B.loc[gd['货箱编号']]
                occupancy = {rtype: sum(r['duration_s'] for r in rows if r['group_id']==gid and r['resource_type']==rtype)/3600 for rtype in ['transport_drone','transport_battery','relay_drone','relay_component']}
                group_rows.append(dict(scenario_id=sid,n_groups=k,group_id=gid,sites='、'.join(values),
                    n_sites=len(values),n_boxes=len(gd),mass_kg=float(source_b['单箱质量（kg）'].sum()),
                    volume_m3=float(source_b['单箱体积（m³）'].sum()),
                    transport_sorties=len(gf),relay_sorties=len(gr),flight_time_h=float(gl['飞行秒'].sum())/3600,
                    transport_occupancy_h=occupancy['transport_drone'],battery_occupancy_h=occupancy['transport_battery'],
                    relay_occupancy_h=occupancy['relay_drone'],component_occupancy_h=occupancy['relay_component'],
                    transport_workload_h=occupancy['transport_drone'],relay_workload_h=occupancy['relay_drone'],
                    aircraft_workload_h=occupancy['transport_drone']+occupancy['relay_drone'],
                    total_workload_h=sum(occupancy.values()),transport_energy_kwh=float(gf['能耗kWh'].sum()),
                    relay_energy_kwh=float(gr['能耗kWh'].sum()),total_energy_kwh=float(gf['能耗kWh'].sum()+gr['能耗kWh'].sum()),
                    final_delivery_s=float(gd['送达秒'].max()),transport_return_s=float(gf['返回秒'].max()),
                    allocated_resources=sum(r['required_count'] for r in resource_rows if r['group_id']==gid)))
                for s in values:
                    sd = gd[gd['服务区']==s]
                    all_site_groups.append(dict(scenario_id=sid,n_groups=k,group_id=gid,component_id=comp_ids[s],
                        **nodes[s],boxes=len(sd),mass_kg=float(B.loc[sd['货箱编号'],'单箱质量（kg）'].sum())))
            scenario_resources = []
            for key, (label, rtype, model) in RESOURCE_META.items():
                required = sum(r['required_count'] for r in resource_rows if r['resource_key']==key)
                scenario_resources.append(dict(scenario_id=sid,n_groups=k,resource_key=key,resource_label=label,
                    resource_type=rtype,model=model,sum_required=required,inventory=inventory[key],global_peak=global_peaks[key],
                    shortage=max(required-inventory[key],0),unused_inventory=max(inventory[key]-required,0),
                    partition_overhead=required-global_peaks[key],audit_s=audit_s,
                    utilization=(sum(r['occupancy_s'] for r in resource_rows if r['resource_key']==key)/(required*audit_s)
                                 if required else float('nan'))))
            shortage=sum(r['shortage'] for r in scenario_resources)
            all_scenarios.append(dict(scenario_id=sid,n_groups=k,
                partition_label='｜'.join('、'.join(v) for v in group_sites.values()),
                component_partition='｜'.join('+'.join(sorted({comp_ids[s] for s in v})) for v in group_sites.values()),
                shortage_count=shortage,total_required=sum(r['sum_required'] for r in scenario_resources),
                partition_overhead=sum(r['partition_overhead'] for r in scenario_resources),
                unused_inventory=sum(r['unused_inventory'] for r in scenario_resources),
                total_workload_h=sum(r['total_workload_h'] for r in group_rows),
                transport_workload_gini=gini([r['transport_workload_h'] for r in group_rows]),
                transport_workload_gini_normalized=gini([r['transport_workload_h'] for r in group_rows])/(1-1/k),
                relay_workload_gini=gini([r['relay_workload_h'] for r in group_rows]),
                relay_workload_gini_normalized=gini([r['relay_workload_h'] for r in group_rows])/(1-1/k),
                group_workload_gini=gini([r['total_workload_h'] for r in group_rows]),
                group_workload_gini_normalized=gini([r['total_workload_h'] for r in group_rows])/(1-1/k),
                group_box_count_gini=gini([r['n_boxes'] for r in group_rows]),
                workload_cv=cv([r['total_workload_h'] for r in group_rows]),
                aircraft_workload_cv=cv([r['aircraft_workload_h'] for r in group_rows]),
                mass_cv=cv([r['mass_kg'] for r in group_rows]),
                boxes_cv=cv([r['n_boxes'] for r in group_rows]),
                sites_cv=cv([r['n_sites'] for r in group_rows]),
                feasible_inventory=shortage==0,conditional_feasible=True))
            assert sum(r['n_boxes'] for r in group_rows)==80
            assert sum(r['transport_sorties'] for r in group_rows)==len(F)
            assert sum(r['relay_sorties'] for r in group_rows)==len(R)
            assert math.isclose(sum(r['total_energy_kwh'] for r in group_rows),F['能耗kWh'].sum()+R['能耗kWh'].sum())
            all_groups.extend(group_rows)
            all_resources.extend(resource_rows)
            all_scenario_resources.extend(scenario_resources)
            all_intervals.extend(rows)
            for tid, gid in task_groups.items():
                all_task_assignments.append(dict(scenario_id=sid,group_id=gid,task_id=tid,
                    task_type='transport' if tid in route_groups else 'relay'))
    recommendations = {}
    for k in (2, 3):
        subset = [r for r in all_scenarios if r['n_groups']==k]
        for item in subset:
            item['non_singleton_admissible']=all(g['n_sites']>=2 for g in all_groups
                                                   if g['scenario_id']==item['scenario_id'])
        admissible=[r for r in subset if r['non_singleton_admissible']]
        if not admissible:
            recommendations[k]=None
            continue
        chosen=min(admissible,key=lambda r:(r['shortage_count'],r['total_required'],
                                        r['transport_workload_gini'],r['workload_cv'],r['scenario_id']))
        recommendations[k]=chosen['scenario_id']
    for rows in [all_scenarios,all_groups,all_resources,all_scenario_resources,all_site_groups,all_intervals,all_task_assignments]:
        for r in rows:
            sid=r['scenario_id']
            r['recommended']=sid in recommendations.values()
    for name,rows in [('scenarios',all_scenarios),('group_summary',all_groups),('group_resources',all_resources),
                      ('scenario_resources',all_scenario_resources),('site_groups',all_site_groups),('intervals',all_intervals),
                      ('task_assignments',all_task_assignments),('components',component_rows),('must_link_edges',links)]:
        write_csv(rows, output/(name+'.csv'))
    write_csv([dict(resource_key=key,resource_label=RESOURCE_META[key][0],inventory=inventory[key],global_peak=global_peaks[key]) for key in RESOURCE_META],output/'inventory.csv')
    write_csv(transport_component_rows,output/'transport_components.csv')
    write_csv(list(nodes.values()),output/'nodes.csv')
    write_csv([dict(relay_task=jid,transport_task=fid)for jid,fids in relay_routes.items()for fid in fids],output/'communication_links.csv')
    exact_proof = {
        'fixed_q3_tasks':dict(transport_sorties=len(F),relay_sorties=len(R),boxes=len(D),service_areas=len(sites)),
        'transport_only_components':transport_components,'n_transport_components':len(transport_components),
        'must_link_components':comps,'n_components':len(comps),
        'partition_counts':{'2':3,'3':1},
        'enumeration':'将运输和中继超边作并查集收缩；对三个联合任务块规范增长串穷举全部无标签二分和三分，供独立核验CP-SAT结果。',
        'resource_minimality':'固定半开区间的冲突图为区间图；最大重叠数是资源数量下界；贪心区间着色恰好达到该下界。',
        'inventory_infeasibility':'按四个方案逐类型核算库存；是否可在现有库存内独立执行以实际输出 shortage_count 判断。',
        'scope':'仅证明当前固定问题三任务与通信保障关系下的分组及最小增配，不证明原问题三联合优化的全局最优。',
    }
    notes={
        'input_boundary':f'完整继承问题三{len(F)}个运输架次、{len(R)}个中继架次、逐箱时间及全部{len(C)}条通信提交记录；中继任务不复制、不切割、不重排。',
        'recommendation_rule':'先要求各组至少2个服务区；在满足该条件的严格方案内最小化分类库存缺口总件数，再比较专属配置和运输机占用工时Gini。三组无可推荐方案。',
        'transport_workload_gini':'各组运输机开始至返航占用工时的组间Gini，作为主要工作负担公平指标；另报中继机同口径占用工时。',
        'group_workload_gini':'运输机、电池、中继机、能源组件四种设备占用工时合计的补充性组间Gini；主要公平指标使用运输机占用工时Gini，不等于人口公平或逐箱送达时间变化。',
        'group_workload_gini_normalized':'原始Gini/(1-1/k)，使k组情景的理论上限均为1；比较不同组数时同时报告，分组推荐仍分别在固定k内进行。',
        'total_workload_h':'运输无人机(开始→返回)+运输电池(开始→充满)+中继无人机(开始→返回后300秒)+中继组件(开始→充满)占用秒数之和/3600；是设备时，不是自然历时或人工作业时间。',
        'aircraft_workload_h':'只累计运输机和中继机占用工时，不累计电池/组件。',
        'partition_overhead':'各组最小专属资源数之和−不分组的全局峰值；表示分组导致共享机会损失，非闲置时间。',
        'unused_inventory':'max(库存−总专属配置,0)，不同资源类型不可互相抵扣。',
        'shortage':'max(总专属配置−库存,0)，逐资源类型求和，未假定购买成本相同，仅报告件数。',
        'intervals':'全部资源占用用半开区间[start,end)，同一秒结束可立即分配给下一架次；ID由逐组区间着色重新生成，不沿用旧编号计数。',
        'utilization':'逐组逐类占用秒数/(该组该类最少专属设备数×统一核算时长)；统一时长覆盖全部设备返航、充电和中继周转；该组无该类任务时留空。',
        'inventory_status':'以逐类计算所得库存缺口为准；有缺口的方案仅在补足对应资源后可实施。',
    }
    qa=dict(status='PASS',computed_utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.perf_counter()-started,
        inventory=inventory,global_peaks=global_peaks,recommendations=recommendations,audit_s=audit_s,
        source_files=provenance,proof=exact_proof,definitions=notes,
        invariant_totals=dict(mass_kg=float(F['质量kg'].sum()),volume_m3=float(F['体积m3'].sum()),
            transport_energy_kwh=float(F['能耗kWh'].sum()),relay_energy_kwh=float(R['能耗kWh'].sum()),
            total_energy_kwh=float(F['能耗kWh'].sum()+R['能耗kWh'].sum()),
            transport_occupancy_h=sum(r['duration_s'] for r in base_intervals if r['resource_type']=='transport_drone')/3600,
            total_device_occupancy_h=sum(r['duration_s'] for r in base_intervals)/3600),
        checks=dict(all_partitions_enumerated=True,resource_coloring_reaches_peak=True,
                    all_tasks_preserved=True,all_communication_assignments_preserved=True,
                    all_boxes_preserved=True,no_rescheduling=True,no_relay_task_duplication=True,
                    feasible_partitions_existing_inventory=sum(r['feasible_inventory'] for r in all_scenarios)))
    (output/'qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    inventory_conclusion=('现有库存下两组、三组均不可行；以下为增配后条件方案。' if all(not r['feasible_inventory'] for r in all_scenarios) else '现有库存可行性逐方案见场景表；缺口方案需增配资源。')
    lines=['# 问题四：固定问题三任务的独立分组核算','',exact_proof['scope'],'',notes['input_boundary'],'',
           inventory_conclusion,'',
           '两组无单站候选：'+recommendations[2]+'；严格三组禁止单站时无可行方案。', '',
           '## 指标定义','']+[f'- **{k}**：{v}'for k,v in notes.items()]+['','## 完备性证据','',exact_proof['enumeration'],exact_proof['resource_minimality'],exact_proof['inventory_infeasibility']]
    (output/'method_and_result.md').write_text('\n'.join(lines),encoding='utf-8')
    print(pd.DataFrame(all_scenarios)[['scenario_id','component_partition','shortage_count','total_required','workload_cv','recommended','feasible_inventory']].to_string(index=False))
    print(f"全部分区穷举与逐组资源着色通过；输出：{output.resolve()}")
    return qa


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    default_source=Path(os.environ.get('Q3_SOURCE', HERE.parent/'q3_v10_execution'))
    if not default_source.exists():
        default_source=HERE/'data'/'sources'
    parser.add_argument('--source','--source-dir',dest='source',type=Path,default=default_source)
    parser.add_argument('--output',type=Path,default=HERE/'data')
    args=parser.parse_args()
    main(args.source.resolve(),args.output.resolve())
