"""Independent Q4 rescue groups with explicit group-dedicated relay templates.

The 22 transport tasks, starts, routes, deliveries and relay locations/windows
remain fixed. If two groups use the same original relay task, the complete
relay sortie is physically copied once per group. This is an extra assumption,
separate from the strict no-duplication feasibility check.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from ortools.sat.python import cp_model
import numpy as np
import pandas as pd

from compute_q4 import RESOURCE_META, color_intervals, gini, peak, write_csv

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'data'
OUT = ROOT / 'no_singletons'
OUT.mkdir(exist_ok=True)
tc = pd.read_csv(BASE/'transport_components.csv')
f = pd.read_csv(BASE/'sources/flights.csv')
r = pd.read_csv(BASE/'sources/relays.csv')
d = pd.read_csv(BASE/'sources/deliveries.csv')
links = pd.read_csv(BASE/'communication_links.csv')
raw = pd.read_csv(BASE/'intervals.csv').drop_duplicates(['task_id','resource_key'])
inv = pd.read_csv(BASE/'inventory.csv').set_index('resource_key')
pair = pd.read_csv(BASE/'spatial_reference_pairwise.csv')
nodes = pd.read_csv(BASE/'nodes.csv')
ids = sorted(tc.block_id.unique())
assert len(ids) == 10 and len(f) == 22 and len(r) == 4 and len(d) == 80
block_sites = [set(tc[tc.block_id.eq(i)].site_id) for i in ids]
all_sites = sorted(set().union(*block_sites))
routes = {row['架次编号']:set(row['访问顺序'].split('→'))-{'O01'} for row in f.to_dict('records')}
fid_block = {fid: next(i for i,s in enumerate(block_sites) if visits <= s)
             for fid,visits in routes.items()}
assert len(fid_block) == len(routes)
relay_routes = {jid:set(z.transport_task) for jid,z in links.groupby('relay_task')}
stock = inv.inventory.to_dict()
keys = list(RESOURCE_META)
interval = {(row.task_id,row.resource_key):(float(row.start_s),float(row.end_s))
            for row in raw.itertuples(index=False)}
assert len(interval) == (len(f)+len(r))*2
duration = {fid:float(row['返回秒']-row['开始秒'])
            for fid,row in f.set_index('架次编号').iterrows()}
full_transport = sum(duration.values())
box_count = d.groupby('服务区').size().to_dict()
site_dist = {(z.from_node,z.to_node):float(z.one_way_empty_B_s)
             for z in pair.itertuples(index=False)}
for a,b in itertools.permutations(all_sites,2):
    assert (a,b) in site_dist

def mask_info(mask):
    sites = sorted(set().union(*(block_sites[b] for b in range(10) if mask & (1<<b))))
    fs = sorted(fid for fid,b in fid_block.items() if mask & (1<<b))
    js = sorted(jid for jid,covered in relay_routes.items() if covered.intersection(fs))
    resources = {key:peak(interval[tid,key] for tid in (fs+js)
                          if (tid,key) in interval) for key in keys}
    work = round(sum(duration[fid] for fid in fs))
    # The medoid is a service area used solely as a spatial statistic.
    medoid = min((sum(box_count[s]*(site_dist[s,m]+site_dist[m,s])/120
                     for s in sites if s!=m),m) for m in sites)
    return dict(mask=mask,sites=sites,flights=fs,relays=js,
                resources=resources,work=work,spatial=medoid[0],medoid=medoid[1],
                nboxes=sum(box_count[s] for s in sites))

all_masks = {mask:mask_info(mask) for mask in range(1,1<<10)}
assert all_masks[1023]['resources'] == inv.global_peak.to_dict()

def score_partition(masks):
    info = [all_masks[mask] for mask in masks]
    required = {key:sum(x['resources'][key] for x in info) for key in keys}
    shortage = sum(max(required[key]-stock[key],0) for key in keys)
    work = [x['work'] for x in info]
    fairness = gini(work)/(1-1/len(work))
    return dict(masks=list(masks),shortage=shortage,required=required,
                fairness=fairness,spatial=sum(x['spatial'] for x in info),
                group_sites=[len(x['sites']) for x in info],work=work,
                relay_sorties=sum(len(x['relays']) for x in info))

def enumerated(k, min_sites, max_sites):
    """Independent complete verifier for CP-SAT's group subset decisions."""
    out=[]
    def rec(rest,groups):
        if len(groups)==k-1:
            if min_sites <= len(all_masks[rest]['sites']) <= max_sites:
                out.append(score_partition(groups+[rest]))
            return
        first = rest & -rest
        sub = rest
        while sub:
            if sub & first and sub != rest and min_sites<=len(all_masks[sub]['sites'])<=max_sites:
                rec(rest^sub,groups+[sub])
            sub=(sub-1)&rest
    rec((1<<10)-1,[])
    return out

def solve(k, min_sites, max_sites, fairness_cap):
    candidates = [mask for mask,x in all_masks.items()
                  if min_sites<=len(x['sites'])<=max_sites]
    M = cp_model.CpModel()
    x = {(g,mask):M.NewBoolVar(f'g{g}_{mask}') for g in range(k) for mask in candidates}
    for g in range(k):M.Add(sum(x[g,mask] for mask in candidates)==1)
    for b in range(10):M.Add(sum(x[g,mask] for g in range(k) for mask in candidates
                                 if mask&(1<<b))==1)
    M.Add(sum(x[0,mask] for mask in candidates if mask&1)==1)
    # All three group permutations are equivalent; canonical smallest block.
    if k==3:
        first = {}
        for g in (1,2):
            first[g]=M.NewIntVar(1,9,f'first_{g}')
            M.Add(first[g]==sum((mask&-mask).bit_length()*x[g,mask]
                                 for mask in candidates))
        M.Add(first[1]<first[2])
    config={}
    shortages={}
    for key in keys:
        config[key]=M.NewIntVar(0,40,f'config_{key}')
        M.Add(config[key]==sum(all_masks[mask]['resources'][key]*x[g,mask]
                               for g in range(k) for mask in candidates))
        shortages[key]=M.NewIntVar(0,40,f'shortage_{key}')
        M.AddMaxEquality(shortages[key],[0,config[key]-stock[key]])
    total_short=M.NewIntVar(0,60,'total_shortage')
    M.Add(total_short==sum(shortages.values()))
    W=[]
    for g in range(k):
        w=M.NewIntVar(0,round(full_transport),f'work_{g}')
        M.Add(w==sum(all_masks[mask]['work']*x[g,mask]for mask in candidates))
        W.append(w)
    differences=[]
    for g,h in itertools.combinations(range(k),2):
        v=M.NewIntVar(0,round(full_transport),f'abs_{g}_{h}')
        M.AddAbsEquality(v,W[g]-W[h]); differences.append(v)
    total_diff=M.NewIntVar(0,k*round(full_transport),'fairness_numerator')
    M.Add(total_diff==sum(differences))
    # For K groups, normalized Gini = sum of pairwise differences / ((K-1)*total work).
    M.Add(total_diff <= int(fairness_cap*(k-1)*round(full_transport)))
    spatial=M.NewIntVar(0,100000,'medoid_box_minutes')
    M.Add(spatial==sum(round(all_masks[mask]['spatial'])*x[g,mask]
                       for g in range(k) for mask in candidates))
    config_total=M.NewIntVar(0,100,'configuration_total')
    M.Add(config_total==sum(config.values()))
    # Explicit rescue trade-off: subject to workload fairness, reduce shortage;
    # spatial compactness and resource total resolve ties in this order.
    M.Minimize(total_short*10_000_000+spatial*100+total_diff+config_total)
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=110
    solver.parameters.num_search_workers=8
    status=solver.Solve(M)
    assert status==cp_model.OPTIMAL,(k,solver.StatusName(status),solver.BestObjectiveBound())
    masks = [next(mask for mask in candidates if solver.Value(x[g,mask]))for g in range(k)]
    exact = score_partition(masks)
    assert exact['shortage']==solver.Value(total_short)
    assert abs(exact['fairness']-solver.Value(total_diff)/((k-1)*round(full_transport)))<1e-9
    baseline = enumerated(k,min_sites,max_sites)
    assert len(baseline)>0 and any(z['masks']==masks for z in baseline)
    eligible=[z for z in baseline if z['fairness']<=fairness_cap+1e-9]
    min_short=min(z['shortage'] for z in eligible)
    assert exact['shortage']==min_short,(k,min_short,exact)
    def objective(z):
        pairdiff=sum(abs(a-b) for a,b in itertools.combinations(z['work'],2))
        medoid=sum(round(all_masks[mask]['spatial']) for mask in z['masks'])
        return z['shortage']*10_000_000+medoid*100+pairdiff+sum(z['required'].values())
    assert objective(exact)==min(map(objective,eligible)), (k,'CP-SAT objective mismatch')
    exact.update(k=k,solver_status='OPTIMAL',min_sites=min_sites,max_sites=max_sites,
                 fairness_cap=fairness_cap,candidates_checked=len(baseline),
                 status='增配中继副本的条件方案')
    return exact

solutions = [solve(2,6,9,.33),solve(3,4,6,.34)]
rows=[];gr=[];sr=[];site_rows=[];int_rows=[];task_rows=[]
for sol in solutions:
    k=sol['k'];sid=f'R{k}_不设单站组'
    sol['scenario_id']=sid
    relays_by_group={}
    group_renames={}
    for g,mask in enumerate(sol['masks'],1):
        gid=f'G{g}'; info=all_masks[mask]
        relays_by_group[gid]=info['relays']
        for site in info['sites']:
            z=nodes[nodes.site_id.eq(site)].iloc[0].to_dict()
            site_rows.append(dict(scenario_id=sid,n_groups=k,group_id=gid,site_id=site,
                                  **{key:value for key,value in z.items() if key!='site_id'},
                                  n_boxes=box_count[site]))
        for fid in info['flights']:
            task_rows.append(dict(scenario_id=sid,group_id=gid,task_id=fid,
                                  source_task_id=fid,task_type='transport',copied=False))
        for jid in info['relays']:
            task_rows.append(dict(scenario_id=sid,group_id=gid,task_id=f'{jid}@{gid}',
                                  source_task_id=jid,task_type='relay',copied=True))
        this = []
        for task in task_rows:
            if task['scenario_id']!=sid or task['group_id']!=gid:continue
            for key in keys:
                source=(task['source_task_id'],key)
                if source not in interval:continue
                start,end=interval[source]
                label,rtype,model=RESOURCE_META[key]
                this.append(dict(scenario_id=sid,group_id=gid,task_id=task['task_id'],
                                 source_task_id=task['source_task_id'],resource_key=key,
                                 resource_label=label,resource_type=rtype,model=model,
                                 start_s=start,end_s=end,duration_s=end-start))
        for key in keys:
            sub=[z for z in this if z['resource_key']==key]
            req=color_intervals(sub)
            assert req==info['resources'][key]
            gr.append(dict(scenario_id=sid,n_groups=k,group_id=gid,resource_key=key,
                           required_count=req,inventory=stock[key],
                           utilization=sum(z['duration_s'] for z in sub)/
                              (req*float(raw.end_s.max())) if req else None))
        int_rows.extend(this)
        rselect=r[r['中继架次'].isin(info['relays'])]
        gr_work=sum(z['duration_s']for z in this if z['resource_type']=='relay_drone')/3600
        tr_work=info['work']/3600
        gr.append(dict(scenario_id=sid,n_groups=k,group_id=gid,resource_key='SUMMARY',
                       required_count=None,inventory=None,utilization=None))
        rows.append(dict(scenario_id=sid,n_groups=k,group_id=gid,
                         sites='、'.join(info['sites']),n_sites=len(info['sites']),
                         n_boxes=info['nboxes'],transport_sorties=len(info['flights']),
                         relay_sorties=len(info['relays']),transport_workload_h=tr_work,
                         relay_workload_h=gr_work,transport_energy_kwh=float(f[f['架次编号'].isin(info['flights'])]['能耗kWh'].sum()),
                         relay_energy_kwh=float(rselect['能耗kWh'].sum()),
                         spatial_medoid=info['medoid'],spatial_box_min=info['spatial']))
    for key in keys:
        required=sum(z['required_count'] for z in gr if z['scenario_id']==sid and z['resource_key']==key)
        sr.append(dict(scenario_id=sid,n_groups=k,resource_key=key,inventory=stock[key],
                       global_peak=int(inv.loc[key,'global_peak']),sum_required=required,
                       shortage=max(required-stock[key],0)))
    sol['relay_copy_count']=sum(len(z)for z in relays_by_group.values())-len(r)
    sol['relay_sorties_total']=sum(len(z)for z in relays_by_group.values())
    sol['energy_kwh']=sum(z['transport_energy_kwh']+z['relay_energy_kwh']
                          for z in rows if z['scenario_id']==sid)
    sol['groups']=[z for z in rows if z['scenario_id']==sid]
    assert sum(z['shortage'] for z in sr if z['scenario_id']==sid)==sol['shortage']
    assert sum(z['n_boxes'] for z in rows if z['scenario_id']==sid)==80
    assert sum(z['transport_sorties'] for z in rows if z['scenario_id']==sid)==22
    owner={z['task_id']:z['group_id']for z in task_rows
           if z['scenario_id']==sid and z['task_type']=='transport'}
    copied={(z['group_id'],z['source_task_id'])for z in task_rows
            if z['scenario_id']==sid and z['task_type']=='relay'}
    assert all((owner[z.transport_task],z.relay_task) in copied
               for z in links.itertuples(index=False))
    assert len(set(t['task_id'] for t in task_rows if t['scenario_id']==sid))==22+sol['relay_sorties_total']

write_csv(rows,OUT/'group_summary.csv')
write_csv([z for z in gr if z['resource_key']!='SUMMARY'],OUT/'group_resources.csv')
write_csv(sr,OUT/'scenario_resources.csv')
write_csv(site_rows,OUT/'site_groups.csv')
write_csv(int_rows,OUT/'intervals.csv')
write_csv(task_rows,OUT/'task_assignments.csv')
write_csv([{key:value for key,value in z.items() if key!='groups'} for z in solutions],OUT/'scenarios.csv')
report=dict(assumption='各组独立复制所需的完整中继服务模板；中继位置、起止时刻、运输访问、逐箱送达均保持原样；复制中继任务导致能耗和资源需求增加。',
            base_q3=dict(transport_sorties=22,relay_sorties=4,
                         transport_plus_relay_energy_kwh=float(f['能耗kWh'].sum()+r['能耗kWh'].sum()),
                         last_delivery_s=float(d['送达秒'].max()),
                         last_joint_return_s=float(max(f['返回秒'].max(),r['返回秒'].max()))),
            strict_model=dict(block_sizes=[1,13,1],
                strict_two_group_without_single_site='K2_02：S001与S006同组，其余13区同组；分别21、59箱，库存缺10件。',
                three_groups_minimum_two_sites='INFEASIBLE',
                reason='13区任务块不能拆，余下两个块各仅1区；三组必有两个单区。'),
            extended=solutions,
            checks=dict(cp_sat_optimal=True,independent_enumeration_crosscheck=True,
                        original_transport_tasks_unchanged=True,
                        all_original_communication_links_within_copied_group=True,
                        no_single_site_group=True,relay_copies_explicit=True,
                        missing_inventory_not_hard_constrained=True))
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
for z in solutions:
    print(z['scenario_id'],'站点',z['group_sites'],'货箱',[a['n_boxes'] for a in z['groups']],
          '任务',z['relay_sorties_total'],'缺口',z['shortage'],
          'Gini',round(z['fairness'],4),'能耗',round(z['energy_kwh'],4),
          '完整穷举',z['candidates_checked'])
