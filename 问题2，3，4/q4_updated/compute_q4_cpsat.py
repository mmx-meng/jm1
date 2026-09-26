"""Solve the revised, independently validated Q3 hypergraph partition with CP-SAT.

The three contracted task blocks form the real decision set here.  All task
events and the B-reference pair times are inherited; no sortie is moved in
time.  The older independent exhaustive partition calculation is a checker.
For larger post-contraction instances the same CP-SAT model is the exact
resource/repair layer of the proposed NSGA-II--CP-SAT architecture.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from ortools.sat.python import cp_model
import ortools
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-dir',type=Path,default=ROOT/'data')
a=p.parse_args()
D=a.data_dir

def load(name):return pd.read_csv(D/(name+'.csv'))
def output(rows,name):pd.DataFrame(rows).to_csv(D/(name+'.csv'),index=False,encoding='utf-8-sig')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

blocks=load('components');site_groups=load('site_groups');I=load('intervals');R=load('scenario_resources')
S=load('scenarios').set_index('scenario_id');pair=load('spatial_reference_pairwise')
trade=load('scenario_spatial_tradeoff_B').set_index('scenario_id')
flights=load('sources/flights');relay_links=load('communication_links')
ids=sorted(blocks.component_id.unique());assert ids==['C1','C2','C3']
sites_by_block={b:set(blocks[blocks.component_id.eq(b)].site_id) for b in ids}
site_block={s:b for b in ids for s in sites_by_block[b]}
transport_edges={r['架次编号']:{s for s in r['访问顺序'].split('→') if s!='O01'}
                 for r in flights.to_dict('records')}
hyper=[]
for task,members in transport_edges.items():
    assert len({site_block[s] for s in members})==1
    hyper.append({'layer':'transport','task_id':task,'site_count':len(members),
                  'member_sites':'、'.join(sorted(members)),'source_transport_tasks':task,
                  'binding':'题目同一运输架次必须同组'})
for task,rows in relay_links.groupby('relay_task'):
    transport=sorted(rows.transport_task)
    members=set().union(*(transport_edges[r] for r in transport))
    assert len({site_block[s] for s in members})==1
    hyper.append({'layer':'relay','task_id':task,'site_count':len(members),
                  'member_sites':'、'.join(sorted(members)),
                  'source_transport_tasks':'、'.join(transport),
                  'binding':'本问约定完整中继任务不拆分、不复制、不跨组'})
assert len(hyper)==len(flights)+relay_links.relay_task.nunique()
output(hyper,'q4_hypergraph_edges')

# The unique 3-group partition identifies which fixed task belongs to each
# immutable block.  Original allocated physical equipment IDs are not used.
baseline=site_groups[site_groups.scenario_id.eq('K3_01')]
group_block={gid:rows.component_id.unique().item() for gid,rows in baseline.groupby('group_id')}
assert len(group_block)==3 and set(group_block.values())==set(ids)
source=I[I.scenario_id.eq('K3_01')].copy()
source['block_id']=source.group_id.map(group_block)
assert len(source)==2*(len(flights)+relay_links.relay_task.nunique())
assert source.task_id.nunique()==len(flights)+relay_links.relay_task.nunique()
assert source.groupby('task_id').block_id.nunique().max()==1
keys=R.resource_key.drop_duplicates().tolist();assert len(keys)==8
stock=R.groupby('resource_key').inventory.first().to_dict()
shared=R.groupby('resource_key').global_peak.first().to_dict()
assert set(stock)==set(keys) and all(int(stock[c])>0 for c in keys)

# Evaluate each half-open interval [start,end) on one point in every open
# segment between successive task start/return/charge/turnaround events.
boundaries=sorted(set(source.start_s)|set(source.end_s))
moments=[(lo+hi)/2 for lo,hi in zip(boundaries,boundaries[1:])]
assert moments and max(boundaries)==int(source.end_s.max())
active={}
for b,c,t in itertools.product(ids,keys,range(len(moments))):
    rows=source[source.block_id.eq(b)&source.resource_key.eq(c)]
    active[b,c,t]=int(((rows.start_s<=moments[t])&(moments[t]<rows.end_s)).sum())
assert all(max(sum(active[b,c,t]for b in ids)for t in range(len(moments)))==int(shared[c]) for c in keys)
work={b:int(source[source.block_id.eq(b)&source.resource_key.str.startswith('transport_drone')].duration_s.sum())
      for b in ids}
total_work=sum(work.values())
assert total_work==int(source[source.resource_key.str.startswith('transport_drone')].duration_s.sum())

P={(r.from_node,r.to_node):float(r.one_way_empty_B_s) for r in pair.itertuples()}
mass=blocks.set_index('site_id').mass_kg.to_dict()
def spatial_of(mask):
    members=sorted(set().union(*(sites_by_block[b] for b in mask)))
    return min(sum(mass[s]*(P[s,m]+P[m,s])/120 if s!=m else 0
                   for s in members) for m in members)
nonempty=[tuple(ids[j]for j in range(len(ids))if (mask>>j)&1) for mask in range(1,2**len(ids))]
spatial_cost={mask:int(round(spatial_of(mask)*1000)) for mask in nonempty}

results=[];solver_records=[]
for k in (2,3):
    M=cp_model.CpModel()
    z={(b,g):M.NewBoolVar(f'z_{b}_{g}') for b in ids for g in range(k)}
    for b in ids:M.Add(sum(z[b,g]for g in range(k))==1)
    for g in range(k):M.Add(sum(z[b,g]for b in ids)>=1)
    M.Add(z['C1',0]==1) # discard permutation of identical group labels
    if k==3:
        M.Add(z['C2',1]==1);M.Add(z['C3',2]==1)
    n={(g,c):M.NewIntVar(0,int(shared[c]),f'n_{g}_{c}') for g in range(k)for c in keys}
    for g,c in n:
        for t in range(len(moments)):
            terms=[active[b,c,t]*z[b,g]for b in ids if active[b,c,t]]
            if terms:M.Add(n[g,c]>=sum(terms))
    shortage={c:M.NewIntVar(0,k*int(shared[c]),f'shortage_{c}') for c in keys}
    for c in keys:M.Add(shortage[c]>=sum(n[g,c]for g in range(k))-int(stock[c]))
    total_shortage=M.NewIntVar(0,60,'total_shortage')
    M.Add(total_shortage==sum(shortage.values()))
    total_config=M.NewIntVar(0,60,'total_config')
    M.Add(total_config==sum(n.values()))
    # The number of seconds is an integer; normalized Gini is monotone in
    # the sum of unordered absolute group differences for a fixed k.
    W=[M.NewIntVar(0,total_work,f'transport_work_s_{g}') for g in range(k)]
    for g in range(k):M.Add(W[g]==sum(work[b]*z[b,g]for b in ids))
    diffs=[]
    for g,h in itertools.combinations(range(k),2):
        d=M.NewIntVar(0,total_work,f'diff_{g}_{h}')
        M.AddAbsEquality(d,W[g]-W[h]);diffs.append(d)
    fairness=M.NewIntVar(0,k*total_work,'transport_pair_abs_s')
    M.Add(fairness==sum(diffs))
    # Subset indicator is an exact table formulation of the best medoid cost
    # for the three immutable blocks, without inserting a new depot.
    selected={(g,mask):M.NewBoolVar(f'subset_{g}_{"_".join(mask)}')
              for g in range(k)for mask in nonempty}
    for g in range(k):
        M.Add(sum(selected[g,mask]for mask in nonempty)==1)
        for b in ids:M.Add(z[b,g]==sum(selected[g,mask]for mask in nonempty if b in mask))
    spatial=M.NewIntVar(0,sum(spatial_cost.values()),'J_kg_min_x1000')
    M.Add(spatial==sum(spatial_cost[mask]*selected[g,mask]
                       for g in range(k)for mask in nonempty))
    # 12 is the LCM of all eight resource stock denominators; this is
    # dimensionless normalization and emphatically not an equipment price.
    assert all(12%int(stock[c])==0 for c in keys)
    resource_ratio=M.NewIntVar(0,360,'resource_stock_ratio_x12')
    M.Add(resource_ratio==sum((12//int(stock[c]))*n[g,c]for g in range(k)for c in keys))

    stages=[('shortage',total_shortage),('configuration',total_config),
            ('resource_stock_ratio_x12',resource_ratio),
            ('transport_fairness',fairness),('spatial_compactness',spatial)]
    sv=None;objective_values={}
    for name,obj in stages:
        M.Minimize(obj)
        sv=cp_model.CpSolver();sv.parameters.num_search_workers=1
        sv.parameters.max_time_in_seconds=30
        status=sv.Solve(M)
        assert status==cp_model.OPTIMAL,(k,name,sv.StatusName(status))
        optimum=int(sv.Value(obj));objective_values[name]=optimum
        M.Add(obj==optimum)
        solver_records.append({'k':k,'objective':name,'status':'OPTIMAL','value':optimum,
                               'best_objective_bound':sv.BestObjectiveBound(),
                               'branches':sv.NumBranches(),'wall_time_s':sv.WallTime()})
    assignment={b:'G'+str(next(g for g in range(k)if sv.Value(z[b,g]))+1) for b in ids}
    chosen=next((sid for sid,r in S[S.n_groups.eq(k)].iterrows()
        if {b:rows.group_id.unique().item() for b,rows in
            site_groups[site_groups.scenario_id.eq(sid)].groupby('component_id')}==assignment),None)
    assert chosen is not None,(k,assignment)
    check=S.loc[chosen]
    assert sv.Value(total_shortage)==int(check.shortage_count)
    assert sv.Value(total_config)==int(check.total_required)
    assert abs(sv.Value(fairness)/((k-1)*total_work)-float(check.transport_workload_gini_normalized))<1e-12
    assert abs(sv.Value(spatial)/1000-float(trade.loc[chosen,'spatial_compactness_kg_min']))<.002
    for c in keys:
        derived=sum(sv.Value(n[g,c])for g in range(k))
        reported=int(R[R.scenario_id.eq(chosen)&R.resource_key.eq(c)].sum_required.iloc[0])
        assert derived==reported,(k,c,derived,reported)
    results.append({'n_groups':k,'scenario_id':chosen,'block_assignment':assignment,
                    'shortage_count':sv.Value(total_shortage),
                    'total_required':sv.Value(total_config),
                    'resource_stock_ratio':sv.Value(resource_ratio)/12,
                    'transport_workload_gini_normalized':sv.Value(fairness)/((k-1)*total_work),
                    'spatial_compactness_kg_min':sv.Value(spatial)/1000,
                    'lexicographic_objectives':objective_values,
                    'all_stages_proved_OPTIMAL':True})

largest=max(work.values())
lower=[]
for k in (2,3):
    bound=max(0.,(k*largest/total_work-1)/(k-1))
    subset=S[S.n_groups.eq(k)]
    actual=float(subset.transport_workload_gini_normalized.min())
    assert actual>=bound-1e-12
    achieved_sid=subset.transport_workload_gini_normalized.idxmin()
    lower.append({'n_groups':k,'dominant_block':max(work,key=work.get),
       'dominant_block_transport_h':largest/3600,'total_transport_h':total_work/3600,
       'dominant_workload_share':largest/total_work,
       'relaxation_gini_lower_bound':bound,
       'best_achievable_exact_gini':actual,'best_scenario_id':achieved_sid,
       'discrete_partition_gap':actual-bound,
       'bound_assumption':'其他任务块可任意均分；下界可能严格小于实际可达值'})
output(lower,'q4_fairness_lower_bound')
output([{'component_id':b,'transport_workload_s':work[b],
         'transport_workload_h':work[b]/3600.,'site_count':len(sites_by_block[b]),
         'member_sites':'、'.join(sorted(sites_by_block[b]))}for b in ids],
       'q4_hypergraph_blocks')
output([{k:v for k,v in r.items()if not isinstance(v,dict)}for r in results],
       'q4_cpsat_solution')
certificate={'status':'PASS','ortools_version':ortools.__version__,
 'hypergraph_edges':len(hyper),'contracted_components':len(ids),
 'transport_only_components':11,'event_segments':len(moments),
 'shared_task_ids':source.task_id.nunique(),
 'NSGA_II_used_on_this_three_block_instance':False,
 'CP_SAT_is_actual_partition_and_resource_optimizer':True,
 'enumeration_used_as_independent_validation':True,
 'no_fixed_task_start_or_end_optimized':True,
 'objectives':['eight-type resource shortages then required quantities',
   'normalized transport work Gini via exact absolute differences',
   'B-reference mass-weighted within-group medoid impedance'],
 'result':results,'stages':solver_records,'fairness_lower_bound':lower,
 'source_hashes':{n:digest(D/(n+'.csv'))for n in
   ['intervals','components','spatial_reference_pairwise','scenarios']}}
(D/'q4_cpsat_certificate.json').write_text(json.dumps(certificate,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','CP_SAT_partitions':results,
    'fairness_lower_bound':lower},ensure_ascii=False,indent=2))
