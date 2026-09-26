"""Fixed-task spatial diagnostics using the corrected DEM and Q3 physics.

The B model is one common *empty-flight reference* for geographic comparison.
No alternative transport mission, depot, delivery time or relay is optimized.
The reference time is not the frozen Q3 box-delivery time.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--data-dir',type=Path,default=ROOT/'data')
ap.add_argument('--q3-dir',type=Path,default=ROOT/'q3_source')
a=ap.parse_args()
sys.path.insert(0,str(a.q3_dir))
from 问题二_物理模型与输入 import read_inputs, pairwise_legs
from q1_flat.精确算法 import utm49, read_dem, touched_cells

def hashfile(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(rows,name):
    pd.DataFrame(rows).to_csv(a.data_dir/name,index=False,encoding='utf-8-sig')

verified={'flights.csv':'_运输_架次.csv','legs.csv':'_运输_航段.csv',
          'deliveries.csv':'_运输_逐箱.csv','relays.csv':'_中继架次.csv',
          'communication.csv':'_精确_通信提交记录.csv'}
source_hash={}
for name,suffix in verified.items():
    target=a.q3_dir/('问题三_时间主方案'+suffix)
    assert hashfile(a.data_dir/'sources'/name)==hashfile(target),name
    source_hash[name]=hashfile(target)

nodes,boxes,models,_,_=read_inputs()
model=models.loc['B']
assert (float(model.cruise_speed_m_s),float(model.climb_speed_m_s),
        float(model.descent_speed_m_s))==(15.,3.,2.5)
dem_file=a.q3_dir/'q1_flat'/'最终工作DEM.tif'
dem,left,top,dx,dy=read_dem(dem_file)
assert 0.0001<dx<0.0005 and 0.0001<dy<0.0005
pair=pairwise_legs(nodes)
assert len(pair)==240
flights=pd.read_csv(a.data_dir/'sources'/'flights.csv')
delivery=pd.read_csv(a.data_dir/'sources'/'deliveries.csv')
groups=pd.read_csv(a.data_dir/'site_groups.csv')
summaries=pd.read_csv(a.data_dir/'group_summary.csv')
resources=pd.read_csv(a.data_dir/'scenario_resources.csv')
group_resources=pd.read_csv(a.data_dir/'group_resources.csv')
scenarios=pd.read_csv(a.data_dir/'scenarios.csv').set_index('scenario_id')
components=pd.read_csv(a.data_dir/'components.csv').set_index('site_id')
hexes=pd.read_csv(a.data_dir/'hex_distance_cells.csv')
assert len(flights)==22 and len(delivery)==80 and len(hexes)>=120
audit_s=float(group_resources.audit_s.unique().item())
assert audit_s>=float(group_resources.peak_end_s.max())

def one_way(leg):
    return (float(leg['distance_m'])/float(model.cruise_speed_m_s)
            +float(leg['climb_m'])/float(model.climb_speed_m_s)
            +float(leg['descent_m'])/float(model.descent_speed_m_s))

site_ids=sorted(components.index)
reference=[]
for u,v in itertools.permutations(nodes.index,2):
    detail=pair[u,v]
    reference.append({'from_node':u,'to_node':v,'horizontal_m':detail['distance_m'],
                      'max_ground_m':detail['peak_m'],
                      'cruise_altitude_m':detail['cruise_altitude_m'],
                      'one_way_empty_B_s':one_way(detail),
                      'reference_only':True})
save(reference,'spatial_reference_pairwise.csv')

# Hex centers are sampling locations, not homes or new transport tasks.  Use
# DEM cell height + the existing 30 m service-point operation convention.
o=nodes.loc['O01']
x0=(float(o.longitude_deg)-left)/dx
y0=(top-float(o.latitude_deg))/dy
base_xy=utm49(float(o.longitude_deg),float(o.latitude_deg))
hexrows=[]
for r in hexes.itertuples(index=False):
    x1=(float(r.center_lon)-left)/dx
    y1=(top-float(r.center_lat))/dy
    cr=int(math.floor(x1));rr=int(math.floor(y1))
    assert 0<=cr<dem.shape[1] and 0<=rr<dem.shape[0], r.hex_id
    cells=touched_cells(x0,y0,x1,y1,dem.shape[1],dem.shape[0])
    assert cells and (rr,cr) in cells
    ground=float(dem[rr,cr]);top_z=max(float(dem[ri,ci])for ri,ci in cells)
    H=top_z+50.;end_z=ground+30.
    horizontal_m=math.dist(base_xy,utm49(float(r.center_lon),float(r.center_lat)))
    t=horizontal_m/float(model.cruise_speed_m_s)+max(0,H-float(o.operating_altitude_m))/float(model.climb_speed_m_s)+max(0,H-end_z)/float(model.descent_speed_m_s)
    hexrows.append({'hex_id':r.hex_id,'center_x_km':r.center_x_km,'center_y_km':r.center_y_km,
        'center_lon':r.center_lon,'center_lat':r.center_lat,
        'nearest_site_id':r.nearest_site_id,
        'group_K2_03':groups[(groups.scenario_id=='K2_03')&(groups.site_id==r.nearest_site_id)].group_id.iloc[0],
        'group_K3_01':groups[(groups.scenario_id=='K3_01')&(groups.site_id==r.nearest_site_id)].group_id.iloc[0],
        'ground_m':ground,'highest_along_line_m':top_z,'cruise_altitude_m':H,
        'horizontal_km':horizontal_m/1000.,'B_one_way_empty_min':t/60.,
        'reference_only':True})
save(hexrows,'hex_terrain_time_B.csv')

# Attribute fixed multi-site sortie occupancy by delivered cargo mass solely
# for the site-burden diagnostic. Never count one sortie twice across sites.
delivery['mass_kg']=delivery['货箱编号'].map(boxes.mass_kg)
assert delivery.mass_kg.notna().all()
shares=delivery.groupby(['架次编号','服务区'],as_index=False).mass_kg.sum()
flight_lookup=flights.set_index('架次编号')
shares['fraction']=shares.apply(lambda r:r.mass_kg/float(flight_lookup.loc[r['架次编号'],'质量kg']),axis=1)
assert np.allclose(shares.groupby('架次编号').fraction.sum(),1.)
shares['sortie_workload_s']=shares.apply(lambda r:r.fraction*(float(flight_lookup.loc[r['架次编号'],'返回秒'])-
                                        float(flight_lookup.loc[r['架次编号'],'开始秒'])),axis=1)
assert math.isclose(shares.sortie_workload_s.sum(),
    (flights['返回秒']-flights['开始秒']).sum(),rel_tol=1e-12)
site_work=shares.groupby('服务区').sortie_workload_s.sum()
deadlines=boxes.hard_deadline_s.reindex(delivery['货箱编号']).to_numpy(float)
delivery['hard_slack_s']=deadlines-delivery['送达秒'].to_numpy(float)
assert (delivery.hard_slack_s>=-1e-9).all()
site_rows=[]
for sid in site_ids:
    local=delivery[delivery['服务区'].eq(sid)]
    finite=local.hard_slack_s[np.isfinite(local.hard_slack_s)]
    site_rows.append({'site_id':sid,'component_id':components.loc[sid,'component_id'],
       'n_boxes':len(local),'mass_kg':float(local.mass_kg.sum()),
       'O01_B_one_way_empty_min':one_way(pair['O01',sid])/60.,
       'O01_distance_km':pair['O01',sid]['distance_m']/1000.,
       'fixed_transport_workload_allocated_h':site_work[sid]/3600.,
       'first_box_min_FIXED':float(local['送达秒'].min())/60.,
       'mean_box_delivery_min_FIXED':float(local['送达秒'].mean())/60.,
       'min_hard_deadline_slack_min_FIXED':float(finite.min())/60. if len(finite) else float('nan'),
       'hard_deadline_boxes':int(len(finite))})
save(site_rows,'site_terrain_workload_B.csv')

# Use mean directed B empty-flight time between service points as a symmetric
# terrain impedance. The medoid is diagnostic and never a new dispatch centre.
sym={}
for u,v in itertools.combinations(site_ids,2):
    sym[u,v]=sym[v,u]=(one_way(pair[u,v])+one_way(pair[v,u]))/120.  # minutes

group_rows=[]
for s in scenarios.itertuples():
    membership=groups[groups.scenario_id.eq(s.Index)]
    for gid,grp in membership.groupby('group_id'):
        members=grp.site_id.tolist()
        weights=dict(zip(grp.site_id,grp.mass_kg))
        candidates={m:sum(weights[i]*(sym[i,m] if i!=m else 0.) for i in members)
                    for m in members}
        medoid=min(candidates,key=lambda m:(candidates[m],m))
        gsum=summaries[(summaries.scenario_id.eq(s.Index))&
                       (summaries.group_id.eq(gid))].iloc[0]
        mean_direct=np.average([one_way(pair['O01',i])/60. for i in members],
                                weights=[weights[i] for i in members])
        group_rows.append({'scenario_id':s.Index,'n_groups':s.n_groups,'group_id':gid,
            'n_sites':len(members),'n_boxes':int(gsum.n_boxes),'mass_kg':float(sum(weights.values())),
            'diagnostic_medoid':medoid,'J_group_kg_min':candidates[medoid],
            'J_group_per_kg_min':candidates[medoid]/sum(weights.values()),
            'mean_O01_B_reference_min':mean_direct,
            'transport_workload_h_FIXED':float(gsum.transport_workload_h),
            'relay_workload_h_FIXED':float(gsum.relay_workload_h),
            'group_utilization_audit_s':audit_s})
save(group_rows,'group_spatial_compactness_B.csv')

spatial={x:sum(r['J_group_kg_min']for r in group_rows if r['scenario_id']==x)
         for x in scenarios.index}
resource_footprint={x:sum(float(r.sum_required)/float(r.inventory)
        for r in resources[resources.scenario_id.eq(x)].itertuples()) for x in scenarios.index}
assert all(v>=0 for v in spatial.values())
pareto={}
for sid,row in scenarios.iterrows():
    v=(resource_footprint[sid],float(row.transport_workload_gini_normalized),spatial[sid])
    pareto[sid]=not any(
       all(x<=y+1e-10 for x,y in zip((resource_footprint[other],
                    float(scenarios.loc[other,'transport_workload_gini_normalized']),spatial[other]),v))
       and any(x<y-1e-10 for x,y in zip((resource_footprint[other],
                    float(scenarios.loc[other,'transport_workload_gini_normalized']),spatial[other]),v))
       for other in scenarios.index if other!=sid)
trade=[]
for sid,row in scenarios.iterrows():
    trade.append({'scenario_id':sid,'n_groups':int(row.n_groups),
       'component_partition':row.component_partition,'total_required':int(row.total_required),
       'shortage_count':int(row.shortage_count),
       'resource_inventory_ratio_sum':resource_footprint[sid],
       'transport_workload_gini_normalized':float(row.transport_workload_gini_normalized),
       'relay_workload_gini_normalized':float(row.relay_workload_gini_normalized),
       'spatial_compactness_kg_min':spatial[sid],
       'spatial_compactness_per_total_kg_min':spatial[sid]/float(components.mass_kg.sum()),
       'pareto_in_three_diagnostics':pareto[sid],
       'recommended_resource_first':bool(row.recommended)})
save(trade,'scenario_spatial_tradeoff_B.csv')

qa={'status':'PASS','q3_scenario':'22 transport + 4 relay; archive 001 with unused relays removed',
    'q3_verified_core_sha256':source_hash,'dem_sha256':hashfile(dem_file),
    'reference_model':'B','reference_payload_kg':0.,
    'reference_speed_m_s':{k:float(getattr(model,k))for k in
          ('cruise_speed_m_s','climb_speed_m_s','descent_speed_m_s')},
    'clearance_m':50.,'hypothetical_hex_operating_agl_m':30.,
    'hex_count':len(hexrows),'directed_site_pairs':len(pair),'site_count':len(site_rows),
    'resource_audit_s':audit_s,
    'input_flights':len(flights),'input_relays':4,'input_boxes':len(delivery),
    'transport_only_blocks':10,'strict_relay_blocks':3,
    'transport_only_partition_count':{'2':2**9-1,'3':(3**10-3*2**10+3)//6},
    'strict_partition_count':{'2':3,'3':1},
    'spatial_medoid_is_new_base':False,
    'delivery_schedule_changed':False,
    'Pareto_scope':'only four strict partitions; resource ratio is dimensionless, not an equipment price',
    'outputs':['spatial_reference_pairwise.csv','hex_terrain_time_B.csv',
       'site_terrain_workload_B.csv','group_spatial_compactness_B.csv',
       'scenario_spatial_tradeoff_B.csv']}
(a.data_dir/'spatial_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(trade)[['scenario_id','shortage_count','transport_workload_gini_normalized',
                            'spatial_compactness_kg_min','pareto_in_three_diagnostics']].to_string(index=False))
