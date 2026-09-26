"""从原始Excel和最终工作DEM重算改进方案，不使用求解器或缓存航段。"""
from pathlib import Path
import json,sys,math,hashlib,collections
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parents[1]/'q3_source'))
from 问题二_物理模型与输入 import read_inputs,pairwise_legs,route,charge_seconds,DATA

nodes,boxes,models,drones,batteries=read_inputs()
legs=pairwise_legs(nodes)
stem='全局认证_集成列池CP_候选007'
f=pd.read_csv(P/f'{stem}_架次.csv');d=pd.read_csv(P/f'{stem}_逐箱.csv');a=pd.read_csv(P/f'{stem}_航段.csv')
assert f['架次编号'].is_unique
assert d['货箱编号'].is_unique and set(d['货箱编号'])==set(boxes.index)
checked=0;total_late=0.;energy=0.;vol=0.;mass=0.;all_alloc=[];arcs_count=0
route_checks=[]
for rec in f.to_dict('records'):
    rid=rec['架次编号'];dl=d.loc[d['架次编号'].eq(rid)]
    seq=tuple(rec['访问顺序'].split('→')[1:-1])
    assert rec['访问顺序'].split('→')[0]=='O01' and rec['访问顺序'].split('→')[-1]=='O01'
    assert seq and len(seq)==len(set(seq))
    assert set(dl['服务区'])==set(seq)
    assert set(rec['货箱编号'].split(','))==set(dl['货箱编号'])
    alloc={site:[bid for bid in dl['货箱编号'] if boxes.loc[bid,'node_id']==site] for site in seq}
    physical=route(seq,alloc,rec['机型'],boxes,models,legs,at=rec['开始秒'])
    assert physical is not None
    assert abs(physical['energy_kwh']-rec['能耗kWh'])<1e-7
    assert abs(physical['mass_kg']-rec['质量kg'])<1e-8
    assert abs(physical['volume_m3']-rec['体积m3'])<1e-8
    m=models.loc[rec['机型']]
    assert rec['质量kg']<=m.max_payload_kg+1e-8
    assert rec['体积m3']<=m.max_volume_m3+1e-8
    assert rec['能耗kWh']<=m.battery_kwh*(1-m.reserve_percent/100)+1e-8
    elapsed=int(m.fixed_prep_s+m.load_per_box_s*len(dl));src='O01'
    ar=a.loc[a['架次编号'].eq(rid)]
    assert len(ar)==len(seq)+1
    for i,dst in enumerate((*seq,'O01')):
        given=ar.iloc[i];raw=physical['leg_rows'][i];leg=legs[src,dst]
        assert given['起点']==src and given['终点']==dst
        for field,expected in [('剩余载荷kg',raw['remaining_payload_kg']),('距离m',leg['distance_m']),
                               ('巡航海拔m',leg['cruise_altitude_m']),('爬升m',leg['climb_m']),
                               ('下降m',leg['descent_m']),('航段能耗kWh',raw['energy_kwh'])]:
            assert abs(float(given[field])-expected)<1e-7,(rid,field)
        flight=int(math.ceil(raw['flight_s']-1e-10))
        assert given['飞行秒']==flight
        elapsed+=flight;arcs_count+=1
        if dst!='O01':
            elapsed+=int(m.handoff_base_s+m.handoff_per_box_s*len(alloc[dst]))
            for bid in alloc[dst]:
                br=dl.loc[dl['货箱编号'].eq(bid)].iloc[0]
                arrival=int(br['送达秒'])
                assert arrival==rec['开始秒']+elapsed
                assert arrival<=boxes.loc[bid,'hard_deadline_s']+1e-8
                assert arrival<=boxes.loc[bid,'target_delivery_s']+1e-8
                assert br['服务区']==boxes.loc[bid,'node_id']
                assert br['期望送达秒']==boxes.loc[bid,'target_delivery_s']
                assert br['优先系数']==boxes.loc[bid,'priority']
                if math.isfinite(boxes.loc[bid,'hard_deadline_s']):
                    assert br['硬截止秒']==boxes.loc[bid,'hard_deadline_s']
                else:assert pd.isna(br['硬截止秒'])
                late=boxes.loc[bid,'priority']*max(0,arrival-boxes.loc[bid,'target_delivery_s'])/boxes.loc[bid,'target_delivery_s']
                assert abs(br['归一化加权延误']-late)<1e-10
                total_late+=late;checked+=1;all_alloc.append(bid)
        src=dst
    assert rec['开始秒']>=0 and int(rec['开始秒'])==rec['开始秒']
    assert rec['返回秒']==rec['开始秒']+elapsed
    soc=1-physical['energy_kwh']/m.battery_kwh
    charge=charge_seconds(soc,batteries[rec['电池']]['full_charge_s'])
    assert rec['电池充满秒']==rec['返回秒']+int(math.ceil(charge-1e-9))
    assert drones[rec['无人机']]['model']==rec['机型']==batteries[rec['电池']]['model']
    assert abs(rec['返航SOC%']-100*soc)<1e-7
    energy+=physical['energy_kwh'];mass+=physical['mass_kg'];vol+=physical['volume_m3']
    route_checks.append({'架次编号':rid,'路线':rec['访问顺序'],'货箱数':len(dl),'质量kg':physical['mass_kg'],
                         '能耗kWh':physical['energy_kwh'],'开始秒':rec['开始秒'],'返回秒':rec['返回秒'],'核验':'通过'})
assert checked==80 and sorted(all_alloc)==sorted(boxes.index)
peaks=[]
for resource,endcol,inventory in [('无人机','返回秒',drones),('电池','电池充满秒',batteries)]:
    for name,part in f.groupby(resource):
        ordered=part.sort_values('开始秒')
        assert name in inventory
        assert np.all(ordered['开始秒'].to_numpy()[1:]>=ordered[endcol].to_numpy()[:-1]),name
    for typ,part in f.groupby('机型'):
        events=[(int(r['开始秒']),1) for r in part.to_dict('records')]+[(int(r[endcol]),-1) for r in part.to_dict('records')]
        count=0;peak=0
        for t,delta in sorted(events):count+=delta;peak=max(peak,count);assert count>=0
        capacity=sum(v['model']==typ for v in inventory.values())
        assert peak<=capacity
        peaks.append({'资源':resource,'机型':typ,'峰值':peak,'库存':capacity})
hard=d.loc[d['硬截止秒'].notna()]
old_vector=[0.0,7582,66.88893290579065,22] # 原审计记录，不用于物理航段复核
new_vector=[float(total_late),int(f['返回秒'].max()),float(energy),len(f)]
assert new_vector[0]<=old_vector[0]+1e-10 and new_vector[1]<=old_vector[1] and new_vector[2]<=old_vector[2]+1e-10 and new_vector[3]<=old_vector[3]
assert new_vector[1]<old_vector[1] and new_vector[2]<old_vector[2]-1e-10
sources={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['最终工作DEM.tif','调度中心与服务区.xlsx','物资需求与配送时限.xlsx','运输无人机数据.xlsx']}
result={'status':'通过','verification_basis':'重新读取三张原始Excel及最终工作DEM，重算所有节点航段与物理能耗；不读取求解缓存',
 'source_sha256':sources,'verified_sorties':len(f),'verified_boxes':checked,'verified_legs':arcs_count,
 'unique_box_coverage':True,'mass_kg':mass,'volume_m3':vol,'energy_kwh':energy,
 'weighted_lateness':total_late,'all_expected_deadlines_met':True,'hard_box_count':len(hard),'all_hard_deadlines_met':True,
 'min_hard_slack_s':float((hard['硬截止秒']-hard['送达秒']).min()),
 'last_delivery_s':int(d['送达秒'].max()),'all_return_s':int(f['返回秒'].max()),
 'min_SOC_percent':float(f['返航SOC%'].min()),'no_UAV_overlap':True,'no_battery_overlap':True,
 'battery_full_before_reuse':True,'resources':peaks,'sorties_by_type':f['机型'].value_counts().sort_index().to_dict(),
 'objective_order':['加权延误','全部返航秒','总能耗kWh','架次数'],
 'original_objectives':old_vector,'candidate_objectives':new_vector,
 'strict_dominance_of_original':True,'time_reduction_s':old_vector[1]-new_vector[1],
 'energy_reduction_kwh':old_vector[2]-new_vector[2],
 'energy_reduction_percent':100*(old_vector[2]-new_vector[2])/old_vector[2],
 'global_conclusion':'该可行改进方案严格支配原推荐解，因此原推荐解不是完整问题的全局Pareto最优；不据此声称改进方案全局最优',
 'fixed_route_proof_scope':'原路线结构下7582秒最优的下界证明保持成立；更换路线后该下界不再适用',
 'route_checks':route_checks}
previous_vector=[0.0,7559,65.4276437615,22] # 原审计记录
assert all(a<=b+1e-9 for a,b in zip(new_vector,previous_vector))
assert new_vector[1]<previous_vector[1] and new_vector[2]<previous_vector[2]-1e-9 and new_vector[3]<previous_vector[3]
result['previous_improved_objectives']=previous_vector
result['strict_dominance_of_previous_improved']=True
result['candidate_file_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [P/(stem+'.json')]+[P/(stem+'_'+s+'.csv') for s in ['架次','逐箱','航段']]}
result['candidate_frozen_id']='优先级方案007'
(P/'优先级方案007_原始DEM独立核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['route_checks','source_sha256']},ensure_ascii=False,indent=2))
