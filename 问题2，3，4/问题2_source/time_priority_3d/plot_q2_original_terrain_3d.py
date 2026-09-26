"""在问题三原V7三维DEM绘图器中绘制问题二19/22架次的真实航段。

调用原Scene.base / flight / nodes，保持同一色带、海拔、三维视角和字号；
航段位置、爬升/巡航/下降时刻从问题二原始物理模型及原航段CSV计算。
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
Q3 = ROOT / 'q3_source'
SLOW = ROOT / 'q2_source/全局认证_冻结最终十九架次'
FAST = OUT / '问题二_真正时间优先_22架次'
os.environ['Q2_CERTIFIED_PREFIX'] = str(FAST)
sys.path.insert(0, str(Q3))
from 问题三_V7_六面板三维山地航线 import Scene, MC, CM, plt, save_v7
from 问题二_物理模型与输入 import read_inputs
from q1_flat.精确算法 import utm49
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

N, B, G, _, _ = read_inputs()
coordinates = {site: np.array([*utm49(float(node.longitude_deg),
                                    float(node.latitude_deg)),
                                float(node.operating_altitude_m)])
               for site, node in N.iterrows()}
relay = pd.read_csv(str(Q3 / '问题三_时间主方案_中继架次.csv'))
relay = relay.rename(columns={'经度': 'lon', '纬度': 'lat'})
scene = Scene(relay)


def trajectory(prefix: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flights = pd.read_csv(str(prefix) + '_架次.csv')
    deliveries = pd.read_csv(str(prefix) + '_逐箱.csv')
    legs = pd.read_csv(str(prefix) + '_航段.csv')
    rows = []
    for r in flights.itertuples(index=False):
        machine = G.loc[r.机型]
        ids = r.货箱编号.split(',')
        t = float(r.开始秒 + machine.fixed_prep_s + machine.load_per_box_s * len(ids))
        for leg in legs[legs['架次编号'].eq(r.架次编号)].itertuples(index=False):
            src, dst = leg.起点, leg.终点
            first = coordinates[src].copy()
            last = coordinates[dst].copy()
            top0 = first.copy()
            top1 = last.copy()
            top0[2] = top1[2] = leg.巡航海拔m
            flight_start = t
            for stage, a, b, duration in [
                ('爬升', first, top0, leg.爬升m / machine.climb_speed_m_s),
                ('巡航', top0, top1, leg.距离m / machine.cruise_speed_m_s),
                ('下降', top1, last, leg.下降m / machine.descent_speed_m_s),
            ]:
                if duration > 1e-10:
                    data = {'route': r.架次编号, 'model': r.机型, 'stage': stage,
                            'from_node': src, 'to_node': dst,
                            'start_s': t, 'end_s': t + duration,
                            'p0_east': a[0], 'p0_north': a[1], 'p0_z': a[2],
                            'p1_east': b[0], 'p1_north': b[1], 'p1_z': b[2]}
                    rows.append(data)
                t += duration
            # 整数秒飞行时间和服务点交接来自原路线模型。
            # 用此前连续三阶段的开始时刻验证向上取整，避免绘图漂移。
            rounded_end = flight_start + int(leg.飞行秒)
            assert rounded_end + 1e-7 >= t, (r.架次编号, src, dst)
            t = rounded_end
            if dst != 'O01':
                local_count = sum(B.loc[bid, 'node_id'] == dst for bid in ids)
                t += machine.handoff_base_s + machine.handoff_per_box_s * local_count
                observed = deliveries[(deliveries['架次编号'].eq(r.架次编号)) &
                                      (deliveries['服务区'].eq(dst))]
                assert observed['送达秒'].eq(t).all()
        assert abs(t-r.返回秒) < 1e-7, (r.架次编号, t, r.返回秒)
    phases = pd.DataFrame(rows)
    assert len(phases) >= len(legs)*2
    assert phases['p0_z'].min() >= 0 and phases['p1_z'].max() < 2500
    return flights, deliveries, phases


old_f, old_d, old_segments = trajectory(SLOW)
new_f, new_d, new_segments = trajectory(FAST)
assert (len(old_f), len(new_f), len(old_d), len(new_d)) == (19,22,80,80)
scene.zlim = (0., float(np.ceil(max(scene.zlim[1], old_segments['p1_z'].max(),
                                    new_segments['p1_z'].max())/200)*200))
old_segments.assign(case='19架次').to_csv(OUT/'三维_19架次_真实航段.csv',index=False,encoding='utf-8-sig')
new_segments.assign(case='22架次').to_csv(OUT/'三维_22架次_真实航段.csv',index=False,encoding='utf-8-sig')


def clip(data: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    keep = data[(data.start_s < t1) & (data.end_s > t0)].copy()
    if keep.empty:
        return keep
    for idx, r in keep.iterrows():
        start = max(float(r.start_s), t0)
        end = min(float(r.end_s), t1)
        first = r[['p0_east','p0_north','p0_z']].to_numpy(float)
        last = r[['p1_east','p1_north','p1_z']].to_numpy(float)
        p0 = first + (last-first)*(start-r.start_s)/(r.end_s-r.start_s)
        p1 = first + (last-first)*(end-r.start_s)/(r.end_s-r.start_s)
        keep.loc[idx,['p0_east','p0_north','p0_z']] = p0
        keep.loc[idx,['p1_east','p1_north','p1_z']] = p1
        keep.loc[idx,['start_s','end_s']] = (start,end)
    return keep


def common_legend(fig):
    handles = [Line2D([],[],color=MC[t],lw=3.3,label=f'{t} 型运输') for t in 'ABC']
    handles += [Line2D([],[],marker='*',markersize=14,markerfacecolor='black',
                       markeredgecolor='white',linestyle='',label='O01 调度中心')]
    fig.legend(handles=handles,ncol=4,loc='upper center',bbox_to_anchor=(.5,1.003),
               frameon=False,fontsize=18,handlelength=2.1)
    cb = fig.colorbar(ScalarMappable(norm=scene.norm,cmap=CM),
                      cax=fig.add_axes([.35,.041,.29,.013]),orientation='horizontal')
    cb.set_ticks([100,400,700,1000])
    cb.ax.tick_params(labelsize=15,length=3,pad=3)
    cb.outline.set_linewidth(.8)
    fig.text(.653,.045,'地形海拔（米）',fontsize=18,va='center')


def draw(panels, filename):
    fig=plt.figure(figsize=(22.7,13.8))
    grid=fig.add_gridspec(2,3,left=.005,right=.990,bottom=.057,top=.955,
                         wspace=.025,hspace=-.025)
    details=[]
    for i,(title,frame) in enumerate(panels):
        ax=fig.add_subplot(grid[i//3,i%3],projection='3d')
        scene.base(ax,chr(97+i),title,axis_labels=i>=3)
        scene.flight(ax,frame,width=3.05,direction=i in (0,1))
        scene.nodes(ax,frame,labels=i in (0,1))
        details.append({'panel': chr(97+i), 'title': title,
                        'drawn_phases': len(frame),
                        'source_routes': int(frame.route.nunique())})
    common_legend(fig)
    target=OUT/filename
    save_v7(fig,target,dpi=235)
    return details


by_model = [
    ('19架次 · 原方案', old_segments), ('22架次 · 时间优先', new_segments),
    ('22架次 · A型运输', new_segments[new_segments.model.eq('A')]),
    ('22架次 · B型运输', new_segments[new_segments.model.eq('B')]),
    ('22架次 · C型运输', new_segments[new_segments.model.eq('C')]),
    ('22架次 · 首小时', clip(new_segments,0,3600)),
]
periods=[]
for lo,hi in [(0,2400),(2400,4800),(4800,7200)]:
    periods.extend([(f'19架次 · {lo//60}—{hi//60}分',clip(old_segments,lo,hi)),
                    (f'22架次 · {lo//60}—{hi//60}分',clip(new_segments,lo,hi))])
qa = {'status':'PASS',
      'case_19': {'sorties':len(old_f),'last_delivery_s':int(old_d.送达秒.max()),
                  'last_return_s':int(old_f.返回秒.max())},
      'case_22': {'sorties':len(new_f),'last_delivery_s':int(new_d.送达秒.max()),
                  'last_return_s':int(new_f.返回秒.max())},
      'dem_sha256':hashlib.sha256((Q3/'q1_flat/最终工作DEM.tif').read_bytes()).hexdigest(),
      'camera':'问题三V7原始elev=44,azim=-62，正投影',
      'terrain':'同一修正DEM、同一RdYlBu_r色带，地形海拔真实米值',
      'vertical_display_exaggeration':3.5,
      'height_data_altered':False,
      'model_panels':draw(by_model,'问题二_原版视角三维_19与22架次_航线机型.png'),
      'period_panels':draw(periods,'问题二_原版视角三维_19与22架次_三时段.png')}
(OUT/'问题二_三维航线绘图核验.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'status':qa['status'],'figures':2,'old':qa['case_19'],
                  'fast':qa['case_22']},ensure_ascii=False),flush=True)
