"""Restore the original Q3 three-dimensional mountain style for Q4 maps.

Figure 07: corrected 3-D DEM, fixed Q3 air routes and both recommended groups.
Figure 12: the three indivisible task components and every legal 2-group choice.
Figure 13: all legal partitions with box-weighted standard-deviation ellipses.

The flights, heights, communication assignments and delivery times are frozen.
The one-standard-deviation contours in Fig. 13 describe the fixed sites'
dispersion; they are not confidence regions for an estimated effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--q3-dir', type=Path, default=ROOT/'q3_source')
ap.add_argument('--data-dir', type=Path, default=ROOT/'data')
ap.add_argument('--output-dir', type=Path, default=ROOT/'results')
ap.add_argument('--dpi', type=int, default=300)
a = ap.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=True)
os.environ['Q2_CERTIFIED_PREFIX'] = str(a.q3_dir/'问题三_时间主方案_运输')
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(a.q3_dir))

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from matplotlib import patheffects
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import pandas as pd
from PIL import Image
from 问题三_V3_基础绘图 import load_v3
from 问题三_V7_六面板三维山地航线 import Scene, CM, INK

plt.rcParams.update({'pdf.fonttype': 42, 'axes.unicode_minus': False,
                     'font.size': 18, 'axes.labelsize': 18,
                     'axes.titlesize': 20, 'legend.fontsize': 16})
GC = {'G1': CM(.035), 'G2': CM(.78), 'G3': CM(.985)}
K2, K3 = 'K2_03', 'K3_01'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, stem):
    png = a.output_dir/(stem+'.png')
    pdf = a.output_dir/(stem+'.pdf')
    tmp_png = a.output_dir/(stem+'.tmp.png')
    tmp_pdf = a.output_dir/(stem+'.tmp.pdf')
    fig.savefig(tmp_png, dpi=a.dpi, bbox_inches='tight', pad_inches=.10)
    # Rasterize the original DEM in PDF; labels, routes and ellipses stay sharp.
    fig.savefig(tmp_pdf, dpi=170, bbox_inches='tight', pad_inches=.10)
    plt.close(fig)
    with Image.open(tmp_png) as im:
        im.verify()
    with Image.open(tmp_png) as im:
        im.load()
        size = list(im.size)
    assert tmp_pdf.stat().st_size > 5000
    tmp_png.replace(png)
    tmp_pdf.replace(pdf)
    return {'png': png.name, 'pdf': pdf.name, 'size_px': size,
            'sha256': sha(png), 'pdf_bytes': pdf.stat().st_size}


f, b, m, c, t, s, provenance = load_v3(
    str(a.q3_dir/'问题三_时间主方案'),
    str(a.q3_dir/'问题三_时间主方案_精确'),
    str(a.q3_dir/'问题三_V3_站点.csv'))
assert len(f) == 22 and len(b) == 80 and len(m) == 4 and len(t)>0
sites = pd.read_csv(a.data_dir/'site_groups.csv')
scenarios = pd.read_csv(a.data_dir/'scenarios.csv').set_index('scenario_id')
components = pd.read_csv(a.data_dir/'components.csv')
assert len(sites) == 15*4 and set(scenarios.index) == {'K2_01', 'K2_02', 'K2_03', 'K3_01'}
assert set(components.component_id) == {'C1', 'C2', 'C3'}
lookup = {sid: sites[sites.scenario_id.eq(sid)].set_index('site_id').group_id.to_dict()
          for sid in scenarios.index}
comp = components.set_index('site_id').component_id.to_dict()
compmap = {s: 'G'+comp[s][1] for s in comp}
assert len(compmap) == 15
routes = {row['架次编号']: set(row['访问顺序'].split('→'))-{'O01'}
          for _, row in f.iterrows()}
route_component = {}
route_group = {}
for rid, visited in routes.items():
    assert visited and len({comp[x] for x in visited}) == 1, rid
    route_component[rid] = comp[next(iter(visited))]
    for sid, mapping in lookup.items():
        assert len({mapping[x] for x in visited}) == 1, (sid, rid)
        route_group.setdefault(sid, {})[rid] = mapping[next(iter(visited))]

scene = Scene(m)
stations = s.set_index('station_id')
used = stations.loc[m.station_id.unique()]
scene.zlim = (0., float(np.ceil(max(scene.zlim[1],t.p0_z.max(),t.p1_z.max(),
                                    m.z.max(),used.cruise_z.max())/200)*200))
for label, xyz in (('p0', scene.xyz(t, 'p0')), ('p1', scene.xyz(t, 'p1'))):
    assert np.array_equal(xyz[:,2], t[label+'_z'].to_numpy(float))


def base(ax, letter, title, axis_labels=True):
    scene.base(ax, letter, title, axis_labels=axis_labels)
    # Keep Q3's exact mesh, viewing angle and unmodified 3.5x display aspect.
    ax.collections[0].set_rasterized(True)


def group_flights(ax, frame, mapping, width=3.15, show_labels=True):
    if frame.empty:
        return
    dd = frame.copy()
    dd['group'] = dd.route.map(mapping)
    assert dd.group.notna().all()
    p, q = scene.xyz(dd, 'p0'), scene.xyz(dd, 'p1')
    moving = np.linalg.norm((q-p)*[1,1,.001], axis=1) > 1e-9
    seg = np.stack([p[moving],q[moving]],axis=1)
    ax.add_collection3d(Line3DCollection(seg,colors='white',linewidths=width+1.2,zorder=7))
    ax.add_collection3d(Line3DCollection(seg,colors=[GC[g] for g in dd.loc[moving,'group']],
                                             linewidths=width,zorder=8))
    observed = sorted((set(dd.from_node)|set(dd.to_node))-{'O01'})
    site_group = {}
    for rid in dd.route.unique():
        for site in routes[rid]:
            assert site not in site_group or site_group[site] == mapping[rid]
            site_group[site] = mapping[rid]
    ax.scatter(*np.array([scene.node[x] for x in observed]).T,
               c=[GC[site_group[x]] for x in observed],edgecolors='white',
               s=62,linewidths=1.05,depthshade=False,zorder=14)
    if show_labels:
        small = [x for x in observed if x in ('S006','S011')]
        other = sorted(observed,key=lambda x:scene.node[x][2],reverse=True)
        for x in list(dict.fromkeys(small+other[:1]))[:3]:
            v = scene.node[x]
            ax.text(v[0]+.09,v[1],v[2]+65,x,fontsize=15,color=INK,zorder=20)


def relay_for(frame):
    ids = set(frame.loc[~frame.direct_ok,'relay_mission_id'].dropna())
    return m[m.relay_mission_id.isin(ids)]


def legend_and_scale(fig, include_models=False):
    hs = [Line2D([],[],color=GC[g],lw=3.3,label=f'第{i}组') for i,g in enumerate(GC,1)]
    if include_models:
        hs += [Line2D([],[],color=CM(v),lw=3.3,label=f'{name}型运输')
               for name,v in [('A',.025),('B',.79),('C',.98)]]
        for h in hs[:3]:h.set_label('c–d '+h.get_label())
        for h in hs[3:6]:h.set_label('b '+h.get_label())
    hs += [Line2D([],[],color=INK,lw=2.7,label='中继航迹'),
           Line2D([],[],color=INK,lw=1.6,ls='--',label='回传链路')]
    fig.legend(handles=hs,ncol=len(hs),loc='upper center',bbox_to_anchor=(.5,1.004),
               frameon=False,fontsize=14.2 if include_models else 17,
               columnspacing=1.25,handlelength=2.0)
    bar=fig.colorbar(ScalarMappable(norm=scene.norm,cmap=CM),
                     cax=fig.add_axes([.345,.042,.30,.014]),orientation='horizontal')
    bar.set_ticks([100,400,700,1000]);bar.ax.tick_params(labelsize=15,length=3,pad=3)
    bar.outline.set_linewidth(.8)
    fig.text(.656,.049,'地形海拔（米）',fontsize=17,va='center')


def fig_terrain_3d():
    fig=plt.figure(figsize=(20.9,14.4))
    gs=fig.add_gridspec(2,2,left=.008,right=.992,bottom=.078,top=.958,
                        wspace=.012,hspace=-.012)
    titles=['30米修正DEM与服务点','固定问题三 · 22条运输航线',
            '两组 K2_03 · 真实航线','三组 K3_01 · 真实航线']
    records=[]
    for i,title in enumerate(titles):
        ax=fig.add_subplot(gs[i//2,i%2],projection='3d')
        base(ax,chr(97+i),title,axis_labels=i>=2)
        if i==0:
            observed=sorted(compmap)
            vals=np.array([scene.node[x] for x in observed])
            ax.scatter(*vals.T,s=68,c='white',edgecolor=INK,lw=1.05,
                       depthshade=False,zorder=13)
            for x in ['S006','S011','S008']:
                v=scene.node[x]
                ax.text(v[0]+.12,v[1],v[2]+65,x,fontsize=15,color=INK,zorder=20)
            records.append({'panel':'a','sorties':0,'terrain_sites':len(observed)})
            continue
        if i==1:
            scene.flight(ax,t,width=3.0,direction=True)
            scene.nodes(ax,t,labels=True)
            frame=t
        else:
            sid=K2 if i==2 else K3
            frame=t
            group_flights(ax,frame,route_group[sid])
        missions=relay_for(frame)
        scene.relay(ax,missions,stations,flight=True)
        records.append({'panel':chr(97+i),'sorties':int(frame.route.nunique()),
                        'intervals':len(frame),'relay_missions':sorted(missions.relay_mission_id),
                        'transport_seconds':float(frame.duration_s.sum())})
    legend_and_scale(fig,include_models=True)
    fig.text(.5,.010,'沿用问题三同一修正DEM、视角和色带；航段按实际爬升、巡航与下降海拔绘制，纵向仅显示夸张3.5倍。',
             ha='center',fontsize=13)
    out=save(fig,'问题四_图07_真实DEM地形与固定航线分组')
    assert [r['sorties'] for r in records]==[0,22,22,22]
    return out,records


def fig_all_groups_3d():
    fig=plt.figure(figsize=(22.7,13.8))
    gs=fig.add_gridspec(2,3,left=.005,right=.990,bottom=.069,top=.955,
                        wspace=.025,hspace=-.025)
    csum=components.groupby('component_id').agg(n_sites=('site_id','size'),
                                                  boxes=('n_boxes','sum'))
    panels=[(cid,f'{cid} · {csum.loc[cid,"n_sites"]}区 / {csum.loc[cid,"boxes"]}箱')
            for cid in ('C1','C2','C3')]
    panels.extend((sid,f'{sid} · {scenarios.loc[sid,"component_partition"]}'
                  + (' ★' if sid==K2 else '')) for sid in ('K2_01','K2_02','K2_03'))
    records=[]
    for i,(key,title) in enumerate(panels):
        ax=fig.add_subplot(gs[i//3,i%3],projection='3d')
        base(ax,chr(97+i),title,axis_labels=i>=3)
        if key.startswith('C'):
            selected={rid for rid,cid in route_component.items() if cid==key}
            mapping={rid:'G'+key[1] for rid in selected}
            site_assignment=compmap
        else:
            selected=set(routes)
            mapping=route_group[key]
            site_assignment=lookup[key]
        frame=t[t.route.isin(selected)]
        assert len(set(frame.route))==len(selected)
        group_flights(ax,frame,mapping,show_labels=i<3)
        missions=relay_for(frame)
        if len(missions):scene.relay(ax,missions,stations,flight=True)
        records.append({'panel':chr(97+i),'key':key,'sorties':len(selected),
                        'route_ids':sorted(selected),'source_intervals':len(frame),
                        'relay_missions':sorted(missions.relay_mission_id),
                        'transport_seconds':float(frame.duration_s.sum())})
        # Frozen sites and phases are unchanged; only route ownership is colored.
        assert all(site_assignment[x] in GC for rid in selected for x in routes[rid])
    assert sum(x['sorties'] for x in records[:3])==22
    assert sum(x['source_intervals'] for x in records[:3])==len(t)
    assert all(x['sorties']==22 for x in records[3:])
    legend_and_scale(fig)
    fig.text(.5,.012,'上排是不可拆分的真实任务组件；下排是完整的三种两组划分，航线与通信任务均继承问题三。',
             ha='center',fontsize=13)
    out=save(fig,'问题四_图12_任务组件及全部两组候选_三维地形航线')
    return out,records


def weighted_geometry(points,weights):
    weights=np.asarray(weights,float)
    pts=np.asarray(points,float)
    mu=np.average(pts,weights=weights,axis=0)
    if len(pts)==1:
        return mu,None,'点',0.,0.,0.
    cov=((pts-mu)*weights[:,None]).T@(pts-mu)/sum(weights)
    val,vec=np.linalg.eigh(cov)
    val=np.maximum(val,0.)[::-1]
    vec=vec[:,::-1]
    # One spatial standard deviation.  A 95% dispersion contour exceeds the
    # mapped DEM for the 13-site component and obscures the partition.
    lengths=np.sqrt(np.maximum(val,0.))
    angle=math.degrees(math.atan2(vec[1,0],vec[0,0]))
    geometry='线' if val[1] < 1e-10 else '椭圆'
    return mu,vec,geometry,float(lengths[0]),float(lengths[1]),angle


def fig_spatial_ellipses():
    fig,axs=plt.subplots(2,2,figsize=(19.9,14.1),
        gridspec_kw={'left':.068,'right':.973,'bottom':.128,'top':.949,
                     'wspace':.135,'hspace':.19})
    cases=['K2_01','K2_02','K2_03','K3_01']
    ellipse_rows=[]
    titles=[f'{sid} · {scenarios.loc[sid,"component_partition"]}'
            + (' ★' if sid==K2 else '') for sid in cases]
    for i,(sid,ax) in enumerate(zip(cases,axs.flat)):
        terrain=ax.pcolormesh(scene.x,scene.y,scene.z,cmap=CM,norm=scene.norm,
                              shading='nearest',rasterized=True,zorder=0)
        ax.contour(scene.x[::5,::5],scene.y[::5,::5],scene.z[::5,::5],
                   levels=[200,400,600,800,1000],colors=INK,linewidths=.55,
                   alpha=.27,zorder=1)
        mapping=lookup[sid]
        for g in sorted(set(mapping.values())):
            member=[x for x in mapping if mapping[x]==g]
            xy=np.array([scene.node[x][:2] for x in member])
            weight=np.array([int(components.loc[components.site_id.eq(x),'n_boxes'].iloc[0])
                             for x in member])
            mu,vec,typ,major,minor,angle=weighted_geometry(xy,weight)
            color=GC[g]
            if typ=='椭圆':
                ellipse=Ellipse(mu,width=2*major,height=2*minor,angle=angle,
                                ec=color,fc=color,lw=2.6,alpha=.14,zorder=2)
                ax.add_patch(ellipse)
                edge=Ellipse(mu,width=2*major,height=2*minor,angle=angle,
                             fc='none',ec=color,lw=2.6,ls=(0,(5,2)),zorder=4)
                edge.set_path_effects([patheffects.Stroke(linewidth=4.5,foreground='white'),
                                       patheffects.Normal()])
                ax.add_patch(edge)
            elif typ=='线':
                direction=vec[:,0]
                ends=np.array([mu-major*direction,mu+major*direction])
                ax.plot(ends[:,0],ends[:,1],lw=4.4,c='white',zorder=4)
                ax.plot(ends[:,0],ends[:,1],lw=2.4,c=color,ls='--',zorder=5)
            ax.scatter(xy[:,0],xy[:,1],s=73+8*weight,facecolors=[color],
                       edgecolors='white',linewidths=1.4,zorder=8)
            if typ=='点':
                ax.scatter(mu[0],mu[1],s=340,facecolors='none',edgecolors=color,
                           linewidths=2.7,zorder=7)
            else:
                ax.scatter(mu[0],mu[1],s=122,marker='X',facecolors=color,
                           edgecolors='white',linewidths=1.0,zorder=9)
            note=f'{g}：{len(member)}区 / {sum(weight)}箱'
            xytext=(7,8) if g=='G1' else (8,-19)
            mark=ax.annotate(note,mu,xytext=xytext,textcoords='offset points',
                             fontsize=12.5,fontweight='bold',color=INK,zorder=10)
            mark.set_path_effects([patheffects.withStroke(linewidth=3,foreground='white')])
            ellipse_rows.append({'scenario_id':sid,'group_id':g,
                                 'n_sites':len(member),'n_boxes':int(sum(weight)),
                                 'centroid_east_km':round(float(mu[0]),5),
                                 'centroid_north_km':round(float(mu[1]),5),
                                 'geometry':typ,'semi_major_km':round(major,5),
                                 'semi_minor_km':round(minor,5),
                                 'angle_degree':round(angle,4)})
        ax.scatter(0,0,s=310,marker='*',c=INK,edgecolor='white',lw=1.4,zorder=11)
        depot=ax.annotate('O01',(0,0),xytext=(0,-21),textcoords='offset points',
                          ha='center',fontsize=14.5,fontweight='bold',zorder=12)
        depot.set_path_effects([patheffects.withStroke(linewidth=3,foreground='white')])
        for site in ('S006','S011'):
            v=scene.node[site]
            anno=ax.annotate(site,v[:2],xytext=(7,7),textcoords='offset points',
                             fontsize=12.5,fontweight='bold',color=INK,zorder=12)
            anno.set_path_effects([patheffects.withStroke(linewidth=3,foreground='white')])
        ax.set(xlim=scene.xlim,ylim=scene.ylim,aspect='equal')
        if i%2==0:ax.set_ylabel('相对O01北向距离（km）',fontsize=15)
        if i>=2:ax.set_xlabel('相对O01东向距离（km）',fontsize=15)
        ax.set_title(f'{chr(97+i)}  {titles[i]}',loc='left',pad=8,
                     fontsize=17,fontweight='bold')
        ax.tick_params(labelsize=12.5,length=4)
        for sp in ax.spines.values():sp.set_linewidth(1.3)
    hs=[Line2D([],[],color=GC[g],lw=2.3,marker='o',mec='white',ms=9,label=f'第{i}组')
        for i,g in enumerate(GC,1)]
    hs.extend([Line2D([],[],color=INK,lw=2.0,ls='--',label='1σ空间标准差椭圆'),
               Line2D([],[],color=INK,marker='X',ls='',ms=10,label='货箱加权质心'),
               Line2D([],[],color=INK,marker='*',ls='',ms=12,label='O01')])
    fig.legend(handles=hs,ncol=6,loc='lower center',bbox_to_anchor=(.5,.069),
               frameon=False,fontsize=13.3,columnspacing=1.2,handlelength=2.1)
    bar=fig.colorbar(ScalarMappable(norm=scene.norm,cmap=CM),
                     cax=fig.add_axes([.37,.053,.26,.010]),orientation='horizontal')
    bar.set_ticks([100,400,700,1000]);bar.ax.tick_params(labelsize=11,pad=1)
    fig.text(.644,.059,'地形海拔（米）',fontsize=12,va='center')
    fig.text(.5,.018,'按货箱加权的1σ空间标准差椭圆，展示实际服务点分布；1站组画点、2站组画线，固定服务点不构成抽样置信区间。',
             ha='center',fontsize=12.1)
    out=save(fig,'问题四_图13_全部候选分组的空间质心与标准差椭圆')
    pd.DataFrame(ellipse_rows).to_csv(a.data_dir/'group_spatial_ellipses.csv',
                                      index=False,encoding='utf-8-sig')
    assert len(ellipse_rows)==9
    return out,ellipse_rows


def main():
    outputs=[]
    terrain,panel07=fig_terrain_3d();outputs.append(terrain)
    groups,panel12=fig_all_groups_3d();outputs.append(groups)
    ellipse,panel13=fig_spatial_ellipses();outputs.append(ellipse)
    qa={'status':'PASS','input_routes':len(f),'input_intervals':len(t),
        'input_relay_sorties':len(m),'input_boxes':len(b),
        'terrain_dem_sha256':sha(a.q3_dir/'q1_flat'/'最终工作DEM.tif'),
        'terrain_and_flight_altitude_unchanged':True,'shared_3d_scene_with_q3':True,
        'visual_vertical_exaggeration':3.5,
        'ellipse_meaning':'one box-weighted spatial standard-deviation ellipse for the fixed sites, not a confidence interval',
        'groups_with_one_or_two_sites_drawn_as_point_or_line':True,
        'panels07':panel07,'panels12':panel12,'ellipse_rows':panel13,
        'outputs':outputs}
    (a.output_dir/'问题四_三维地形与分组椭圆核验.json').write_text(
        json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','outputs':outputs,'ellipses':len(panel13)},
                     ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
