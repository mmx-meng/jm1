"""问题四：沿用问题三已批准样式的六面板平面与三维地图。

输入：已核验的 Q3 固定运输/中继/通信结果 + 问题四精确分组 CSV。
不改变航线几何、飞行海拔、时间或中继关联；不构造行政边界。
python plot_q4_maps.py --q3-dir ../q3_v10_execution --data-dir data --output-dir results
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--q3-dir','--source-dir',dest='q3_dir',default=os.environ.get('Q3_SOURCE',str(ROOT/'q3_source' if (ROOT/'q3_source').exists() else ROOT.parent/'q3_v10_execution')))
parser.add_argument('--data-dir',default=str(ROOT/'data'))
parser.add_argument('--output-dir',default=str(ROOT/'results'))
parser.add_argument('--dpi',type=int,default=340)
a=parser.parse_args()
Q3=Path(a.q3_dir).resolve(); DATA=Path(a.data_dir).resolve(); OUT=Path(a.output_dir).resolve();OUT.mkdir(parents=True,exist_ok=True)
os.environ['Q2_CERTIFIED_PREFIX']=str(Q3/'问题三_V3_最终方案_运输')
os.environ.setdefault('MPLBACKEND','Agg')
sys.path.insert(0,str(Q3))
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image
from 问题三_V3_基础绘图 import load_v3
from 问题三_V4_地图绘图 import CompactMap, halo, COORD, OFFSETS
from 问题三_V7_六面板三维山地航线 import Scene, plt, CM, INK, CN, save_v7
from 认证更新_地图剖面公用 import N

plt.rcParams.update({'font.family':['STIXGeneral',CN], 'font.size':19,
    'axes.labelsize':19,'xtick.labelsize':16,'ytick.labelsize':16,
    'axes.titlesize':22,'legend.fontsize':18,'axes.linewidth':1.7,
    'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black'})
GC={'G1':CM(.035),'G2':CM(.78),'G3':CM(.985)}
COMP_COLORS={'C1':GC['G1'],'C2':GC['G2'],'C3':GC['G3']}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bools(s):return s.astype(str).str.lower().isin(['true','1','1.0'])
def save(fig,path):
    tmp=path.with_suffix('.tmp.png');fig.savefig(tmp,dpi=a.dpi,bbox_inches='tight',pad_inches=.07);plt.close(fig)
    with Image.open(tmp) as im:im.verify()
    with Image.open(tmp) as im:im.load()
    tmp.replace(path)

def mapping_for(scenario):
    d=groups[groups.scenario_id.eq(scenario)].copy()
    assert len(d)==15 and d.site_id.is_unique
    return d.set_index('site_id').group_id.to_dict()

def attach_routes(mapping):
    rows=[]
    for _,r in f.iterrows():
        sites=set(r['访问顺序'].split('→'))-{'O01'}
        g={mapping[s] for s in sites};assert len(g)==1, f'架次跨组：{r["架次编号"]}'
        rows.append((r['架次编号'],g.pop()))
    return dict(rows)

def draw_nodes(ax,mapping,show=None,labels=True):
    for site,xy in COORD.items():
        if site=='O01':
            ax.scatter(*xy,s=255,marker='*',c=INK,ec='white',lw=1.15,zorder=25)
            halo(ax.annotate('O01',xy,xytext=(0,-22),textcoords='offset points',ha='center',fontsize=17,zorder=28,fontweight='bold'));continue
        active=show is None or mapping[site] in show
        ax.scatter(*xy,s=95 if active else 40,c=[GC[mapping[site]]] if active else '#e2e2e2',ec='white' if active else '#777777',lw=1.4 if active else .8,zorder=22)
        if labels and active:
            ox,oy=OFFSETS[site]
            halo(ax.annotate(site,xy,xytext=(ox,oy),textcoords='offset points',ha='right' if ox<0 else 'left',fontsize=16.2,zorder=26),3.3)

def planar_routes(ax,mapping,selected=None,alpha=.95,neutral=False,directions=True):
    rg=attach_routes(mapping);seen=set();drawn=[]
    for _,r in f.iterrows():
        g=rg[r['架次编号']]
        if selected is not None and g not in selected:continue
        seq=r['访问顺序'].split('→')
        for x,y in zip(seq[:-1],seq[1:]):
            key=(g,tuple(sorted([x,y])))
            if key in seen:continue
            seen.add(key);v=np.array([COORD[x],COORD[y]])
            color='#42464c' if neutral else GC[g];width=1.9 if neutral else 2.5
            ax.plot(v[:,0],v[:,1],c='white',lw=width+1.8,alpha=alpha,zorder=3)
            ax.plot(v[:,0],v[:,1],c=color,lw=width,alpha=alpha,zorder=4)
            length=np.linalg.norm(v[1]-v[0]);drawn.append((length,v,color))
    if directions and not neutral:
        mids=[]
        for _,v,c in sorted(drawn,key=lambda d:d[0],reverse=True):
            mid=v.mean(axis=0)
            if any(np.linalg.norm(mid-p)<.017 for p in mids):continue
            delta=(v[1]-v[0])*.10
            ax.annotate('',xy=mid+delta/2,xytext=mid-delta/2,arrowprops=dict(arrowstyle='-|>',mutation_scale=15,lw=1.9,color=c),zorder=9)
            mids.append(mid)
            if len(mids)==3:break

def base_map(ax,i,title,sty):
    ax.imshow(sty.rgb,extent=sty.zext,origin='upper',interpolation='nearest',zorder=0)
    ax.set_xlim(sty.ext[:2]);ax.set_ylim(sty.ext[2:]);ax.set_aspect(sty.aspect)
    ax.set_xticks([109.16,109.20,109.24,109.28]);ax.set_yticks([23.00,23.04,23.08])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'));ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(labelsize=16,width=1.6,length=4.5,pad=3,labelleft=i%3==0,labelbottom=i>=3)
    for sp in ax.spines.values():sp.set_linewidth(1.7);sp.set_color('black')
    ax.set_title(f'{chr(97+i)}  {title}',loc='left',fontsize=22,pad=9)
    if i%3==0:ax.set_ylabel('纬度（°）',fontsize=19,labelpad=6)
    if i>=3:ax.set_xlabel('经度（°）',fontsize=19,labelpad=5)

def shared_map_legend(fig,sty):
    handles=[Line2D([],[],color=GC[g],lw=3.4,marker='o',mec='white',ms=8,label=f'第{k}组') for k,g in enumerate(GC,1)]
    handles += [Line2D([],[],ls='',marker='*',mfc=INK,mec='white',ms=15,label='调度中心'),Line2D([],[],ls='--',c=INK,lw=2,label='中继回传'),Line2D([],[],ls='',marker='D',mfc=GC['G3'],mec='white',ms=10,label='中继悬停')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.055,.006),ncol=6,frameon=False,fontsize=18,columnspacing=1.2,handlelength=1.45)
    cb=fig.colorbar(ScalarMappable(norm=sty.norm,cmap=CM),cax=fig.add_axes([.735,.029,.21,.012]),orientation='horizontal')
    cb.set_ticks([200,600,1000]);cb.ax.tick_params(labelsize=16,width=1.1,length=3,pad=2);cb.outline.set_linewidth(1.0)
    fig.text(.840,.052,'地形海拔（米）',ha='center',fontsize=17)

def plot_planar():
    sty=CompactMap(m)
    cases=scenarios[scenarios.n_groups.eq(2)].sort_values('scenario_id')
    ids=cases.scenario_id.tolist();assert len(ids)==3
    tri=scenarios[scenarios.n_groups.eq(3)].scenario_id.iloc[0]
    maps=[mapping_for(tri)]+[mapping_for(x) for x in ids]+[mapping_for(tri),mapping_for(tri)]
    titles=['不可拆分任务组']+[f'两组方案{k+1}'+(' ★' if x==recommended else '') for k,x in enumerate(ids)]+['三组方案','中继共同保障']
    fig=plt.figure(figsize=(23.2,13.9));gs=fig.add_gridspec(2,3,left=.058,right=.989,bottom=.11,top=.955,wspace=.035,hspace=.12)
    axs=[]
    for i,title in enumerate(titles):
        ax=fig.add_subplot(gs[i//3,i%3]);axs.append(ax);base_map(ax,i,title,sty)
        planar_routes(ax,maps[i],neutral=i in [0,5],alpha=.65 if i in [0,5] else .95,directions=i not in [0,5])
        draw_nodes(ax,maps[i])
    sty.north(axs[0])
    # a 显示实际共同任务的不可拆分组件；f 的中继保障为核验通过的真实区间。
    cov=sty.coordinates(t[~t.direct_ok])
    relay_route_groups=attach_routes(maps[5])
    for _,r in cov.iterrows():
        if np.linalg.norm([r.p1_lon-r.p0_lon,r.p1_lat-r.p0_lat])<1e-11:continue
        xx=[r.p0_lon,r.p1_lon];yy=[r.p0_lat,r.p1_lat]
        axs[5].plot(xx,yy,c='white',lw=4.5,zorder=5);axs[5].plot(xx,yy,c=GC[relay_route_groups[r.route]],lw=2.8,zorder=6)
    for (lon,lat),part in m.groupby(['lon','lat'],sort=False):
        xy=COORD['O01'];axs[5].plot([xy[0],lon],[xy[1],lat],c='white',lw=3.8,zorder=9)
        axs[5].plot([xy[0],lon],[xy[1],lat],c=INK,lw=2.0,ls=(0,(4,3)),zorder=10)
        axs[5].scatter(lon,lat,s=165,marker='D',c=[GC['G3']],ec='white',lw=1.4,zorder=31)
        label='/'.join(part.relay_mission_id)
        halo(axs[5].annotate(label,(lon,lat),xytext=(9,9),textcoords='offset points',fontsize=16,zorder=32,fontweight='bold'))
    shared_map_legend(fig,sty)
    out=OUT/'问题四_图01_六面板分组空间比较.png';save(fig,out)
    return out,{'panels':list(zip(list('abcdef'),titles)),'all_geographic_limits':sty.ext,'no_administrative_polygon':True}

def spatial_flight(scene,ax,frame,mapping,selected=None,width=3.1,direction=True):
    rg=attach_routes(mapping);dd=frame.copy();dd['task_group']=dd.route.map(rg)
    if selected is not None:dd=dd[dd.task_group.isin(selected)].copy()
    if dd.empty:return dd
    p=scene.xyz(dd,'p0');q=scene.xyz(dd,'p1');move=np.linalg.norm((q-p)*[1,1,.001],axis=1)>1e-9
    seg=np.stack([p[move],q[move]],axis=1);colors=[GC[x] for x in dd.loc[move,'task_group']]
    ax.add_collection3d(Line3DCollection(seg,colors='white',linewidths=width+1.3,zorder=7))
    ax.add_collection3d(Line3DCollection(seg,colors=colors,linewidths=width,zorder=8))
    if direction:
        options=[]
        for _,part in dd[dd.stage.eq('巡航')].groupby(['route','from_node','to_node'],sort=False):
            part=part.sort_values('start_s');p0=scene.xyz(part.iloc[[0]],'p0')[0];p1=scene.xyz(part.iloc[[-1]],'p1')[0]
            length=np.linalg.norm((p1-p0)[:2])
            if length>1:options.append((length,p0,p1,GC[part.task_group.iloc[0]]))
        mids=[]
        for _,p0,p1,col in sorted(options,key=lambda v:v[0],reverse=True):
            mid=(p0+p1)/2
            if any(np.linalg.norm((mid-x)[:2])<1.5 for x in mids):continue
            v=(p1-p0)/np.linalg.norm((p1-p0)[:2])*.45
            ax.quiver(*(mid-v/2),*v,color=col,arrow_length_ratio=.7,linewidth=2,length=1,normalize=False,zorder=10)
            mids.append(mid)
            if len(mids)==2:break
    for site in sorted(set(dd.from_node)|set(dd.to_node)):
        if site=='O01':continue
        xyz=scene.node[site];ax.scatter(*xyz,s=62,c=[GC[mapping[site]]],ec='white',lw=1.1,depthshade=False,zorder=13)
    # 孤立小组标清全部节点，大组仅标两处高点，保持读图空间。
    sites=sorted((set(dd.from_node)|set(dd.to_node))-{'O01'},key=lambda x:scene.node[x][2],reverse=True)
    for site in sites[:2]:
        xyz=scene.node[site];ax.text(xyz[0]+.10,xyz[1],xyz[2]+75,site,fontsize=16,color=INK,zorder=20)
    assert np.array_equal(scene.xyz(dd,'p0')[:,2],dd.p0_z.to_numpy(float))
    assert np.array_equal(scene.xyz(dd,'p1')[:,2],dd.p1_z.to_numpy(float))
    return dd

def plot_spatial():
    scene=Scene(m);stations=s.set_index('station_id');used=stations.loc[m.station_id.unique()]
    scene.zlim=(0.,float(np.ceil(max(scene.zlim[1],t.p0_z.max(),t.p1_z.max(),m.z.max(),used.cruise_z.max())/200)*200))
    tri=scenarios[scenarios.n_groups.eq(3)].scenario_id.iloc[0]
    two=mapping_for(recommended);three=mapping_for(tri)
    sets=[None,{'G1'},{'G2'},None,{'G1'},{'G2','G3'}]
    maps=[two]*3+[three]*3
    titles=['两组 · 最少增配','两组 · 第1组','两组 · 第2组','三组方案','三组 · 第1组','三组 · 第2、3组']
    fig=plt.figure(figsize=(22.7,13.8));gs=fig.add_gridspec(2,3,left=.005,right=.990,bottom=.057,top=.955,wspace=.025,hspace=-.025)
    records=[];intervals=[]
    for i,(mapping,selection,title) in enumerate(zip(maps,sets,titles)):
        ax=fig.add_subplot(gs[i//3,i%3],projection='3d');scene.base(ax,chr(97+i),title,axis_labels=i>=3)
        frame=spatial_flight(scene,ax,t,mapping,selection)
        relay_ids=set(frame.loc[~frame.direct_ok,'relay_mission_id']);missions=m[m.relay_mission_id.isin(relay_ids)]
        if len(missions):scene.relay(ax,missions,stations,flight=True)
        part=frame.copy();part['panel']=chr(97+i);part['scenario_id']=recommended if i<3 else tri;intervals.append(part)
        records.append({'panel':chr(97+i),'scenario':recommended if i<3 else tri,'groups':sorted(set(mapping.values())) if selection is None else sorted(selection),'routes':sorted(frame.route.unique()),'sorties':int(frame.route.nunique()),'relay_sorties':list(missions.relay_mission_id),'transport_seconds':float(frame.duration_s.sum())})
    handles=[Line2D([],[],c=GC[g],lw=3.4,label=f'第{k}组') for k,g in enumerate(GC,1)]
    handles += [Line2D([],[],c=INK,lw=2.8,label='中继航迹'),Line2D([],[],c=INK,lw=1.7,ls='--',label='回传链路'),Line2D([],[],marker='D',ls='',mfc=GC['G3'],mec='white',ms=10,label='中继悬停')]
    fig.legend(handles=handles,ncol=6,loc='upper center',bbox_to_anchor=(.5,1.002),frameon=False,fontsize=18,columnspacing=1.7,handlelength=2.1)
    cb=fig.colorbar(ScalarMappable(norm=scene.norm,cmap=CM),cax=fig.add_axes([.35,.041,.29,.013]),orientation='horizontal')
    cb.set_ticks([100,400,700,1000]);cb.ax.tick_params(labelsize=16,length=3,pad=3);cb.outline.set_linewidth(.8)
    fig.text(.653,.045,'地形海拔（米）',fontsize=18,va='center')
    out=OUT/'问题四_图02_六面板三维分组航线.png';save(fig,out)
    pd.concat(intervals,ignore_index=True).to_csv(OUT/'问题四_图02_绘图区间.csv',index=False,encoding='utf-8-sig')
    assert abs(records[1]['transport_seconds']+records[2]['transport_seconds']-records[0]['transport_seconds'])<1e-6
    assert abs(records[4]['transport_seconds']+records[5]['transport_seconds']-records[3]['transport_seconds'])<1e-6
    return out,{'panels':records,'actual_altitudes_unchanged':True,'visual_vertical_exaggeration':3.5,'all_limits':{'east_km':scene.xlim,'north_km':scene.ylim,'altitude_m':scene.zlim},'group_interval_partition_conserved':True}

f,b,m,c,t,s,provenance=load_v3(str(Q3/'问题三_V3_最终方案'),str(Q3/'问题三_V3_最终方案_精确'),str(Q3/'问题三_V3_站点.csv'))
groups=pd.read_csv(DATA/'site_groups.csv');scenarios=pd.read_csv(DATA/'scenarios.csv')
rr=scenarios[scenarios.n_groups.eq(2)&bools(scenarios.recommended)]
assert len(rr)==1,'两组必须明确唯一推荐方案';recommended=rr.scenario_id.iloc[0]
assert len(f)==20 and len(b)==80 and len(m)==3 and len(t)==443
outputs=[];diagnostics={}
for fn in [plot_planar,plot_spatial]:
    out,qa=fn()
    with Image.open(out) as im:im.load();size=list(im.size)
    outputs.append({'file':out.name,'size_px':size,'sha256':sha(out),'dpi':a.dpi,'decode':'PASS'})
    diagnostics[out.stem]=qa
    print(json.dumps(outputs[-1],ensure_ascii=False),flush=True)
report={'PASS':True,'source':provenance,'q4_input_sha256':{x:sha(DATA/x) for x in ['site_groups.csv','scenarios.csv']},'transport_sorties':20,'relay_sorties':3,'boxes':80,'source_intervals':443,'recommended_two_group_scenario':recommended,'dem_sha256':sha(Q3/'q1_flat'/'最终工作DEM.tif'),'no_new_transport_or_relay_optimization':True,'outputs':outputs,'details':diagnostics}
(OUT/'问题四_地图核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('问题四两张地图及真实航段核验完成。',flush=True)
