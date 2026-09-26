"""Updated Q4 hex and resource figures, using the original Q3 DEM palette.

Figure 14: DEM-computed B-model reference time on a spatial hex sampling mesh.
Figure 15: eight dedicated resource types plus true simultaneous-use steps.
The comparative fairness and inventory summary is drawn by plot_time_decisions.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--data-dir',type=Path,default=ROOT/'data')
ap.add_argument('--q3-dir',type=Path,default=ROOT/'q3_source')
ap.add_argument('--output-dir',type=Path,default=ROOT/'results')
ap.add_argument('--dpi',type=int,default=300)
a=ap.parse_args()
a.output_dir.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(a.q3_dir))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize,LinearSegmentedColormap,LightSource
from matplotlib.collections import PolyCollection,LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Circle
import numpy as np
import pandas as pd
from PIL import Image
from q1_flat.精确算法 import read_dem

font_manager.fontManager.addfont(str(a.q3_dir/'q1_flat'/'中文字体_绘图临时.otf'))
CN=font_manager.FontProperties(fname=str(a.q3_dir/'q1_flat'/'中文字体_绘图临时.otf')).get_name()
plt.rcParams.update({'font.family':['STIXGeneral',CN],'axes.unicode_minus':False,
    'pdf.fonttype':42,'font.size':13,'axes.titlesize':17,'axes.labelsize':14,
    'xtick.labelsize':11,'ytick.labelsize':11,'text.color':'#151515',
    'figure.facecolor':'white','savefig.facecolor':'white'})
TERRAIN=plt.colormaps['RdYlBu_r']
GC={'G1':TERRAIN(.035),'G2':TERRAIN(.78),'G3':TERRAIN(.985)}
SCOL={'K2_01':'#7898BC','K2_02':'#F07847','K2_03':'#31538C','K3_01':'#A50026'}
TCMAP=LinearSegmentedColormap.from_list('DEM_reference_time',
    ['#E7F3F8','#9FCFD8','#F8D18C','#F07847','#A50026'])

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(fig,name):
    p=a.output_dir/name
    png=a.output_dir/(name+'.png');pdf=a.output_dir/(name+'.pdf')
    tp=a.output_dir/(name+'.tmp.png');td=a.output_dir/(name+'.tmp.pdf')
    fig.savefig(tp,dpi=a.dpi,bbox_inches='tight',pad_inches=.12)
    fig.savefig(td,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    with Image.open(tp) as im:im.verify()
    with Image.open(tp) as im:im.load();dims=list(im.size)
    assert td.stat().st_size>5000
    tp.replace(png);td.replace(pdf)
    return {'png':png.name,'pdf':pdf.name,'px':dims,'sha256':digest(png)}

hexes=pd.read_csv(a.data_dir/'hex_terrain_time_B.csv')
sites=pd.read_csv(a.data_dir/'site_terrain_workload_B.csv')
nodes=pd.read_csv(a.data_dir/'nodes.csv').set_index('site_id')
groups=pd.read_csv(a.data_dir/'site_groups.csv')
blocks=pd.read_csv(a.data_dir/'transport_components.csv')
links=pd.read_csv(a.data_dir/'must_link_edges.csv')
res=pd.read_csv(a.data_dir/'scenario_resources.csv')
gres=pd.read_csv(a.data_dir/'group_resources.csv')
times=pd.read_csv(a.data_dir/'intervals.csv')
sumg=pd.read_csv(a.data_dir/'group_summary.csv')
trade=pd.read_csv(a.data_dir/'scenario_spatial_tradeoff_B.csv')
blockwork=pd.read_csv(a.data_dir/'q4_hypergraph_blocks.csv')
fairbound=pd.read_csv(a.data_dir/'q4_fairness_lower_bound.csv')
certificate=json.loads((a.data_dir/'q4_cpsat_certificate.json').read_text())
qsp=json.loads((a.data_dir/'spatial_qa.json').read_text())
assert qsp['status']=='PASS' and len(hexes)==166 and len(sites)==15
assert len(trade)==4 and len(res)==32 and len(gres)==72
assert len(blocks.block_id.unique())==10
assert certificate['status']=='PASS' and all(s['all_stages_proved_OPTIMAL']for s in certificate['result'])
assert len(blockwork)==3 and len(fairbound)==2
assert len(links[links.relation.eq('同一运输架次')])>0
assert set(trade.scenario_id)=={'K2_01','K2_02','K2_03','K3_01'}
audit=float(gres.audit_s.unique().item())
assert audit==qsp['resource_audit_s']
source_hash={name:digest(a.data_dir/name) for name in
    ['hex_terrain_time_B.csv','site_terrain_workload_B.csv',
     'group_resources.csv','scenario_resources.csv','scenario_spatial_tradeoff_B.csv']}

o=nodes.loc['O01'];lon0=float(o.longitude);lat0=float(o.latitude)
def local(lon,lat):
    return np.column_stack(((np.asarray(lon)-lon0)*111.32*math.cos(math.radians(lat0)),
                            (np.asarray(lat)-lat0)*111.132))
def place_sites():
    df=sites.join(nodes[['longitude','latitude']],on='site_id',validate='one_to_one')
    xy=local(df.longitude,df.latitude)
    return df,xy

positions,site_xy=place_sites()
hex_xy=hexes[['center_x_km','center_y_km']].to_numpy()
radius=.51
vertices=np.array([[radius*math.cos(i*math.pi/3),radius*math.sin(i*math.pi/3)]
                   for i in range(6)])
polys=hex_xy[:,None,:]+vertices[None,:,:]
lim=[min(hex_xy[:,0].min()-.7,site_xy[:,0].min()-1.1),
     max(hex_xy[:,0].max()+.7,site_xy[:,0].max()+1.1),
     min(hex_xy[:,1].min()-.7,site_xy[:,1].min()-1.1),
     max(hex_xy[:,1].max()+.7,site_xy[:,1].max()+1.1)]

def relief():
    dem,left,top,dx,dy=read_dem(a.q3_dir/'q1_flat'/'最终工作DEM.tif')
    def longitude(x):return lon0+x/(111.32*math.cos(math.radians(lat0)))
    def latitude(y):return lat0+y/111.132
    c0=max(0,int((longitude(lim[0]-.4)-left)/dx))
    c1=min(dem.shape[1],int((longitude(lim[1]+.4)-left)/dx)+2)
    r0=max(0,int((top-latitude(lim[3]+.4))/dy))
    r1=min(dem.shape[0],int((top-latitude(lim[2]-.4))/dy)+2)
    z=dem[r0:r1,c0:c1]
    shade=LightSource(azdeg=305,altdeg=47).hillshade(z,dx=30,dy=30,vert_exag=1.0)
    colored=np.clip(TERRAIN(Normalize(0,1150)(z))[...,:3]*(.58+.54*shade[...,None]),0,1)
    gray=np.repeat((.89+.11*shade)[...,None],3,axis=2)
    ext=[(left+c0*dx-lon0)*111.32*math.cos(math.radians(lat0)),
         (left+c1*dx-lon0)*111.32*math.cos(math.radians(lat0)),
         (top-r1*dy-lat0)*111.132,(top-r0*dy-lat0)*111.132]
    return colored,gray,ext
DEM_COLORED,DEM_GRAY,DEM_EXT=relief()

def halo(t,width=2.8):t.set_path_effects([patheffects.withStroke(linewidth=width,foreground='white')])
def bounds(labels):
    edges={}
    for j,p in enumerate(polys):
        for u,v in zip(p,np.roll(p,-1,axis=0)):
            k=tuple(sorted((tuple(np.round(u,5)),tuple(np.round(v,5)))))
            edges.setdefault(k,[]).append(j)
    return [edge for edge,idx in edges.items()if len(idx)==2 and labels[idx[0]]!=labels[idx[1]]]

def make_map(ax,title,letter,colored=False):
    ax.imshow(DEM_COLORED if colored else DEM_GRAY,extent=DEM_EXT,
              origin='upper',interpolation='nearest',zorder=0)
    ax.set(xlim=lim[:2],ylim=lim[2:],aspect='equal')
    ax.set_title(f'{letter}  {title}',loc='left',fontweight='bold',pad=8)
    ax.tick_params(length=4,width=1.1)
    for sp in ax.spines.values():sp.set_linewidth(1.2)

def points(ax,sid=None,show_values=False,all_labels=False):
    mapping={} if sid is None else groups[groups.scenario_id.eq(sid)].set_index('site_id').group_id.to_dict()
    maxw=float(positions.fixed_transport_workload_allocated_h.max())
    for j,row in positions.reset_index(drop=True).iterrows():
        xy=site_xy[j]
        col='white' if sid is None else GC[mapping[row.site_id]]
        size=50+160*float(row.fixed_transport_workload_allocated_h)/maxw
        ax.scatter(*xy,s=size,c=[col],ec='#202020' if sid is None else 'white',
                   lw=1.35,zorder=12)
        if not all_labels and row.site_id not in ('S002','S003','S006','S007','S011'):
            continue
        label=row.site_id
        if show_values:label+=f' · {row.O01_B_one_way_empty_min:.1f}分'
        ox=-7 if row.site_id in ('S005','S007','S009','S011') else 8
        oy=-10 if row.site_id in ('S006','S010','S011','S013','S014') else 7
        mark=ax.annotate(label,xy,xytext=(ox,oy),textcoords='offset points',
                         ha='right' if ox<0 else 'left',fontsize=10.7,
                         fontweight='bold',zorder=15)
        halo(mark)
    ax.scatter(0,0,s=215,marker='*',c='#111111',ec='white',lw=1.3,zorder=18)
    mark=ax.annotate('O01',(0,0),xytext=(0,-22),textcoords='offset points',
                     ha='center',fontsize=12,fontweight='bold',zorder=19)
    halo(mark)

def time_hex(ax,sid=None):
    c=TCMAP(Normalize(0,14)(hexes.B_one_way_empty_min.to_numpy()))
    c[:,-1]=.88
    ax.add_collection(PolyCollection(polys,facecolors=c,edgecolors='#53636A',
                     linewidths=.22,zorder=3))
    if sid:
        labels=hexes['group_'+sid].to_numpy()
        ax.add_collection(LineCollection(bounds(labels),colors='#171717',
                          linewidths=1.4,alpha=.85,zorder=7))
        ax.add_patch(Circle((0,0),radius=.43,facecolor='white',
                            ec='none',alpha=.80,zorder=9))
    points(ax,sid,show_values=sid is None)

def figure_hex():
    fig,axs=plt.subplots(2,2,figsize=(18.7,12.9),gridspec_kw={
        'left':.065,'right':.982,'bottom':.20,'top':.96,'wspace':.10,'hspace':.17})
    titles=['修正DEM、15个服务点和固定运输负担',
            'B型参考机 · O01到蜂巢的地形飞行时间',
            '两组 K2_03 · 固定困难任务的归属',
            '三组 K3_01 · 固定困难任务的归属']
    for i,ax in enumerate(axs.flat):
        make_map(ax,titles[i],chr(97+i),colored=i==0)
        if i==0:points(ax,all_labels=True)
        else:time_hex(ax,'K2_03' if i==2 else 'K3_01' if i==3 else None)
        if i%2==0:ax.set_ylabel('相对O01北向距离（km）')
        if i>=2:ax.set_xlabel('相对O01东向距离（km）')
    handles=[Patch(fc=GC[g],label=f'第{i}组') for i,g in enumerate(GC,1)]
    handles.extend([Line2D([],[],ls='',marker='o',ms=7,mfc='white',mec='#333333',label='圆点大小：运输工时'),
                    Line2D([],[],ls='',marker='*',ms=12,color='#171717',label='O01')])
    fig.legend(handles=handles,ncol=5,loc='lower center',bbox_to_anchor=(.5,.127),
               fontsize=12.0,frameon=False)
    for x,norm,cmap,label in [(.20,Normalize(0,1150),TERRAIN,'DEM海拔（m）'),
                              (.61,Normalize(0,14),TCMAP,'B型空载参考单程（分钟）')]:
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),
                        cax=fig.add_axes([x,.055,.20,.010]),orientation='horizontal')
        cb.ax.tick_params(labelsize=10.5,pad=2)
        if x>.5:cb.set_ticks([0,3,6,9,12,14])
        fig.text(x+.10,.083,label,ha='center',fontsize=11.2)
    return save(fig,'问题四_图14_DEM参考飞行时间与分组负担蜂巢图')

KEYS=['transport_drone_A','transport_drone_B','transport_drone_C',
      'transport_battery_A','transport_battery_B','transport_battery_C',
      'relay_drone_R','relay_component_R']
SHORT=['A型运输机','B型运输机','C型运输机','A型共享电池',
       'B型共享电池','C型共享电池','中继机','中继能源组件']

def timeline(ax,sid,key,title):
    d=times[times.scenario_id.eq(sid)&times.resource_key.eq(key)]
    edges=np.r_[0.,np.sort(np.unique(np.r_[d.start_s.to_numpy(float),d.end_s.to_numpy(float)])),audit]
    edges=np.unique(edges)
    counts={}
    for gid in ['G1','G2','G3']:
        v=d[d.group_id.eq(gid)]
        counts[gid]=np.array([sum((v.start_s<=s)&(v.end_s>s))for s in edges[:-1]],float)
    total=sum(counts.values())
    records=res[(res.scenario_id.eq(sid))&res.resource_key.eq(key)].iloc[0]
    assert max(total)==records.global_peak
    assert sum(max(v)for v in counts.values())==records.sum_required
    for gid,ys in counts.items():
        if not max(ys):continue
        ax.step(edges/60.,np.r_[ys,0],where='post',c=GC[gid],lw=2.1,
                label=f'第{gid[1]}组峰值{int(max(ys))}')
    ax.step(edges/60.,np.r_[total,0],where='post',c='#151515',lw=1.5,
            ls='--',alpha=.86,label=f'全局峰值{int(max(total))}')
    ax.axhline(records.inventory,c='#A50026',lw=1.4,ls=(0,(2,3)),alpha=.70)
    ax.set(xlim=(0,math.ceil(audit/3600)*60),ylim=(-.10,max(5.,float(records.sum_required)+.6)),
           yticks=range(6),xlabel='问题三固定时轴（分钟）',ylabel='同时占用（件）')
    ax.set_title(f'{title}：专属{int(records.sum_required)} / 库存{int(records.inventory)}',
                 loc='left',fontsize=14.2,fontweight='bold')
    ax.tick_params(labelsize=10.4)
    ax.grid(axis='y',alpha=.16)
    ax.spines[['top','right']].set_visible(False)

def figure_resource():
    fig=plt.figure(figsize=(19.9,12.4))
    gs=fig.add_gridspec(2,3,left=.113,right=.982,bottom=.120,top=.92,
                        width_ratios=[1.08,1.0,1.0],wspace=.29,hspace=.34)
    main=fig.add_subplot(gs[:,0]);small=[fig.add_subplot(gs[i,j]) for i in range(2)for j in (1,2)]
    for j,(key,label) in enumerate(zip(KEYS,SHORT)):
        y=7-j
        r=res[(res.resource_key.eq(key))&res.scenario_id.eq('K2_03')].iloc[0]
        t=res[(res.resource_key.eq(key))&res.scenario_id.eq('K3_01')].iloc[0]
        assert r.global_peak==t.global_peak and r.inventory==t.inventory
        main.plot([0,r.inventory],[y,y],c='#D4D8DA',lw=7,zorder=1,solid_capstyle='round')
        main.scatter(r.global_peak,y,s=74,marker='s',c='#151515',zorder=5)
        main.scatter(r.sum_required,y+.18,s=115,marker='o',c='#31538C',
                     ec='white',lw=.8,zorder=7)
        main.scatter(t.sum_required,y-.18,s=120,marker='D',c='#F07847',
                     ec='white',lw=.8,zorder=7)
        main.scatter(r.inventory,y,s=120,marker='|',c='#A50026',lw=2.5,zorder=9)
    main.set(yticks=range(8),yticklabels=SHORT[::-1],xlim=(-.25,7.0),
             xticks=range(7),xlabel='独立配置件数 / 分类库存')
    main.set_title('a  八类资源：共享峰值、两组、三组与库存',loc='left',
                   fontsize=16.2,fontweight='bold',pad=13)
    main.grid(axis='x',alpha=.15)
    main.tick_params(length=0,labelsize=11.7)
    main.spines[['top','right']].set_visible(False)
    for ax,sp in zip(small,[('K3_01','transport_drone_B','b  三组 · B型运输机'),
                            ('K2_03','transport_drone_C','c  两组 · C型运输机'),
                            ('K3_01','transport_battery_B','d  三组 · B型电池'),
                            ('K2_03','transport_battery_C','e  两组 · C型电池')]):
        timeline(ax,*sp)
    hs=[Line2D([],[],ls='',marker='s',c='#151515',ms=8,label='全局共享峰值'),
        Line2D([],[],ls='',marker='o',c='#31538C',ms=9,label='两组专属需求'),
        Line2D([],[],ls='',marker='D',c='#F07847',ms=8,label='三组专属需求'),
        Line2D([],[],ls='',marker='|',c='#A50026',ms=13,label='分类库存'),
        Line2D([],[],c=GC['G1'],lw=2.3,label='第1组占用'),
        Line2D([],[],c=GC['G2'],lw=2.3,label='第2组占用'),
        Line2D([],[],c=GC['G3'],lw=2.3,label='第3组占用')]
    fig.legend(handles=hs,ncol=7,loc='upper center',bbox_to_anchor=(.5,.994),
               frameon=False,fontsize=10.7,columnspacing=1.1)
    fig.text(.5,.039,'阶梯为问题三真实任务的同时占用；专属配置取各组峰值之和，峰值不要求同时发生。统一核算至全部返航、充电与中继周转完成。',
             ha='center',fontsize=11.5)
    return save(fig,'问题四_图15_八类资源配置与B_C型占用阶梯')

def main():
    out=[figure_hex(),figure_resource()]
    assert len(out)==2
    qa={'status':'PASS','revised_q3_input_csv_byte_identical':True,
        'reference_model':'B','hex_values_source':'corrected DEM supercover cell maxima and Q3 climb, cruise, descent speeds',
        'transport_routes_fixed':22,'relay_routes_fixed':4,'boxes_fixed':80,
        'common_audit_s':audit,'all_four_strict_partitions':True,
        'figure_14_hex_sample_not_new_delivery_tasks':True,
        'figure_15_eight_types_exact_peak_scan':True,
        'site_delivery_times_not_changed_by_grouping':True,
        'input_hashes':source_hash,'outputs':out}
    (a.output_dir/'问题四_正文四组图补充核验.json').write_text(
        json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','outputs':out},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
