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
ap.add_argument('--data-dir',type=Path,default=ROOT/'no_singletons')
ap.add_argument('--q3-dir',type=Path,default=ROOT.parent/'q3_source')
ap.add_argument('--output-dir',type=Path,default=ROOT/'no_singletons'/'figures')
ap.add_argument('--dpi',type=int,default=220)
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

SOURCE=ROOT/'data'
hexes=pd.read_csv(SOURCE/'hex_terrain_time_B.csv')
sites=pd.read_csv(SOURCE/'site_terrain_workload_B.csv')
nodes=pd.read_csv(SOURCE/'nodes.csv').set_index('site_id')
groups=pd.read_csv(a.data_dir/'site_groups.csv')
for sid in ('R2_不设单站组','R3_不设单站组'):
    look=groups[groups.scenario_id.eq(sid)].set_index('site_id').group_id.to_dict()
    assert len(look)==15 and groups[groups.scenario_id.eq(sid)].group_id.nunique()==int(sid[1])
    hexes['group_'+sid]=hexes.nearest_site_id.map(look)
assert len(hexes)==166 and len(sites)==15
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
            '两组 · 8区 / 7区 · 地形困难任务',
            '三组 · 4区 / 6区 / 5区 · 地形困难任务']
    for i,ax in enumerate(axs.flat):
        make_map(ax,titles[i],chr(97+i),colored=i==0)
        if i==0:points(ax,all_labels=True)
        else:time_hex(ax,'R2_不设单站组' if i==2 else 'R3_不设单站组' if i==3 else None)
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
    return save(fig,'问题四_图04_原配色DEM飞行时间与分组蜂巢图')

if __name__=='__main__':
    print(figure_hex(),flush=True)
