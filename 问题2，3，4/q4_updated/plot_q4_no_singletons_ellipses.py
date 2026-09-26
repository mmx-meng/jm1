"""Fixed-site spatial standard-deviation ellipses on the original Q3 DEM."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parent
Q3=ROOT.parent/'q3_source'
DATA=ROOT/'no_singletons'
OUT=DATA/'figures'
OUT.mkdir(exist_ok=True)
os.environ['Q2_CERTIFIED_PREFIX']=str(Q3/'问题三_时间主方案_运输')
sys.path.insert(0,str(Q3))
from 问题三_V3_基础绘图 import load_v3
from 问题三_V7_六面板三维山地航线 import Scene,CM,INK

plt.rcParams.update({'font.size':14,'axes.labelsize':14,'pdf.fonttype':42})
_,boxes,m,_,_,_,_=load_v3(str(Q3/'问题三_时间主方案'),
                          str(Q3/'问题三_时间主方案_精确'),
                          str(Q3/'问题三_V3_站点.csv'))
scene=Scene(m)
groups=pd.read_csv(DATA/'site_groups.csv')
COL={'G1':CM(.035),'G2':CM(.78),'G3':CM(.985)}
ids=['R2_不设单站组','R3_不设单站组']
titles=['两组 · 8区 / 7区','三组 · 4区 / 6区 / 5区']

def geometry(points,weight):
    mu=np.average(points,weights=weight,axis=0)
    centered=points-mu
    cov=(centered*weight[:,None]).T@centered/weight.sum()
    eig,vec=np.linalg.eigh(cov)
    eig=np.maximum(eig[::-1],0)
    vec=vec[:,::-1]
    angle=math.degrees(math.atan2(vec[1,0],vec[0,0]))
    return mu,np.sqrt(eig),angle

fig,axs=plt.subplots(1,2,figsize=(19.8,9.2),
    gridspec_kw={'left':.06,'right':.982,'bottom':.235,'top':.940,'wspace':.085})
records=[]
for i,(ax,sid) in enumerate(zip(axs,ids)):
    k=int(sid[1])
    ax.pcolormesh(scene.x,scene.y,scene.z,cmap=CM,norm=scene.norm,
                  shading='nearest',rasterized=True,zorder=0)
    ax.contour(scene.x[::5,::5],scene.y[::5,::5],scene.z[::5,::5],
               levels=[200,400,600,800,1000],colors=INK,
               linewidths=.6,alpha=.28,zorder=1)
    frame=groups[groups.scenario_id.eq(sid)]
    for gid,part in frame.groupby('group_id'):
        member=part.site_id.tolist()
        xy=np.asarray([scene.node[s][:2]for s in member])
        w=part.n_boxes.to_numpy(float)
        mu,sd,theta=geometry(xy,w)
        assert len(member)>=4 and sd.min()>0
        col=COL[gid]
        ax.add_patch(Ellipse(mu,2*sd[0],2*sd[1],angle=theta,
                             facecolor=col,edgecolor='none',alpha=.15,zorder=3))
        edge=Ellipse(mu,2*sd[0],2*sd[1],angle=theta,
                     facecolor='none',edgecolor=col,
                     linewidth=2.8,ls=(0,(5,2)),zorder=4)
        edge.set_path_effects([patheffects.Stroke(linewidth=4.6,foreground='white'),
                               patheffects.Normal()])
        ax.add_patch(edge)
        ax.scatter(xy[:,0],xy[:,1],s=80+9*w,c=[col],ec='white',lw=1.6,zorder=8)
        ax.scatter(*mu,s=115,marker='X',c=[col],ec='white',lw=1.1,zorder=9)
        displacement=((-112,16) if gid=='G3' else (12,-18) if k==3 and gid=='G2' else (7,8))
        mark=ax.annotate(f'{gid} · {len(member)}区 / {int(w.sum())}箱',mu,
                         xytext=displacement,textcoords='offset points',
                         fontsize=13,fontweight='bold',color=INK,zorder=12)
        mark.set_path_effects([patheffects.withStroke(linewidth=3.4,foreground='white')])
        records.append(dict(scenario_id=sid,group_id=gid,n_sites=len(member),n_boxes=int(w.sum()),
                            centroid_east_km=mu[0],centroid_north_km=mu[1],
                            sigma_major_km=sd[0],sigma_minor_km=sd[1],angle_deg=theta))
    ax.scatter(0,0,s=290,marker='*',c=INK,ec='white',lw=1.5,zorder=15)
    o=ax.annotate('O01',(0,0),xytext=(0,-20),textcoords='offset points',
                  ha='center',fontsize=14,fontweight='bold',zorder=16)
    o.set_path_effects([patheffects.withStroke(linewidth=3,foreground='white')])
    ax.set(xlim=scene.xlim,ylim=scene.ylim,aspect='equal')
    ax.set_xlabel('相对O01东向距离（km）')
    if i==0:ax.set_ylabel('相对O01北向距离（km）')
    ax.set_title(f'{chr(97+i)}  {titles[i]}',loc='left',fontsize=19,pad=9)
    ax.tick_params(labelsize=12,length=4)
fig.legend(handles=[Line2D([],[],c=COL[g],marker='o',mec='white',lw=2.8,
                            ms=9,label=f'第{g[1]}组') for g in COL]+
                    [Line2D([],[],c=INK,lw=2,ls='--',label='1σ空间离散椭圆')],
           loc='lower center',bbox_to_anchor=(.5,.090),ncol=4,
           frameon=False,fontsize=14)
bar=fig.colorbar(ScalarMappable(norm=scene.norm,cmap=CM),
                 cax=fig.add_axes([.365,.049,.27,.012]),orientation='horizontal')
bar.set_ticks([100,400,700,1000]);bar.ax.tick_params(labelsize=11,pad=1)
fig.text(.645,.056,'地形海拔（米）',fontsize=12,va='center')
png=OUT/'问题四_图05_原DEM分组空间椭圆.png'
pdf=png.with_suffix('.pdf')
fig.savefig(png,dpi=230,bbox_inches='tight',pad_inches=.12)
fig.savefig(pdf,dpi=160,bbox_inches='tight',pad_inches=.12)
plt.close(fig)
with Image.open(png) as im: im.verify()
pd.DataFrame(records).to_csv(DATA/'group_ellipses.csv',index=False,encoding='utf-8-sig')
print(png,flush=True)
