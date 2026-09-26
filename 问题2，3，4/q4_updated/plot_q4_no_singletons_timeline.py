"""Original-palette relay/transport occupancy steps from fixed task intervals."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image

ROOT=Path(__file__).resolve().parent
Q3=ROOT.parent/'q3_source'
OUT=ROOT/'no_singletons'/'figures'
OUT.mkdir(exist_ok=True)
FONT=Q3/'q1_flat'/'中文字体_绘图临时.otf'
font_manager.fontManager.addfont(str(FONT))
CN=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.family':['STIXGeneral',CN], 'font.size':13,
                     'axes.titlesize':16,'axes.labelsize':13,'pdf.fonttype':42})
CM=plt.colormaps['RdYlBu_r']
colors={'G1':CM(.035),'G2':CM(.78),'G3':CM(.985)}
intervals=pd.read_csv(ROOT/'no_singletons'/'intervals.csv')
resources=pd.read_csv(ROOT/'no_singletons'/'scenario_resources.csv')
cases=[('R2_不设单站组','relay_drone_R','两组 · 中继无人机'),
       ('R3_不设单站组','relay_drone_R','三组 · 中继无人机'),
       ('R2_不设单站组','transport_drone_C','两组 · C型运输机'),
       ('R3_不设单站组','transport_drone_C','三组 · C型运输机')]
fig,axs=plt.subplots(2,2,figsize=(17.3,10.5),
    gridspec_kw={'left':.075,'right':.98,'bottom':.15,'top':.91,'wspace':.15,'hspace':.26})
for i,((sid,key,title),ax) in enumerate(zip(cases,axs.flat)):
    d=intervals[intervals.scenario_id.eq(sid)&intervals.resource_key.eq(key)]
    st=float(d.start_s.min());end=float(d.end_s.max())
    ticks=np.unique(np.r_[0.,d.start_s.to_numpy(float),d.end_s.to_numpy(float),end])
    assigned=[]
    for gid in ('G1','G2','G3'):
        sub=d[d.group_id.eq(gid)]
        if sub.empty:continue
        y=np.array([sum((sub.start_s<=x)&(sub.end_s>x)) for x in ticks[:-1]])
        assigned.append(y)
        ax.step(ticks/60,np.r_[y,0],where='post',lw=2.45,
                c=colors[gid],label=f'第{gid[1]}组峰值{y.max()}')
    series=sum(assigned)
    r=resources[resources.scenario_id.eq(sid)&resources.resource_key.eq(key)].iloc[0]
    assert sum(y.max() for y in assigned)==r.sum_required
    ax.step(ticks/60,np.r_[series,0],where='post',lw=1.65,ls='--',
            c='#202026',label='实际总占用')
    ax.axhline(r.inventory,lw=1.35,c=CM(.985),ls=(0,(3,3)))
    ax.set(xlim=(0,140),ylim=(-.15,max(6,r.sum_required+.6)),
           xlabel='问题三固定时钟（分钟）',ylabel='同时占用（架）')
    ax.set_title(f'{chr(97+i)}  {title} · 专属{int(r.sum_required)} / 库存{int(r.inventory)}',
                 loc='left',pad=8,fontweight='bold')
    ax.grid(axis='y',color='#C4CCD1',alpha=.42)
    ax.spines[['top','right']].set_visible(False)
handles=[Line2D([],[],c=colors[g],lw=2.8,label=f'第{g[1]}组')for g in colors]
handles += [Line2D([],[],c='#202026',lw=1.8,ls='--',label='同时占用总量'),
            Line2D([],[],c=CM(.985),lw=1.5,ls='--',label='现有库存')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.035),
           ncol=5,frameon=False,fontsize=13.5)
png=OUT/'问题四_图06_原版时序资源峰值.png'
pdf=png.with_suffix('.pdf')
fig.savefig(png,dpi=220,bbox_inches='tight',pad_inches=.12)
fig.savefig(pdf,dpi=150,bbox_inches='tight',pad_inches=.12)
plt.close(fig)
with Image.open(png) as im:im.verify()
print(png)
