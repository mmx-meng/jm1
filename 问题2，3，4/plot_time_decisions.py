"""Replot time selection and resource evidence in the original Q3 DEM palette.

All coordinates and metrics come from preserved, independently checked CSVs.
The hillshade, hex grid and 3-D route figures use the original Q3/Q4 scripts.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

P=Path(__file__).resolve().parent
O=P/'figures';O.mkdir(exist_ok=True)
font=P/'q3_source/q1_flat/中文字体_绘图临时.otf'
font_manager.fontManager.addfont(str(font));CN=font_manager.FontProperties(fname=str(font)).get_name()
CM=plt.colormaps['RdYlBu_r']
INK='#171717';BLUE=CM(.075);RED=CM(.96);ORANGE=CM(.78);GRAY='#88939B'
plt.rcParams.update({'font.family':['STIXGeneral',CN],'font.size':14,
 'axes.titlesize':16,'axes.labelsize':13,'xtick.labelsize':11.5,
 'ytick.labelsize':11.5,'legend.fontsize':11.7,'axes.linewidth':1.2,
 'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
 'figure.facecolor':'white','savefig.facecolor':'white','axes.unicode_minus':False,
 'pdf.fonttype':42})

q2d=pd.read_csv(P/'q2_source/全局认证_冻结最终十九架次_逐箱.csv')
q2f=pd.read_csv(P/'q2_source/全局认证_冻结最终十九架次_架次.csv')
q2u_d=pd.read_csv(P/'q2_source/time_priority_3d/问题二_真正时间优先_22架次_逐箱.csv')
q2u_f=pd.read_csv(P/'q2_source/time_priority_3d/问题二_真正时间优先_22架次_架次.csv')
q3d=pd.read_csv(P/'q3_source/问题三_时间主方案_运输_逐箱.csv')
q3f=pd.read_csv(P/'q3_source/问题三_时间主方案_运输_架次.csv')
q3r=pd.read_csv(P/'q3_source/问题三_时间主方案_中继架次.csv')
oldd=pd.read_csv(P/'q3_legacy/deliveries.csv')
oldf=pd.read_csv(P/'q3_legacy/flights.csv')
oldr=pd.read_csv(P/'q3_legacy/relays.csv')
archive=pd.read_csv(P/'q3_source/original_candidates.csv')
q4=P/'q4_updated/data'

def save(fig,stem):
    path=O/(stem+'.png');pdf=O/(stem+'.pdf')
    tmp=O/(stem+'.tmp.png');ptmp=O/(stem+'.tmp.pdf')
    fig.savefig(tmp,dpi=245,bbox_inches='tight',pad_inches=.12)
    fig.savefig(ptmp,dpi=155,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    from PIL import Image
    with Image.open(tmp) as img:img.load()
    assert ptmp.stat().st_size>5000
    tmp.replace(path);ptmp.replace(pdf)
    return {'png':path.name,'pdf':pdf.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def label(ax,letter,title):
    ax.set_title(letter+'  '+title,loc='left',fontweight='bold',pad=12)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.14,zorder=0)

def count_steps(ax,d,kind,color,ls='-',alpha=1,labeltext=None):
    a=d[d.货箱编号.str.contains('-'+kind+'-')]
    if kind=='WAT':a=a[a.硬截止秒.notna()]
    events=np.sort(a.送达秒.to_numpy(dtype=float))/60
    xs=np.r_[0,events,125]
    ys=np.r_[0,np.arange(1,len(a)+1),len(a)]/len(a)*100
    ax.step(xs,ys,where='post',lw=2.5,c=color,ls=ls,alpha=alpha,
            label=labeltext,zorder=4)
    return len(a)

def time_evidence(which):
    if which==2:
        d=q2d;f=q2f;r=None;old=None;title='问题二 · 19次运输 / 80箱'
    else:
        d=q3d;f=q3f;r=q3r;old=oldd;title='问题三 · 22次运输 + 4次中继 / 80箱'
    fig=plt.figure(figsize=(19.6,11.3))
    gs=fig.add_gridspec(2,2,left=.065,right=.974,bottom=.09,top=.91,
                        hspace=.38,wspace=.21,height_ratios=[1,1.02])
    axm=fig.add_subplot(gs[0,0]);axw=fig.add_subplot(gs[0,1]);
    axh=fig.add_subplot(gs[1,0]);axt=fig.add_subplot(gs[1,1]);
    fig.suptitle(title,fontsize=21,fontweight='bold',y=.983)
    for ax,k,col,n,letter in [(axm,'MED',BLUE,'医疗物资','a'),
                              (axw,'WAT',ORANGE,'首批保障饮水','b')]:
        label(ax,letter,n+' · 分别计数')
        if old is not None:count_steps(ax,old,k,GRAY,ls='--',labeltext='原 20+3')
        nbox=count_steps(ax,d,k,col,labeltext='当前方案' if old is not None else '19架次')
        ax.axvline(60,c=RED,lw=1.4,ls=(0,(4,3)),alpha=.8)
        ax.set(xlim=(0,125),ylim=(-3,106),xlabel='任务开始后的分钟',ylabel='累计交付比例（%）')
        ax.set_xticks([0,30,60,90,120]);ax.set_yticks([0,25,50,75,100]);
        ax.text(.98,.05,f'{nbox} 箱',ha='right',va='bottom',transform=ax.transAxes,
                fontsize=12.5,color=col,fontweight='bold')
        ax.legend(loc='upper left',frameon=False)
    hard=d[d.硬截止秒.notna()].copy()
    hard['slack']=(hard.硬截止秒-hard.送达秒)/60
    hard=hard.sort_values(['slack','货箱编号']).reset_index(drop=True)
    label(axh,'c','31箱硬时限余量 · 升序')
    col=[RED if x<1e-8 else ORANGE if x<5 else BLUE for x in hard.slack]
    axh.bar(np.arange(len(hard)),hard.slack,color=col,width=.76,zorder=3)
    axh.axhline(5,c=GRAY,lw=1.1,ls='--')
    axh.set(xlim=(-.7,len(hard)-.3),xlabel='硬截止货箱（按余量排列）',ylabel='截止时间 − 实际交付（分钟）')
    axh.set_xticks([0,5,10,15,20,25,30])
    for idx in range(min(3,len(hard))):
        row=hard.iloc[idx]
        axh.annotate(row.货箱编号.replace('-01',''),(idx,row.slack),
                     xytext=(8,12+idx*15),textcoords='offset points',
                     fontsize=10.5,rotation=16,color=INK)
    label(axt,'d','末箱交付与全部返航分开报告')
    data=[('末箱交付',d.送达秒.max()/60,BLUE),('运输全部返航',f.返回秒.max()/60,ORANGE)]
    if r is not None:data.append(('中继全部返航',r.返回秒.max()/60,RED))
    pos=np.arange(len(data))[::-1]
    for y,(name,value,color) in zip(pos,data):
        axt.barh(y,value,height=.48,color=color,alpha=.92,zorder=3)
        axt.text(value-1,y,f'{value:.1f}分',ha='right',va='center',
                 color='white',fontsize=12.8,fontweight='bold')
    axt.set(yticks=pos,yticklabels=[v[0] for v in data],xlim=(0,143),xlabel='任务开始后的分钟')
    if old is not None:
        axt.scatter([oldd.送达秒.max()/60,oldf.返回秒.max()/60,oldr.返回秒.max()/60],
                    pos,c=GRAY,marker='D',s=55,zorder=7,label='原20+3');axt.legend(frameon=False,loc='lower right')
    fig.text(.5,.026,'医疗物资与首批保障饮水是两个独立集合；虚线标记任务启动后60分钟，所有交付与返航按同一时钟计算。',
             ha='center',fontsize=12)
    return save(fig,f'时效图0{which-1}_问题{["","","二","三"][which]}_逐箱硬余量及返航')

def tradeoff():
    s=archive[archive.加权延误.abs()<1e-9].copy()
    fig,ax=plt.subplots(figsize=(14.1,8.7))
    ref=6899/60;cap=1.05*ref
    ax.axvspan(0,ref*1.03,color=BLUE,alpha=.09,zorder=0,label='3%范围')
    ax.axvspan(ref*1.03,cap,color=ORANGE,alpha=.16,zorder=0,label='5%增量')
    ax.axvspan(cap,ref*1.10,color=GRAY,alpha=.09,zorder=0,label='10%增量')
    ax.scatter(s.联合完成秒/60,s.总能耗kWh,s=54,c=GRAY,alpha=.77,
               edgecolor='white',lw=.6,zorder=3,label='已核验零延误候选')
    candidates=[('最快候选 003',6899/60,66.3860045,ORANGE,'s'),
                ('原主方案 010',7366/60,64.988446,GRAY,'D'),
                ('当前 22+4',6936/60,66.21552118,BLUE,'o')]
    for lab,x,y,col,marker in candidates:
        ax.scatter(x,y,s=200,c=[col],marker=marker,ec='white',lw=1.6,zorder=8,label=lab)
        dx,dy={'最快候选 003':(-5,13),'原主方案 010':(8,9),'当前 22+4':(10,-21)}[lab]
        ax.annotate(lab,(x,y),xytext=(dx,dy),textcoords='offset points',
                    fontsize=13,fontweight='bold',color=INK)
    ax.axvline(cap,color=RED,ls='--',lw=1.65,zorder=2)
    ax.set(xlim=(110,148),ylim=(64.4,69.6),xlabel='联合完成时间（分钟，包含中继返航）',
           ylabel='总能耗（kWh）')
    label(ax,'','零延误候选 · 5%时间容许差与推荐方案')
    ax.legend(loc='upper right',frameon=False,ncol=2,fontsize=11)
    ax.text(.035,.055,'5%上限 7244秒；新方案 6936秒 / 66.216 kWh',
            transform=ax.transAxes,fontsize=13.4,fontweight='bold',color=INK)
    return save(fig,'时效图03_问题三候选_5百分比容许差')

def q2_policy_comparison():
    """Compare verified Q2 priorities with the existing Q2 palette and axes."""
    scenarios=[('少架次 · 19架次',q2d,q2f,BLUE),
               ('时间优先 · 22架次',q2u_d,q2u_f,ORANGE)]
    fig=plt.figure(figsize=(19.6,11.3))
    gs=fig.add_gridspec(2,2,left=.072,right=.974,bottom=.095,top=.905,
                        hspace=.39,wspace=.23)
    ax0=fig.add_subplot(gs[0,0]);ax1=fig.add_subplot(gs[0,1])
    ax2=fig.add_subplot(gs[1,0]);ax3=fig.add_subplot(gs[1,1])
    fig.suptitle('问题二 · 少架次与真正缩短完成时间',
                 fontsize=21,fontweight='bold',y=.982)
    label(ax0,'a','一小时硬截止货箱 · 累计送达')
    for name,d,f,color in scenarios:
        u=d[d['硬截止秒'].eq(3600)].sort_values('送达秒')
        assert len(u)==17
        xs=np.r_[0,u['送达秒'].to_numpy()/60,60]
        ys=np.r_[0,np.arange(1,len(u)+1),len(u)]
        ax0.step(xs,ys,where='post',color=color,lw=2.8,label=name)
    ax0.axvline(60,color=RED,ls=(0,(4,3)),lw=1.4)
    ax0.set(xlim=(0,61),ylim=(-.3,18),xlabel='任务开始后的分钟',ylabel='已交付货箱（17箱）')
    ax0.set_xticks([0,15,30,45,60]);ax0.set_yticks([0,4,8,12,17]);ax0.legend(frameon=False,loc='upper left')
    label(ax1,'b','逐箱硬时限余量 · 同一货箱对齐')
    a=q2d[q2d['硬截止秒'].notna()].set_index('货箱编号')
    b=q2u_d[q2u_d['硬截止秒'].notna()].set_index('货箱编号')
    ids=(a['硬截止秒']-a['送达秒']).sort_values().index
    x=np.arange(len(ids))
    for idx,(name,d,f,color) in enumerate(scenarios):
        part=(a if idx==0 else b).loc[ids]
        ax1.scatter(x+(idx-.5)*.14,(part['硬截止秒']-part['送达秒'])/60,
                    s=42,c=[color],ec='white',lw=.5,zorder=4,label=name)
    ax1.axhline(0,c=RED,lw=1.5,ls='--');ax1.set(xlim=(-1,31),
         xlabel='31箱硬截止货箱（按19架次余量排序）',ylabel='截止时间 − 交付时间（分钟）')
    ax1.legend(frameon=False,loc='upper right')
    label(ax2,'c','最紧急一小时货箱的末箱')
    for i,(name,d,f,color) in enumerate(scenarios):
        val=d.loc[d['硬截止秒'].eq(3600),'送达秒'].max()/60
        ax2.barh(1-i,val,height=.46,color=color,zorder=3)
        ax2.text(val-.6,1-i,f'{val:.1f} 分',ha='right',va='center',
                 color='white',fontsize=12.8,fontweight='bold')
    ax2.axvline(60,color=RED,lw=1.5,ls='--')
    ax2.set(yticks=[1,0],yticklabels=[x[0] for x in scenarios],xlim=(0,66),
            xlabel='最后一箱一小时硬截止货物送达（分钟）')
    label(ax3,'d','整体末箱交付与全部返航')
    vals=[]
    for i,(name,d,f,color) in enumerate(scenarios):
        y=1-i
        delivered=d['送达秒'].max()/60;returned=f['返回秒'].max()/60
        ax3.hlines(y,delivered,returned,lw=5,color=color,alpha=.8,zorder=3)
        ax3.scatter([delivered,returned],[y,y],s=120,c=[color,color],
                    edgecolor='white',lw=1.2,zorder=4)
        vals.append((name,delivered,returned))
        ax3.text(delivered-1,y+.13,f'{delivered:.1f}',fontsize=12,
                 color=INK,ha='right')
        ax3.text(returned+.7,y+.13,f'{returned:.1f}',fontsize=12,
                 color=INK,ha='left')
    ax3.set(yticks=[1,0],yticklabels=[x[0] for x in scenarios],
            xlim=(100,133),xlabel='送达末箱 ● ─ ● 全部返航（分钟）')
    fig.text(.5,.026,'两案均送达80箱、期望时间无延误；19架次为61.940 kWh，22架次为63.542 kWh。',
             ha='center',fontsize=12)
    return save(fig,'时效图05_问题二原色带_真实时间缩短对比')

def main():
    ans=[time_evidence(2),time_evidence(3),tradeoff(),q2_policy_comparison()]
    (O/'时效图_输入及输出核验.json').write_text(json.dumps({
       'status':'PASS','references':{'Q2_last_delivery_s':int(q2d.送达秒.max()),
        'Q2_last_return_s':int(q2f.返回秒.max()),'Q3_last_delivery_s':int(q3d.送达秒.max()),
        'Q2_urgent_last_return_s':int(q2u_f.返回秒.max()),
        'Q2_urgent_min_hard_slack_s':int((q2u_d['硬截止秒']-q2u_d['送达秒']).dropna().min()),
        'Q3_joint_return_s':int(max(q3f.返回秒.max(),q3r.返回秒.max())),
        'Q3_previous_joint_return_s':int(max(oldf.返回秒.max(),oldr.返回秒.max()))},
       'figures':ans},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(ans,ensure_ascii=False))
if __name__=='__main__':main()
