"""问题四：严格分组的资源缺口、任务负载与真实资源时序。

数据由 compute_q4.py 产生；所有图均使用冻结的真实问题三任务。
分组改变固定资源配额，不能改变同一批任务的实际占用时间序列。
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
CM = plt.colormaps['RdYlBu_r']
BLUE, ORANGE, RED = CM(.10), CM(.74), CM(.96)


def setup(q3_dir=None):
    options = [ROOT/'中文字体_绘图临时.otf',
               ROOT/'data'/'中文字体_绘图临时.otf',
               ROOT/'q3_source'/'q1_flat'/'中文字体_绘图临时.otf',
               ROOT.parent/'q3_v10_execution'/'q1_flat'/'中文字体_绘图临时.otf']
    if q3_dir is not None:
        options.insert(0,Path(q3_dir)/'q1_flat'/'中文字体_绘图临时.otf')
    cn = 'DejaVu Sans'
    for f in options:
        if f.exists():
            font_manager.fontManager.addfont(str(f))
            cn = font_manager.FontProperties(fname=str(f)).get_name()
            break
    plt.rcParams.update({'font.family': ['STIXGeneral', cn], 'font.size': 18,
        'axes.titlesize': 22, 'axes.labelsize': 18.5, 'xtick.labelsize': 16.5,
        'ytick.labelsize': 16.5, 'legend.fontsize': 16, 'axes.linewidth': 1.7,
        'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black',
        'ytick.color': 'black', 'figure.facecolor': 'white',
        'savefig.facecolor': 'white', 'axes.unicode_minus': False})


def panel(ax, letter, title):
    ax.set_title(f'{letter}  {title}', loc='left', fontweight='bold', pad=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(width=1.4, length=5)
    ax.grid(False)


def gradbar(ax, x, y, w, h, lo=.07, hi=.95, vertical=False, z=2):
    if w <= 0 or h <= 0:
        return
    r = Rectangle((x, y), w, h, facecolor='none', edgecolor='black', lw=.75, zorder=z+.2)
    ax.add_patch(r)
    a = np.linspace(lo, hi, 160)
    im = ax.imshow(a[:,None] if vertical else a[None,:], origin='lower',
        extent=[x, x+w, y, y+h], aspect='auto', cmap=CM, vmin=0, vmax=1, zorder=z)
    im.set_clip_path(r)


def gradline(ax, x, y, lo=.08, hi=.95, lw=4, step=False):
    x, y = np.asarray(x,float), np.asarray(y,float)
    if step:
        x, y = np.repeat(x,2)[1:], np.repeat(y,2)[:-1]
    if len(x) < 2:
        return
    seg = np.stack([np.c_[x[:-1],y[:-1]],np.c_[x[1:],y[1:]]],axis=1)
    ax.add_collection(LineCollection(seg, colors=CM(np.linspace(lo,hi,len(seg))), lw=lw, zorder=4))


def atomic_png(fig, path, dpi):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.stem+'.tmp.png')
    fig.savefig(tmp,dpi=dpi,bbox_inches='tight',pad_inches=.14)
    plt.close(fig)
    from PIL import Image
    with Image.open(tmp) as im:
        im.load()
        shape = im.size
    tmp.replace(path)
    print(f'{path.name}: {shape[0]} × {shape[1]}',flush=True)
    return {'file':path.name,'width':shape[0],'height':shape[1],
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def truth(s):
    return s.astype(str).str.lower().isin(['true','1','yes','是'])


def load(data):
    names = ['scenarios','group_resources','scenario_resources','intervals','group_summary']
    frames = {n:pd.read_csv(data/(n+'.csv')) for n in names}
    sc = frames['scenarios']
    rec = sc[(sc.n_groups == 2)&truth(sc.recommended)]
    assert len(rec)==1, '必须只有一个两组推荐分区'
    three = sc[sc.n_groups == 3]
    assert len(three)==1, '当前必须同组关系下应只有一个三组分区'
    frames['two_id'],frames['three_id'] = rec.iloc[0].scenario_id,three.iloc[0].scenario_id
    order = frames['group_resources'].resource_key.drop_duplicates().tolist()
    assert len(order)==8
    frames['order'] = order
    inv = frames['intervals']
    assert (inv.end_s>=inv.start_s).all()
    assert not truth(sc.feasible_inventory).any(), '绘图口径按现库存均有缺口，需检查新计算结果'
    return frames


def labels(fr):
    # 简化面板中的长资源名，来源与含义保留在数据表。
    out=[]
    for _,r in fr.iterrows():
        model=str(r.get('model',''))
        text=str(r.resource_label)
        if model in ['A','B','C']:
            text=f'{model}型电池' if '电池' in text else f'{model}型运输机'
        elif '组件' in text:
            text='中继能源组件'
        else:
            text='中继无人机'
        out.append(text)
    return out


def resource_frame(f, sid):
    g=f['group_resources'];g=g[g.scenario_id==sid]
    rows=[]
    for key in f['order']:
        a=g[g.resource_key==key]
        rows.append(dict(resource_key=key,resource_label=a.iloc[0].resource_label,
            model=a.iloc[0].model,required=int(a.required_count.sum()),
            inventory=int(a.iloc[0].inventory),occupancy_s=float(a.occupancy_s.sum())))
    return pd.DataFrame(rows)


def grouped_sites(row):
    sites=str(row.sites)
    if int(row.n_sites)>2:
        return f'其余{int(row.n_sites)}区'
    # 支持 JSON、分号或逗号形式的源表。
    import re
    return '、'.join(re.findall(r'S\d{3}',sites)) or str(row.group_id)


def figure_statistics(f, output, dpi):
    two,three=f['two_id'],f['three_id']
    rt,rr=resource_frame(f,two),resource_frame(f,three)
    gs=f['group_summary']
    fig,axs=plt.subplots(2,3,figsize=(23.5,13.2))
    fig.subplots_adjust(left=.08,right=.985,bottom=.075,top=.95,wspace=.35,hspace=.36)
    for ax,fr,letter,title in [(axs[0,0],rt,'a','两组资源配置'),(axs[0,1],rr,'b','三组资源配置')]:
        panel(ax,letter,title)
        for i,r in fr.iterrows():
            gradline(ax,np.linspace(0,r.required,32),np.full(32,i),lw=5)
            ax.scatter(r.required,i,s=105,c=[CM(.96 if r.required>r.inventory else .13)],ec='black',lw=.7,zorder=5)
            ax.scatter(r.inventory,i,s=235,marker='|',c='black',lw=2.2,zorder=6)
            txt=f'{r.required} / {r.inventory}'
            ax.text(7.20,i,txt,va='center',ha='right',fontsize=17,
                    color='black',fontweight='bold' if r.required>r.inventory else 'normal')
            if r.required>r.inventory:
                ax.annotate(f'+{r.required-r.inventory}',xy=((r.inventory+r.required)/2,i),
                    xytext=(0,12),textcoords='offset points',ha='center',color=RED,fontsize=15,fontweight='bold')
        ax.set_yticks(range(len(fr)),labels(fr));ax.set_ylim(8.15,-.6)
        ax.set_xlim(0,7.40);ax.set_xticks([0,2,4,6]);ax.set_xlabel('最低配置数量')
        ax.text(.985,1.01,'需求 / 库存',ha='right',transform=ax.transAxes,fontsize=15.5)
        ax.legend(handles=[Line2D([],[],marker='|',color='black',lw=0,markersize=16,label='分类库存'),
                           Line2D([],[],marker='o',color=RED,lw=0,markersize=8,label='资源缺口')],
                  loc='lower right',frameon=False,ncol=2,fontsize=14.5,handletextpad=.35,columnspacing=.8)
    ax=axs[0,2];panel(ax,'c','两组候选比较')
    ss=f['scenarios'];ss=ss[ss.n_groups==2].copy().sort_values(['shortage_count','total_required','transport_workload_gini'])
    yl=[]
    for i,(_,s) in enumerate(ss.iterrows()):
        gg=gs[gs.scenario_id==s.scenario_id]
        small=gg.sort_values('n_sites').iloc[0]
        yl.append(grouped_sites(small))
        gradbar(ax,0,i-.20,float(s.shortage_count),.40,.08,.94)
        ax.scatter(s.shortage_count,i,s=240 if str(s.scenario_id)==str(two) else 95,
                   marker='*' if str(s.scenario_id)==str(two) else 'o',c=[RED],ec='white',lw=.8,zorder=5)
        ax.text(6.20,i,f'{int(s.shortage_count)}件   {s.transport_workload_gini_normalized:.3f}',va='center',ha='right',fontsize=17)
    ax.set_yticks(range(len(ss)),yl);ax.set_ylim(len(ss)-.25,-.75)
    ax.set_xlim(0,6.45);ax.set_xticks([0,1,2,3,4]);ax.set_xlabel('各类资源缺口之和')
    ax.text(.98,1.01,'增配量   归一化运输工时Gini',transform=ax.transAxes,ha='right',fontsize=15)
    ax.legend(handles=[Line2D([],[],marker='*',color=RED,lw=0,markersize=13,label='最小缺口下负载较均衡')],loc='lower right',frameon=False,fontsize=14.5)
    ax=axs[1,0];panel(ax,'d','分组资源占用时长')
    dd=pd.concat([gs[gs.scenario_id==two].assign(case='两组'),gs[gs.scenario_id==three].assign(case='三组')],ignore_index=True)
    parts=[('transport_occupancy_h','运输机',.06,.20),('battery_occupancy_h','运输电池',.24,.41),
           ('relay_occupancy_h','中继机',.65,.81),('component_occupancy_h','能源组件',.85,.97)]
    for i,r in dd.iterrows():
        left=0
        for key,name,lo,hi in parts:
            v=float(r[key]);gradbar(ax,left,i-.31,v,.62,lo,hi);left+=v
        ax.text(left+.65,i,f'{left:.1f}',va='center',fontsize=16.5)
    ax.axhline(1.5,c='#777777',ls=':',lw=1)
    ax.set_yticks(range(len(dd)),[f'{r.case} · {grouped_sites(r)}' for _,r in dd.iterrows()]);ax.set_ylim(4.6,-.6)
    ax.set_xlim(0,dd.total_workload_h.max()*1.17);ax.set_xlabel('累计设备占用时间（小时）')
    ax.legend(handles=[Patch(fc=CM((lo+hi)/2),ec='black',label=name) for _,name,lo,hi in parts],
              frameon=False,ncol=2,loc='lower right',fontsize=14.5,columnspacing=.8,handlelength=1.2)
    ax=axs[1,1];panel(ax,'e','分组任务份额')
    metrics=[('n_sites','服务区'),('n_boxes','货箱'),('mass_kg','质量'),('total_energy_kwh','能耗')]
    spans=[(.035,.23),(.68,.84),(.88,.985)]
    rows=[]
    for case,sid in [('两组',two),('三组',three)]:
        gg=gs[gs.scenario_id==sid].sort_values('n_sites',ascending=False)
        for key,name in metrics:
            y=len(rows);left=0;total=gg[key].sum()
            for j,(_,r) in enumerate(gg.iterrows()):
                share=r[key]/total*100;lo,hi=spans[j]
                gradbar(ax,left,y-.28,share,.56,lo,hi)
                if share>=12:ax.text(left+share/2,y,f'{share:.1f}%',ha='center',va='center',fontsize=15.5,color='black')
                left+=share
            rows.append(f'{case} · {name}')
    ax.axhline(3.5,c='#777777',ls=':',lw=1);ax.set_yticks(range(8),rows);ax.set_ylim(8.25,-.6)
    ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('组内占全任务比例（%）')
    ax.legend(handles=[Patch(fc=CM(v),ec='black',label=name) for name,v in
                       [('主体组',.035),('S006',.78),('S011',.985)]],loc='lower right',
              frameon=False,ncol=3,fontsize=14,handlelength=1,columnspacing=.7)
    ax=axs[1,2];panel(ax,'f','固定资源配额利用率')
    horizon=float(f['intervals'].end_s.max())
    for i,(a,b) in enumerate(zip(rt.itertuples(),rr.itertuples())):
        ua=a.occupancy_s/(a.required*horizon)*100 if a.required else 0
        ub=b.occupancy_s/(b.required*horizon)*100 if b.required else 0
        ax.plot([ua,ub],[i-.10,i+.10],c='#8c8c8c',lw=1.7,zorder=1)
        ax.scatter(ua,i-.10,s=95,c=[BLUE],marker='o',ec='black',lw=.7,zorder=4)
        ax.scatter(ub,i+.10,s=90,c=[RED],marker='s',ec='black',lw=.7,zorder=3)
    ax.set_yticks(range(8),labels(rt));ax.set_ylim(8.2,-.6);ax.set_xlim(0,100)
    ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('累计占用 / 可配置总时长（%）')
    ax.legend(handles=[Line2D([],[],color=BLUE,marker='o',lw=0,label='两组'),
                       Line2D([],[],color=RED,marker='s',lw=0,label='三组')],frameon=False,ncol=2,loc='lower right',fontsize=15.5)
    return atomic_png(fig,output/'问题四_图03_分组配置与任务负载.png',dpi)


def concurrency(rows):
    ev={0.:0}
    for a,b in rows[['start_s','end_s']].itertuples(index=False,name=None):
        if b<=a:continue
        ev[float(a)]=ev.get(float(a),0)+1
        ev[float(b)]=ev.get(float(b),0)-1
    cur=0;xx=[];yy=[]
    for t,delta in sorted(ev.items()):
        cur+=delta;xx.append(t/3600);yy.append(cur)
    assert cur==0
    return np.array(xx),np.array(yy)


def figure_time(f,output,dpi):
    two,three=f['two_id'],f['three_id']
    rt,rr=resource_frame(f,two),resource_frame(f,three)
    inv=f['intervals'];horizon=float(inv.end_s.max())/3600
    fig,axs=plt.subplots(2,4,figsize=(26,11.4),sharex=True)
    fig.subplots_adjust(left=.045,right=.99,bottom=.08,top=.87,wspace=.22,hspace=.32)
    qa=[]
    for i,(ax,key) in enumerate(zip(axs.flat,f['order'])):
        a=rt[rt.resource_key==key].iloc[0];b=rr[rr.resource_key==key].iloc[0]
        panel(ax,'abcdefgh'[i],labels(pd.DataFrame([a]))[0])
        d2=inv[(inv.scenario_id==two)&(inv.resource_key==key)]
        d3=inv[(inv.scenario_id==three)&(inv.resource_key==key)]
        # 冻结相同任务，分组前后事件及整体占用必须完全一致。
        x,y=concurrency(d2);x3,y3=concurrency(d3)
        assert np.array_equal(x,x3) and np.array_equal(y,y3), f'{key}: 分组改变了任务时刻'
        if x[-1]<horizon:x=np.r_[x,horizon];y=np.r_[y,0]
        fill=ax.fill_between(x,0,y,step='post',facecolor='none',edgecolor='none')
        if y.max()>0:
            from matplotlib.patches import PathPatch
            clip=PathPatch(fill.get_paths()[0],transform=ax.transData)
            im=ax.imshow(np.linspace(.05,.38,200)[:,None],origin='lower',
                extent=[0,horizon,0,y.max()],aspect='auto',cmap=CM,vmin=0,vmax=1,zorder=1)
            im.set_clip_path(clip)
        ax.step(x,y,where='post',c=CM(.08),lw=2.15,zorder=5)
        # 配置门槛以相同真实整数值绘制；相同值分别用终端符号标识。
        ax.axhline(a.inventory,color='black',ls=':',lw=2.1,zorder=4)
        ax.axhline(a.required,color=CM(.75),ls='--',lw=2.2,zorder=4)
        ax.axhline(b.required,color=RED,ls='-.',lw=2.1,zorder=4)
        ax.scatter(horizon*.90,a.required,marker='o',s=95,c=[CM(.75)],ec='black',lw=.65,zorder=7)
        ax.scatter(horizon*.97,b.required,marker='s',s=90,c=[RED],ec='black',lw=.65,zorder=7)
        ax.scatter(horizon*1.04,a.inventory,marker='|',s=250,c='black',lw=2,zorder=7)
        # 缺口带只表示必须补足的分类设备数量，不改变实际占用曲线。
        if b.required>a.inventory:
            ax.axhspan(a.inventory,b.required,facecolor=CM(.95),alpha=.065,zorder=0)
        hi=max(a.required,b.required,a.inventory)
        ax.set_ylim(0,hi+.75);ax.set_xlim(0,horizon*1.07)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=6));ax.set_xticks([0,1,2])
        if i>=4:ax.set_xlabel('任务开始后时间（小时）')
        if i%4==0:ax.set_ylabel('资源数量')
        qa.append({'resource_key':key,'actual_peak':int(y.max()),'two_group_required':int(a.required),
            'three_group_required':int(b.required),'inventory':int(a.inventory),'same_occupation_curve':True})
    handles=[Patch(fc=CM(.21),ec=CM(.08),label='实际同时占用'),
        Line2D([],[],c=CM(.75),ls='--',lw=2.2,marker='o',label='两组配置下限'),
        Line2D([],[],c=RED,ls='-.',lw=2.2,marker='s',label='三组配置下限'),
        Line2D([],[],c='black',ls=':',lw=2.2,label='现有分类库存')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.985),ncol=4,
        frameon=False,fontsize=20,columnspacing=2,handlelength=2.2)
    result=atomic_png(fig,output/'问题四_图04_分类资源占用与分组门槛.png',dpi)
    return result,qa


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=ROOT/'data')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results')
    ap.add_argument('--q3-dir',type=Path,default=None)
    ap.add_argument('--dpi',type=int,default=320)
    args=ap.parse_args();setup(args.q3_dir);f=load(args.data_dir)
    p1=figure_statistics(f,args.output_dir,args.dpi)
    p2,curves=figure_time(f,args.output_dir,args.dpi)
    qa={'figures':[p1,p2],'two_group_scenario':str(f['two_id']),'three_group_scenario':str(f['three_id']),
        'resource_curve_validation':curves,'input_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(args.data_dir.glob('*.csv'))},
        'utilization_definition':'total occupied resource-seconds / (sum of group resource peaks × common last resource availability time)',
        'source_note':'固定问题三最终方案；各组独占资源；图中未重新优化运输或中继任务。'}
    (args.output_dir/'问题四_统计绘图_QA.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    return qa


if __name__=='__main__':
    main()
