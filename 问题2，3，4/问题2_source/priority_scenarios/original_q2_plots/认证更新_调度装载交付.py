"""认证后运输方案的资源、装载和交付图；保留 v6 全渐变风格。

用法：python 认证更新_调度装载交付.py --source <方案JSON或前缀> [--output <目录>]
只读取显式指定方案，所有PNG、绘图数据及来源清单均写入独立目录。
精确重组和固定路线时序改善不归属于原始NSGA-II结果。
"""
from pathlib import Path
import sys,json,os,argparse,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors,font_manager,patheffects
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Rectangle
from matplotlib.legend_handler import HandlerBase
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
from 问题二_物理模型与输入 import read_inputs
N,B,M,U,BAT=read_inputs()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',required=True,help='经核验的方案JSON或其不含扩展名前缀')
parser.add_argument('--output',default=str(P/'认证更新'),help='独立输出目录')
parser.add_argument('--dpi',type=int,default=350)
args=parser.parse_args()
SOURCE=Path(args.source).resolve()
if SOURCE.suffix=='.json':SOURCE=SOURCE.with_suffix('')
OUT=Path(args.output).resolve();OUT.mkdir(parents=True,exist_ok=True)
F=pd.read_csv(str(SOURCE)+'_架次.csv')
D=pd.read_csv(str(SOURCE)+'_逐箱.csv')
L=pd.read_csv(str(SOURCE)+'_航段.csv')
D=D.merge(B[['material_type','mass_kg','volume_m3','first_batch']].rename_axis('货箱编号').reset_index(),on='货箱编号',validate='one_to_one')
META=json.loads(SOURCE.with_suffix('.json').read_text(encoding='utf-8'))
assert len(D)==len(B) and set(D.货箱编号)==set(B.index)
assert set(F.架次编号)==set(D.架次编号)==set(L.架次编号)
assert len(F)==int(META['metrics'][3])
assert abs(F.能耗kWh.sum()-META['metrics'][2])<1e-7
assert int(F.返回秒.max())==round(META['metrics'][1]*3600)
assert abs(D.归一化加权延误.sum()-META['metrics'][0])<1e-7
for entity,end in [('无人机','返回秒'),('电池','电池充满秒')]:
    for name,sub in F.groupby(entity):
        s=sub.sort_values('开始秒')
        assert np.all(s['开始秒'].to_numpy()[1:]>=s[end].to_numpy()[:-1]),(entity,name)
FONT=P/'q1_flat'/'中文字体_绘图临时.otf';font_manager.fontManager.addfont(str(FONT));CN=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.family':['STIXGeneral',CN],'font.size':16,'axes.labelsize':17,'xtick.labelsize':13,'ytick.labelsize':13,'axes.titlesize':19,'legend.fontsize':13,'axes.unicode_minus':False,'axes.linewidth':1.4,'figure.facecolor':'white','savefig.facecolor':'white','text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','pdf.fonttype':42,'ps.fonttype':42})
CM=plt.colormaps['RdYlBu_r'];MC={'A':CM(.08),'B':CM(.77),'C':CM(.98)}
ALGS=['地形ALNS–CP-SAT','普通ALNS–CP-SAT','遗传算法–CP-SAT','蚁群算法–CP-SAT','NSGA-II–CP-SAT','地形ALNS（无重组）']
LABELS=['地形ALNS','普通ALNS','遗传算法','蚁群算法','NSGA-II','地形ALNS对照']
AC={a:CM(v) for a,v in zip(ALGS,[.07,.20,.71,.84,.98,.34])}

def panel(ax,letter,title=''):
    ax.set_title(f'{letter}  {title}',loc='left',fontsize=19,pad=10,color='black')
    ax.tick_params(width=1.25,length=4)
    for sp in ax.spines.values():sp.set_linewidth(1.4)

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=13,width=1.2,length=4)

def save(fig,name):
    path=OUT/('认证更新_图'+name);temp=path.with_suffix('.tmp.png')
    fig.savefig(temp,dpi=args.dpi,bbox_inches='tight',pad_inches=.10)
    plt.close(fig);os.replace(temp,path);print(name,flush=True)

def gradient_box(ax,patch,lo=.05,hi=.95):
    # 用户渐变配色方式：对一份RdYlBu_r连续取值，再裁切到真实四分位箱体。
    vertices=patch.get_path().vertices
    xx=vertices[:,0];yy=vertices[:,1]
    if np.ptp(xx)<1e-12 or np.ptp(yy)<1e-12:return
    im=ax.imshow(np.linspace(lo,hi,256).reshape(256,1),origin='lower',aspect='auto',
      extent=[xx.min(),xx.max(),yy.min(),yy.max()],cmap=CM,vmin=0,vmax=1,zorder=patch.get_zorder()-.1,alpha=.68)
    im.set_clip_path(patch);patch.set_facecolor('none');patch.set_edgecolor('black');patch.set_linewidth(1.3)

def gradient_patch(ax,patch,lo=.05,hi=.95,orientation='vertical',alpha=1,zorder=None):
    """数据坐标内裁切渐变；不更改柱长/区间，不改变已有坐标范围。"""
    xy=patch.get_path().transformed(patch.get_patch_transform()).vertices
    x0,y0=np.min(xy,axis=0);x1,y1=np.max(xy,axis=0)
    if x1-x0<=1e-14 or y1-y0<=1e-14:return None
    grad=np.linspace(lo,hi,256)
    grad=grad[:,None] if orientation=='vertical' else grad[None,:]
    limits=(ax.get_xlim(),ax.get_ylim());aspect=ax.get_aspect()
    z=patch.get_zorder()-.05 if zorder is None else zorder
    im=ax.imshow(grad,extent=[x0,x1,y0,y1],origin='lower',aspect='auto',
                 cmap=CM,vmin=0,vmax=1,alpha=alpha,zorder=z,interpolation='bilinear')
    im.set_clip_path(patch);patch.set_facecolor('none')
    ax.set_xlim(limits[0]);ax.set_ylim(limits[1]);ax.set_aspect(aspect)
    return im

def gradient_fill(ax,collection,xmin,xmax,ymin,ymax,lo=.05,hi=.95,
                  orientation='vertical',alpha=1,zorder=None):
    from matplotlib.patches import PathPatch
    if xmax<=xmin or ymax<=ymin:return None
    grad=np.linspace(lo,hi,256)
    grad=grad[:,None] if orientation=='vertical' else grad[None,:]
    limits=(ax.get_xlim(),ax.get_ylim());aspect=ax.get_aspect()
    im=ax.imshow(grad,extent=[xmin,xmax,ymin,ymax],origin='lower',aspect='auto',
        cmap=CM,vmin=0,vmax=1,alpha=alpha,
        zorder=collection.get_zorder()-.05 if zorder is None else zorder,
        interpolation='bilinear')
    im.set_clip_path(PathPatch(collection.get_paths()[0],transform=ax.transData))
    collection.set_facecolor('none')
    ax.set_xlim(limits[0]);ax.set_ylim(limits[1]);ax.set_aspect(aspect)
    return im

def gradient_line(ax,x,y,lo=.05,hi=.95,lw=2.5,step=None,alpha=1,zorder=3):
    """沿已有折线着色，post阶梯在原事件位置跳变，不插值数据值。"""
    from matplotlib.collections import LineCollection
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if len(x)<2:return None
    if step=='post':
        x=np.repeat(x,2)[1:];y=np.repeat(y,2)[:-1]
    path=np.c_[x,y];pieces=[];positions=[]
    # 对直线作绘图细分，仅改变渲染色阶；数据折点原位保留。
    for i,(a,b) in enumerate(zip(path[:-1],path[1:])):
        if np.allclose(a,b,rtol=0,atol=1e-14):continue
        t=np.linspace(0,1,20);fine=a+(b-a)*t[:,None]
        pieces.extend(np.stack([fine[:-1],fine[1:]],axis=1))
        positions.extend((i+(t[:-1]+t[1:])/2)/(len(path)-1))
    lc=LineCollection(pieces,colors=CM(lo+(hi-lo)*np.asarray(positions)),
                      linewidths=lw,alpha=alpha,zorder=zorder,capstyle='butt')
    ax.add_collection(lc)
    return lc

from matplotlib.patches import Patch as _Patch,Rectangle as _Rectangle
from matplotlib.legend_handler import HandlerBase as _HandlerBase
from matplotlib.legend import Legend as _Legend
class GradientKey(_Patch):
    def __init__(self,label,lo,hi,hatch=None):
        super().__init__(facecolor='none',edgecolor='black',label=label,hatch=hatch)
        self.lo=lo;self.hi=hi
class GradientKeyHandler(_HandlerBase):
    def create_artists(self,legend,handle,xd,yd,w,h,fs,trans):
        artists=[_Rectangle((-xd+j*w/32,-yd),w/32+.02,h,
                    fc=CM(value),ec='none',transform=trans)
                  for j,value in enumerate(np.linspace(handle.lo,handle.hi,32))]
        artists.append(_Rectangle((-xd,-yd),w,h,fc='none',ec='black',lw=.45,
                                  hatch=handle.get_hatch(),transform=trans))
        return artists
_Legend.update_default_handler_map({GradientKey:GradientKeyHandler()})

plt.rcParams.update({'font.size':17,'axes.labelsize':18,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':15,'axes.linewidth':1.3})
BLUE=CM(.08);LIGHT=CM(.26);ORANGE=CM(.75);RED=CM(.97)
PHASE_RANGES=[(.03,.17),(.18,.41),(.62,.82),(.84,1.0)]

def panel(ax,letter,title):
    ax.set_title(f'{letter}  {title}',loc='left',fontsize=20,pad=10)
    ax.spines[['top','right']].set_visible(False);ax.tick_params(width=1.1,length=4)

def occupancy(rows,start,end,horizon):
    ev={0:0,horizon:0}
    for r in rows:
        ev[r[start]]=ev.get(r[start],0)+1;ev[r[end]]=ev.get(r[end],0)-1
    t=sorted(ev);value=0;counts=[]
    for time in t:value+=ev[time];counts.append(value)
    return np.array(t),np.array(counts)

def resources():
    fig=plt.figure(figsize=(20.8,13.4))
    grid=fig.add_gridspec(3,3,height_ratios=[1.05,1.25,.95],left=.063,right=.989,bottom=.074,top=.90,wspace=.23,hspace=.47)
    horizon=int(np.ceil(F['电池充满秒'].max()/1800)*1800)
    hour_ticks=np.arange(0,horizon/3600+.01,.5 if horizon<=7200 else 1.)
    pc=[BLUE,LIGHT,ORANGE,RED];phase_names=['准备装载','去程及续飞','交接','返航']
    handles=[GradientKey(name,*ran) for name,ran in zip(phase_names,PHASE_RANGES)]
    handles += [GradientKey('电池充电',.62,.96,hatch='///')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.985),ncol=5,frameon=False,fontsize=16,columnspacing=1.8)
    records=[];peaks=[];stages=[]
    for col,g in enumerate('ABC'):
        axes=[fig.add_subplot(grid[row,col]) for row in range(3)];a,b,c=axes
        us=sorted(k for k,v in U.items() if v['model']==g);bs=sorted(k for k,v in BAT.items() if v['model']==g)
        subset=F.loc[F.机型.eq(g)].to_dict('records')
        for r in subset:
            own=D.loc[D.架次编号.eq(r['架次编号'])];m=M.loc[g];time=int(r['开始秒']);iu=us.index(r['无人机']);ib=bs.index(r['电池'])
            segments=[(int(m.fixed_prep_s+m.load_per_box_s*len(own)),0)]
            for leg in L.loc[L.架次编号.eq(r['架次编号'])].to_dict('records'):
                segments.append((int(leg['飞行秒']),3 if leg['终点']=='O01' else 1))
                if leg['终点']!='O01':segments.append((int(m.handoff_base_s+m.handoff_per_box_s*own.服务区.eq(leg['终点']).sum()),2))
            for duration,phase in segments:
                bars=a.barh(iu,duration/3600,left=time/3600,height=.56,fc='none',ec='white',lw=.5,zorder=3)
                gradient_patch(a,bars[0],*PHASE_RANGES[phase],orientation='horizontal')
                stages.append({'架次':r['架次编号'],'机型':g,'无人机':r['无人机'],'阶段':phase_names[phase],'开始秒':time,'结束秒':time+duration})
                time+=duration
            assert time==r['返回秒']
            a.text((r['开始秒']+r['返回秒'])/7200,iu-.41,r['架次编号'],ha='center',va='center',fontsize=13.3)
            bar=b.barh(ib,(r['返回秒']-r['开始秒'])/3600,left=r['开始秒']/3600,height=.55,fc='none',ec='white',lw=.5,zorder=3)
            gradient_patch(b,bar[0],.06,.39,orientation='horizontal')
            bar=b.barh(ib,(r['电池充满秒']-r['返回秒'])/3600,left=r['返回秒']/3600,height=.55,fc='none',ec=RED,lw=.4,hatch='///',zorder=3)
            gradient_patch(b,bar[0],.62,.96,orientation='horizontal')
            b.text((r['开始秒']+r['返回秒'])/7200,ib-.40,r['架次编号'],ha='center',va='center',fontsize=13.2)
        for ax,names in [(a,us),(b,bs)]:
            ax.set_yticks(range(len(names)),names);ax.set_ylim(len(names)-.25,-.83)
            ax.set_xlim(0,horizon/3600);ax.set_xticks(hour_ticks);ax.grid(axis='x',color='#e1e1e1',ls=':',lw=.7);ax.set_axisbelow(True)
        panel(a,'abc'[col],f'{g} 型无人机');panel(b,'def'[col],f'{g} 型电池')
        for label,end,stock,color in [('无人机','返回秒',len(us),BLUE),('电池','电池充满秒',len(bs),RED)]:
            times,count=occupancy(subset,'开始秒',end,horizon)
            ran=(.32,.03) if label=='无人机' else (.78,1.0)
            gradient_line(c,times/3600,count,*ran,lw=2.6,step='post')
            c.axhline(stock,c=color,ls=(0,(3,3)),lw=1.0,alpha=.8)
            c.text(horizon/3600-.035,stock+.12,f'{stock} 架' if label=='无人机' else f'{stock} 组',ha='right',fontsize=12.5,color='black')
            peak=int(max(count));assert peak<=stock
            peaks.append({'机型':g,'资源':label,'峰值':peak,'库存':stock})
            records += [{'机型':g,'资源':label,'时刻秒':int(t),'同时占用':int(n),'库存':stock} for t,n in zip(times,count)]
        c.set_ylim(-.08,6.75);c.set_yticks([0,2,4,6]);c.set_xlim(0,horizon/3600);c.set_xticks(hour_ticks);c.set_xlabel('时间（h）');c.set_ylabel('同时占用数' if col==0 else '')
        panel(c,'ghi'[col],f'{g} 型资源峰值')
        if col==0:c.legend(handles=[GradientKey('无人机',.32,.03),GradientKey('电池',.78,1.0)],loc='upper center',bbox_to_anchor=(.52,1.03),ncol=2,frameon=False,fontsize=13.8)
    save(fig,'04_联合调度与资源峰值.png')
    pd.DataFrame(records).to_csv(OUT/'认证更新_图04_资源占用阶梯.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(peaks).to_csv(OUT/'认证更新_图04_资源峰值核验.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(stages).to_csv(OUT/'认证更新_图04_飞行任务阶段.csv',index=False,encoding='utf-8-sig')

def deliveries():
    data=D.copy();hard=data.loc[np.isfinite(data.硬截止秒)].copy();hard['余量min']=(hard.硬截止秒-hard.送达秒)/60
    agg=data.groupby('服务区').agg(首次秒=('送达秒','min'),完成秒=('送达秒','max'),箱数=('货箱编号','size'),质量kg=('mass_kg','sum'))
    agg=agg.join(hard.groupby('服务区').agg(最小余量min=('余量min','min'),最大余量min=('余量min','max'),硬时限箱数=('货箱编号','size')))
    agg=agg.sort_values(['最小余量min','完成秒']);sites=agg.index.tolist();y=np.arange(len(sites))
    assert len(sites)==len(N)-1 and len(data)==len(B) and len(hard)==np.isfinite(B.hard_deadline_s).sum() and hard.余量min.min()>=0
    delivery_end=float(data.送达秒.max()/3600)
    trace_end=float(np.ceil(delivery_end*10)/10+.02)
    axis_end=trace_end+.18
    hour_ticks=np.arange(0,axis_end,.5)
    fig,axes=plt.subplots(1,3,figsize=(21.0,6.9),gridspec_kw={'width_ratios':[1.04,1.02,1.02]})
    fig.subplots_adjust(left=.052,right=.985,top=.88,bottom=.13,wspace=.31)
    a,b,c=axes;progress=[]
    for label,sub,col,style in [('全部',data,BLUE,'-'),('首批',data[data.first_batch.eq('是')],RED,'--'),('医疗',data[data.material_type.eq('医疗物资')],ORANGE,':')]:
        counts=sub.groupby('送达秒').size().sort_index();xx=np.r_[0,counts.index/3600,trace_end];yy=np.r_[0,counts.cumsum().values,len(sub)]
        ran={'全部':(.37,.02),'首批':(.81,1.0),'医疗':(.63,.84)}[label]
        gradient_line(a,xx,yy,*ran,lw=2.8,step='post')
        if label=='全部':
            area=a.fill_between(xx,0,yy,step='post',fc='none',ec='none',zorder=.7)
            gradient_fill(a,area,0,trace_end,0,len(data),.08,.44,alpha=.20)
        a.text(trace_end+.01,len(sub),f'{len(sub)}',ha='left',va='center',fontsize=16,color='black')
        progress += [{'类别':label,'时刻秒':float(t*3600),'累计箱数':int(n)} for t,n in zip(xx,yy)]
    a.set_xlim(0,axis_end);a.set_ylim(0,len(data)+7);a.set_xticks(hour_ticks);a.set_yticks([0,20,40,60,80]);a.set_xlabel('任务时间（h）');a.set_ylabel('累计交付（箱）')
    a.legend(handles=[GradientKey('全部',.37,.02),GradientKey('首批',.81,1.),GradientKey('医疗',.63,.84)],frameon=False,loc='upper left',fontsize=15)
    panel(a,'a','累计交付进度')
    for i,site in enumerate(sites):
        r=agg.loc[site]
        gradient_line(b,[r.首次秒/3600,r.完成秒/3600],[i,i],.43,.04,lw=3.6,zorder=1)
        b.scatter(r.首次秒/3600,i,s=52,fc='white',ec=BLUE,lw=1.3,zorder=3)
        b.scatter(r.完成秒/3600,i,s=50,fc=BLUE,ec='black',lw=.45,zorder=4)
        gradient_line(c,[r.最小余量min,r.最大余量min],[i,i],.95,.12,lw=3.4,zorder=1)
        c.scatter(r.最大余量min,i,s=41,fc='white',ec=BLUE,lw=1.1,zorder=3)
        c.scatter(r.最小余量min,i,s=45,fc=RED,ec='black',lw=.4,zorder=4)
    b.set_yticks(y,sites);b.set_ylim(len(sites)-.45,-.9);b.set_xlim(0,trace_end+.06);b.set_xticks(hour_ticks[hour_ticks<=trace_end+.06]);b.set_xlabel('送达时刻（h）')
    b.legend(handles=[Line2D([],[],ls='',marker='o',mfc='white',mec=BLUE,ms=7,label='首箱'),Line2D([],[],ls='',marker='o',mfc=BLUE,mec='black',ms=7,label='全部')],loc='upper center',bbox_to_anchor=(.5,1.15),ncol=2,frameon=False,fontsize=14)
    c.set_yticks(y,['']*len(y));c.set_ylim(len(sites)-.45,-.9);c.set_xlim(-6,180);c.set_xticks([0,60,120,180]);c.set_xlabel('硬时限余量（min）');c.axvline(0,c='black',ls='--',lw=1.1)
    c.legend(handles=[Line2D([],[],ls='',marker='o',mfc=RED,mec='black',ms=7,label='最小'),Line2D([],[],ls='',marker='o',mfc='white',mec=BLUE,ms=7,label='最大')],loc='upper center',bbox_to_anchor=(.5,1.15),ncol=2,frameon=False,fontsize=14)
    c.annotate(f'{agg.iloc[0].最小余量min:.1f} min',(agg.iloc[0].最小余量min,0),xytext=(10,7),textcoords='offset points',fontsize=14,ha='left')
    for ax in [b,c]:ax.grid(axis='y',color='#ececec',lw=.55);ax.set_axisbelow(True)
    panel(b,'b','服务区交付时段');panel(c,'c','硬时限余量')
    save(fig,'07_交付进度与时限保障.png')
    agg.to_csv(OUT/'认证更新_图07_服务区交付与时限.csv',encoding='utf-8-sig')
    pd.DataFrame(progress).to_csv(OUT/'认证更新_图07_累计交付阶梯.csv',index=False,encoding='utf-8-sig')
    hard.to_csv(OUT/'认证更新_图07_硬时限逐箱核验.csv',index=False,encoding='utf-8-sig')

class GradientLegend(HandlerBase):
    """与面板填充相同的连续色段，不引入额外颜色。"""
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        lo,hi,orient=orig_handle._gradient
        out=[]
        for j,v in enumerate(np.linspace(lo,hi,32)):
            if orient=='vertical':
                x,y,w,h=-xdescent,-ydescent+j*height/32,width,height/32+.02
            else:
                x,y,w,h=-xdescent+j*width/32,-ydescent,width/32+.02,height
            out.append(Rectangle((x,y),w,h,facecolor=CM(v),edgecolor='none',transform=trans))
        out.append(Rectangle((-xdescent,-ydescent),width,height,facecolor='none',edgecolor='black',linewidth=.55,transform=trans))
        return out


def gradient_key(label,lo,hi,orientation='vertical'):
    p=Patch(facecolor='none',edgecolor='black',label=label)
    p._gradient=(lo,hi,orientation)
    return p

def gradient_segment(ax, x0, x1, y, colors=None, linewidth=2.8):
    """用户原 100 段渐变线；连接的两个端点均为实际计算值。"""
    xx = np.linspace(float(x0), float(x1), 101)
    yy = np.full_like(xx, y)
    pts = np.column_stack((xx, yy))
    segs = np.stack((pts[:-1], pts[1:]), axis=1)
    cc = CM(np.linspace(.10, .91, 100)) if colors is None else colors
    lc = LineCollection(segs, colors=cc, linewidth=linewidth, zorder=2)
    ax.add_collection(lc)


def plotting_data():
    frame = F.sort_values(['返航SOC%', '架次编号'], kind='stable').reset_index(drop=True).copy()
    frame.insert(0, '图中自上而下排序', np.arange(1, len(frame) + 1))
    parameter_names = {
        'max_payload_kg': '机型额定载质量kg',
        'max_volume_m3': '机型额定体积m3',
        'empty_mass_kg': '机型含电池空载质量kg',
        'battery_kwh': '机型可用电池能量kWh',
        'climb_efficiency': '机型爬升效率',
        'reserve_percent': '机型返航SOC下限%',
        'cruise_speed_m_s': '机型巡航速度m每秒',
    }
    for source, name in parameter_names.items():
        frame[name] = frame['机型'].map(M[source])
    frame['质量利用率%'] = 100 * frame['质量kg'] / frame['机型额定载质量kg']
    frame['体积利用率%'] = 100 * frame['体积m3'] / frame['机型额定体积m3']
    climb = []
    for row in frame.to_dict('records'):
        arcs = L[L['架次编号'].eq(row['架次编号'])]
        model = M.loc[row['机型']]
        ec = ((model.empty_mass_kg + arcs['剩余载荷kg']) * 9.80665 * arcs['爬升m']
              / (3.6e6 * model.climb_efficiency)).sum()
        climb.append(ec)
    frame['爬升附加能耗kWh'] = climb
    frame['水平飞行能耗kWh'] = frame['能耗kWh'] - frame['爬升附加能耗kWh']
    frame['返航SOC余量百分点'] = frame['返航SOC%'] - frame['机型返航SOC下限%']
    assert len(frame)==len(F)
    assert (frame[['质量利用率%','体积利用率%']]<=100+1e-7).all().all()
    assert frame['返航SOC%'].is_monotonic_increasing
    assert np.allclose(frame['水平飞行能耗kWh'] + frame['爬升附加能耗kWh'], frame['能耗kWh'])
    assert np.allclose(frame['返航SOC%'], 100 * (1 - frame['能耗kWh'] / frame['机型可用电池能量kWh']))
    assert frame['返航SOC余量百分点'].min() >= -1e-7
    return frame


def draw():
    plt.rcParams.update({'font.size':17, 'axes.labelsize':18,
        'xtick.labelsize':16, 'ytick.labelsize':16, 'legend.fontsize':16,
        'axes.linewidth':1.2})
    frame = plotting_data()
    frame.to_csv(OUT / '认证更新_图06_逐架次排序及物理参数.csv', index=False, encoding='utf-8-sig')
    n = len(frame)
    y = np.arange(n)
    fig, axs = plt.subplots(1, 3, figsize=(16.9, max(7.9,.39*n+0.6)), sharey=True,
                            gridspec_kw={'width_ratios':[1.08, 1.05, 1.0]})
    fig.subplots_adjust(left=.083, right=.985, bottom=.091, top=.870, wspace=.18)
    blue, red = CM(.10), CM(.91)
    legend = [
        Line2D([], [], marker='o', color='none', markerfacecolor=blue,
               markeredgecolor='black', markeredgewidth=.55, markersize=7, label='质量利用率'),
        Line2D([], [], marker='s', color='none', markerfacecolor=red,
               markeredgecolor='black', markeredgewidth=.55, markersize=7, label='体积利用率'),
        gradient_key('水平飞行能耗',.04,.40,'horizontal'),
        gradient_key('爬升附加能耗',.58,.97,'horizontal'),
        Line2D([], [], color='black', linestyle='--', linewidth=1.35, label='返航安全阈值'),
    ]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.53, .995),
               ncol=5, frameon=False, handlelength=1.6, columnspacing=1.35,
               handletextpad=.55, fontsize=16,handler_map={Patch:GradientLegend()})

    ax = axs[0]
    mass, volume = frame['质量利用率%'].to_numpy(), frame['体积利用率%'].to_numpy()
    for i in y:
        gradient_segment(ax, mass[i], volume[i], i)
    ax.scatter(mass, y, s=54, facecolors=blue, edgecolors='black', linewidths=.6, zorder=4)
    ax.scatter(volume, y, s=51, facecolors=red, edgecolors='black', linewidths=.6,
               marker='s', zorder=4)
    ax.axvline(100, color='#777777', ls=':', lw=1.0, zorder=0)
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('装载利用率（%）', labelpad=9)
    ax.set_yticks(y, [f'{r.架次编号} · {r.机型}' for r in frame.itertuples()])
    ax.get_yticklabels()[0].set_fontweight('bold')

    ax = axs[1]
    horizontal = frame['水平飞行能耗kWh'].to_numpy()
    climb = frame['爬升附加能耗kWh'].to_numpy()
    bh=ax.barh(y,horizontal,height=.55,color='none',edgecolor='white',linewidth=.55,zorder=3)
    bc=ax.barh(y,climb,left=horizontal,height=.55,color='none',edgecolor='white',linewidth=.55,zorder=3)
    for p in bh.patches:gradient_patch(ax,p,.04,.40,orientation='horizontal',alpha=1,zorder=2.8)
    for p in bc.patches:gradient_patch(ax,p,.58,.97,orientation='horizontal',alpha=1,zorder=2.8)
    ax.set_xlim(0, max(6.7,float(frame.能耗kWh.max())*1.04))
    ax.set_xticks([0, 2, 4, 6])
    ax.set_xlabel('架次能耗（kWh）', labelpad=9)

    ax = axs[2]
    soc = frame['返航SOC%'].to_numpy()
    threshold = float(frame['机型返航SOC下限%'].iloc[0])
    assert frame['机型返航SOC下限%'].nunique() == 1
    for i in y:
        gradient_segment(ax, threshold, soc[i], i,
                         colors=CM(np.linspace(.10, .10 + .81 * (soc[i] - 20) / 60, 100)),
                         linewidth=2.6)
    ax.scatter(soc, y, c=CM(.10 + .81 * (soc - 20) / 60), s=54,
               edgecolor='black', linewidth=.65, zorder=4)
    ax.scatter(soc[0], y[0], s=130, marker='o', facecolor='none',
               edgecolor=red, linewidth=1.65, zorder=6)
    ax.annotate(f'最低 {soc[0]:.1f}%', (soc[0], y[0]), xytext=(14, 0),
                textcoords='offset points', ha='left', va='center',
                fontsize=17, fontweight='bold', color='black')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.35, zorder=3)
    ax.set_xlim(17, 82)
    ax.set_xticks([20, 40, 60, 80])
    ax.set_xlabel('返航 SOC（%）', labelpad=9)

    for i, ax in enumerate(axs):
        ax.set_ylim(n - .45, -.82)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e6e6e6', linewidth=.55)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_linewidth(1.2)
        ax.tick_params(axis='both', width=1.1, length=4)
        if i:
            ax.tick_params(axis='y', left=False, labelleft=False)
            ax.spines['left'].set_visible(False)
        ax.set_title(f'{"abc"[i]}  ' + ['质量与体积', '能耗组成', '返航余量'][i],
                     loc='left', fontsize=21, pad=14, color='black')
    target = OUT / '认证更新_图06_装载能耗与返航余量.png'
    temporary=target.with_suffix('.tmp.png')
    fig.savefig(temporary, dpi=args.dpi, bbox_inches='tight', pad_inches=.09)
    plt.close(fig);os.replace(temporary,target)
    print(target, flush=True)
    print(frame[['图中自上而下排序', '架次编号', '返航SOC%', '返航SOC余量百分点']].head().to_string(index=False), flush=True)



def write_provenance():
    files=[SOURCE.with_suffix('.json')]+[Path(str(SOURCE)+'_'+x+'.csv') for x in ['架次','航段','逐箱']]
    check=Path(str(SOURCE)+'_独立核验.json')
    if check.exists():files.append(check)
    hard=D.loc[np.isfinite(D.硬截止秒)]
    metadata={
        '方案来源':str(SOURCE),
        '方法归属':'集成列池精确重组，随后固定路线集合精确时序优化；不属于NSGA-II自身搜索结果',
        '认证范围':'源JSON中OPTIMAL仅适用于该固定路线集合的时序优化，未据此宣称原问题全局最优',
        '绘图风格':'延续v6九面板资源图及三面板哑铃图，RdYlBu_r连续渐变；不使用热图',
        '指标':{'归一化加权延误':float(D.归一化加权延误.sum()),'完工秒':int(F.返回秒.max()),'能耗kWh':float(F.能耗kWh.sum()),'架次数':len(F),'逐箱数':len(D),'硬时限箱数':len(hard),'最小硬时限余量秒':float((hard.硬截止秒-hard.送达秒).min()),'最低返航SOC百分比':float(F['返航SOC%'].min())},
        '输入文件':[{'文件':str(f),'SHA256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files],
        '资源占用口径':'无人机[开始,返回)，电池[开始,充满)，相同时刻释放与占用合并；电池占用包含任务与充电',
        '累计交付口径':'全部、首批、医疗为包含或交叉集合，不相加；每箱在交接完成时计入',
        '区间口径':'服务区首箱至末箱、硬时限最小至最大余量均为真实区间，不是置信区间',
        '图06排序':'按返航SOC升序，架次编号次序打破并列；能耗由水平飞行与爬升附加组成',
    }
    (OUT/'认证更新_图04_06_07_数据来源.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'认证更新_图04_06_07_绘图口径.md').write_text(
        '# 资源、装载与交付图\n\n'
        +'三图均读取同一份精确重组后、固定路线集合时序优化方案。全部曲线、条长、点位和区间来自方案CSV，不插值或修改任务数据。渐变线仅细分渲染色阶，数据折点保持原位。\n\n'
        +'源方案：`'+SOURCE.name+'`。对应数值、输入文件SHA256、核验文件及口径见同目录数据来源JSON。源文件OPTIMAL只表示固定路线集合调度最优，不表示完整原问题已全局最优。精确改善不计入NSGA-II自身结果。\n\n'
        +'图04：半开资源占用区间；无人机执行至返回，电池执行及充电至充满。同一事件时刻合并计数。\n\n'
        +'图06：按返航SOC自上而下升序；利用率按各机型额定质量/体积计算；爬升附加能耗使用真实逐段载荷和爬升高度。\n\n'
        +'图07：累计全部、首批与医疗集合互有重叠，不相加。服务区按最小硬时限余量升序。哑铃线为真实首箱—末箱、最小—最大余量。\n',encoding='utf-8')

if __name__=='__main__':
    resources();deliveries();draw();write_provenance()
