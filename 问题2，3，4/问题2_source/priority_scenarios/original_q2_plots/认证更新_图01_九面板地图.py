"""沿用用户原九面板地图结构；实际数据由冻结的独立核验方案读取。
原布局来源：问题二_真实数据多面板绘图.py 的 save_map/terrain/route_lines。
不生成热力图、不改变路线几何或优化值。全部空间线均是实际节点间直线。
"""
from 认证更新_地图剖面公用 import *
from matplotlib.colors import LightSource, Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FixedLocator, FormatStrFormatter

OUT=P
TYPE_RANGE={'A':(.02,.24),'B':(.64,.82),'C':(.87,.995)}
TYPE={g:CM(np.mean(v)) for g,v in TYPE_RANGE.items()}
COORD={s:np.array([r.longitude_deg,r.latitude_deg]) for s,r in N.iterrows()}
SITES=[s for s in N.index if s!='O01']
MASS=B.groupby('node_id').mass_kg.sum().reindex(SITES)
HARD=B.groupby('node_id').hard_deadline_s.min().reindex(SITES)
EXT=[N.longitude_deg.min()-.009,N.longitude_deg.max()+.009,
     N.latitude_deg.min()-.008,N.latitude_deg.max()+.008]
ASPECT=1/np.cos(np.deg2rad(N.latitude_deg.mean()))
Z,left,top,dx,dy=read_dem(P/'q1_flat'/'最终工作DEM.tif')
r0=max(0,int((top-EXT[3])/dy));r1=min(Z.shape[0],int((top-EXT[2])/dy)+2)
c0=max(0,int((EXT[0]-left)/dx));c1=min(Z.shape[1],int((EXT[1]-left)/dx)+2)
ZZ=Z[r0:r1,c0:c1];ZEXT=[left+c0*dx,left+c1*dx,top-r1*dy,top-r0*dy]
# 沿用原九面板图的统一高程显示范围，仅影响显示，计算使用原始栅格值。
DEM_NORM=Normalize(*np.percentile(Z,[1,99]))
shade=LightSource(azdeg=305,altdeg=54).hillshade(ZZ,dx=30,dy=30)
terrain=CM(DEM_NORM(ZZ))[:,:,:3]*(.85+.15*shade[:,:,None])
RGB=.54+.46*terrain
MULTI=F.loc[F.访问顺序.str.count('→')>2]
FT=F.set_index('架次编号');IDs=set(MULTI.架次编号)
BASE=pd.read_csv(P/'问题二_统一_直接往返新基准_航段.csv')
OFFSETS={'S001':(6,7),'S002':(6,6),'S003':(6,-16),'S004':(7,5),
 'S005':(-8,7),'S006':(-5,-17),'S007':(-6,-17),'S008':(-8,7),
 'S009':(-6,-17),'S010':(6,3),'S011':(-12,7),'S012':(-3,7),
 'S013':(5,7),'S014':(-6,-17),'S015':(-8,7)}
INK='#171717'

def panel_base(ax,k,title):
    ax.imshow(RGB,extent=ZEXT,origin='upper',interpolation='bilinear',zorder=0)
    ax.set_xlim(EXT[:2]);ax.set_ylim(EXT[2:]);ax.set_aspect(ASPECT)
    ax.set_xticks([109.16,109.20,109.24,109.28]);ax.set_yticks([23.00,23.04,23.08])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'));ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(labelsize=12.8,pad=3.5,width=1.15,length=3.6)
    for spine in ax.spines.values():spine.set_linewidth(1.25);spine.set_color(INK)
    ax.set_title(f'{chr(97+k)}  {title}',loc='left',fontsize=18.5,pad=8)
    if k%3==0:ax.set_ylabel('纬度（°）',fontsize=16,labelpad=2)
    if k//3==2:ax.set_xlabel('经度（°）',fontsize=16,labelpad=3)

def nodes(ax,mode='plain',values=None,norm=None,labels=True,delivered=None):
    for s in SITES:
        xy=COORD[s];size=40+MASS[s]*1.08 if mode in ['demand','slack'] else 52
        if mode=='demand':color={3600:CM(.98),7200:CM(.73),10800:CM(.18)}.get(HARD[s],'white')
        elif mode=='slack':color=CM(norm(values[s])) if s in values else 'white'
        elif mode=='delivered':color=CM(.08) if delivered and s in delivered else 'white'
        else:color='white'
        ax.scatter(*xy,s=size,c=[color],edgecolor=INK,lw=.85,zorder=12)
        if labels:
            ox,oy=OFFSETS[s]
            txt=ax.annotate(s,xy,xytext=(ox,oy),textcoords='offset points',
                ha='right' if ox<0 else 'left',va='bottom',fontsize=12.5,zorder=15,color='black')
            txt.set_path_effects([patheffects.withStroke(linewidth=1.9,foreground='white')])
    xy=COORD['O01'];ax.scatter(*xy,s=235,marker='*',c='black',ec='white',lw=1.0,zorder=18)
    txt=ax.annotate('O01',xy,xytext=(0,-17),textcoords='offset points',ha='center',fontsize=12.5,zorder=19)
    txt.set_path_effects([patheffects.withStroke(linewidth=1.8,foreground='white')])

def arrow(ax,a,b,color,lw,at=.58):
    p=a+(b-a)*(at-.025);q=a+(b-a)*(at+.025)
    ax.annotate('',q,xytext=p,arrowprops={'arrowstyle':'-|>','color':color,
                'lw':max(1.0,lw*.55),'mutation_scale':12.5,'shrinkA':0,'shrinkB':0},zorder=10)

def route_gradient(ax,a,b,g,lw,alpha=1.,zorder=5,ret=False):
    lo,hi=TYPE_RANGE[g]
    if not ret:
        gradient_line(ax,[a[0],b[0]],[a[1],b[1]],lo,hi,lw,alpha=alpha,zorder=zorder)
    else:
        # 整条真实直线上的规则虚线段，色阶沿飞行方向连续变化。
        # 不对地理线段做偏移，也不在每个渐变采样小段上重置虚线。
        count=24
        for j in range(count):
            u=j/count;v=min((j+.62)/count,1)
            p=a+(b-a)*u;q=a+(b-a)*v
            gradient_line(ax,[p[0],q[0]],[p[1],q[1]],lo+(hi-lo)*u,lo+(hi-lo)*v,
                          lw,alpha=alpha,zorder=zorder)
    return CM(lo+(hi-lo)*(.63 if ret else .53))

def routes(ax,subset,focus=True,ids=False):
    rows=subset.to_dict('records')
    repeated=subset['访问顺序'].value_counts().to_dict()
    for r in sorted(rows,key=lambda r:len(r['访问顺序'].split('→'))):
        seq=r['访问顺序'].split('→');is_multi=len(seq)>3;g=r['机型']
        lw=(2.9 if is_multi else 1.35) if focus else 2.0
        if ids:lw=3.5 if g=='A' else 2.5
        if is_multi and repeated[r['访问顺序']]>1:
            lw=4.9 if g=='A' else 2.5
        alpha=.98 if is_multi or not focus else .33
        if not is_multi:
            a,b=COORD[seq[0]],COORD[seq[1]]
            if focus:
                ax.plot([a[0],b[0]],[a[1],b[1]],lw=lw,color='#777777',alpha=alpha,zorder=3)
            else:
                col=route_gradient(ax,a,b,g,lw,alpha,zorder=3)
                arrow(ax,a,b,col,lw)
            continue
        for u,v in zip(seq[:-1],seq[1:]):
            a,b=COORD[u],COORD[v];ret=v=='O01'
            ax.plot([a[0],b[0]],[a[1],b[1]],color='white',lw=lw+1.2,alpha=.85,zorder=4)
            col=route_gradient(ax,a,b,g,lw,alpha,zorder=5,ret=ret)
            arrow(ax,a,b,col,lw,at=.63 if ret else .53)
    if ids:
        for itinerary,group in subset.groupby('访问顺序',sort=False):
            seq=itinerary.split('→');a,b=COORD[seq[1]],COORD[seq[2]]
            xy=.46*a+.54*b;label='/'.join(group.架次编号)
            ofs=(6,8) if len(group)%2 else (-5,-15)
            ax.annotate(label,xy,xytext=ofs,textcoords='offset points',fontsize=12.6,
                color=INK,zorder=20,bbox=dict(fc='white',ec='none',alpha=.86,pad=.8))

class GradientLegend:
    def __init__(self,lo,hi,label):self.lo=lo;self.hi=hi;self.label=label
    def get_label(self):return self.label

from matplotlib.legend_handler import HandlerBase
class GradientLegendHandler(HandlerBase):
    def create_artists(self,legend,orig_handle,xdescent,ydescent,width,height,fontsize,trans):
        xs=np.linspace(xdescent,xdescent+width,25)
        return [Line2D(xs[j:j+2],[ydescent+height/2]*2,lw=3.0,
                       color=CM(orig_handle.lo+(orig_handle.hi-orig_handle.lo)*(j+.5)/24),
                       solid_capstyle='butt',transform=trans) for j in range(24)]

def north_scale(ax):
    ax.annotate('',xy=(.085,.90),xytext=(.085,.79),xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>',color='black',lw=1.2,mutation_scale=17))
    ax.text(.085,.925,'北',ha='center',transform=ax.transAxes,fontsize=13)
    x=EXT[0]+.005;y=EXT[2]+.004;length=2000/(111320*np.cos(np.deg2rad(N.latitude_deg.mean())))
    ax.plot([x,x+length],[y,y],color='black',lw=2,zorder=20)
    ax.plot([x,x],[y-.0005,y+.0005],color='black',lw=1.2)
    ax.plot([x+length,x+length],[y-.0005,y+.0005],color='black',lw=1.2)
    ax.text(x+length/2,y+.0012,'2 km',ha='center',fontsize=11.5)

def main():
    fig=plt.figure(figsize=(20.8,16.7))
    gs=fig.add_gridspec(3,3,left=.054,right=.985,bottom=.118,top=.972,wspace=.12,hspace=.16)
    wave_step=int(np.ceil(float(D.送达秒.max())/3/300)*300)
    windows=[(i*wave_step,(i+1)*wave_step) for i in range(3)]
    titles=['需求与硬截止','单点往返基准','多点运输方案','跨区访问顺序',
            '实飞航段能耗','硬时限余量']+[f'{lo//60}—{hi//60} 分钟交付' for lo,hi in windows]
    axs=[fig.add_subplot(gs[i//3,i%3]) for i in range(9)]
    for k,ax in enumerate(axs):panel_base(ax,k,titles[k])
    nodes(axs[0],'demand');north_scale(axs[0])
    for site,group in BASE[BASE.终点.ne('O01')].groupby('终点'):
        a,b=COORD['O01'],COORD[site]
        axs[1].plot([a[0],b[0]],[a[1],b[1]],c='#626262',lw=1.05+.35*(len(group)-1),alpha=.80,zorder=3)
    nodes(axs[1]);routes(axs[2],F,focus=True);nodes(axs[2])
    routes(axs[3],MULTI,focus=False,ids=True);nodes(axs[3])
    # 重合地理航段汇总为距离加权单位能耗，避免最后一条覆盖前面数值。
    energy=L.copy()
    energy['端点对']=energy.apply(lambda r:'|'.join(sorted([r['起点'],r['终点']])),axis=1)
    energy=energy.groupby('端点对',as_index=False).agg(总能耗kWh=('航段能耗kWh','sum'),累计距离m=('距离m','sum'),经过次数=('架次编号','size'))
    energy['平均单位能耗']=1000*energy.总能耗kWh/energy.累计距离m
    enorm=Normalize(energy.平均单位能耗.min(),energy.平均单位能耗.max())
    for r in energy.itertuples():
        u,v=r.端点对.split('|');a,b=COORD[u],COORD[v]
        axs[4].plot([a[0],b[0]],[a[1],b[1]],lw=3.7,color='#777777',zorder=3)
        axs[4].plot([a[0],b[0]],[a[1],b[1]],lw=2.8,color=CM(enorm(r.平均单位能耗)),zorder=4)
    nodes(axs[4])
    hd=D[np.isfinite(D.硬截止秒)].copy();hd['余量min']=(hd.硬截止秒-hd.送达秒)/60
    slack=hd.groupby('服务区')['余量min'].min().to_dict();snorm=Normalize(0,max(slack.values()))
    nodes(axs[5],'slack',values=slack,norm=snorm)
    wave_records=[]
    for k,(lo,hi) in enumerate(windows):
        w=D[(D.送达秒>lo)&(D.送达秒<=hi)];ids=set(w.架次编号)
        routes(axs[6+k],F[F.架次编号.isin(ids)],focus=False)
        nodes(axs[6+k],mode='delivered',delivered=set(w.服务区),labels=True)
        for r in w.to_dict('records'):
            wave_records.append({'波次':k+1,'开始秒':lo,'结束秒':hi,'货箱编号':r['货箱编号'],'架次编号':r['架次编号'],'服务区':r['服务区'],'送达秒':r['送达秒'],'质量kg':r['mass_kg']})
    # 共用底部图例和两个数值色标，九幅地图保留同样大小。
    deadline=[Line2D([],[],ls='',marker='o',mfc=CM(v),mec=INK,ms=8,label=f'{h} h') for h,v in [(1,.98),(2,.73),(3,.18)]]
    masshandles=[Line2D([],[],ls='',marker='o',mfc='white',mec=INK,ms=np.sqrt(40+v*1.08),label=f'{v} kg') for v in [20,80,160]]
    fig.legend(handles=deadline,loc='center',bbox_to_anchor=(.175,.074),ncol=3,frameon=False,fontsize=13,handletextpad=.4,columnspacing=1.0)
    fig.legend(handles=masshandles,loc='center',bbox_to_anchor=(.175,.048),ncol=3,frameon=False,fontsize=13,handletextpad=.4,columnspacing=.9)
    for pos,norm,label,ticks in [([.408,.067,.21,.010],enorm,'航段平均能耗（kWh/km）',None),([.743,.067,.21,.010],snorm,'硬时限余量（min）',[0,60,120])]:
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap=CM),cax=fig.add_axes(pos),orientation='horizontal')
        cb.set_label(label,fontsize=14,labelpad=3);cb.ax.tick_params(labelsize=12,length=3,width=1)
        cb.outline.set_linewidth(.8)
        if ticks is not None:cb.set_ticks(ticks)
        else:cb.locator=MaxNLocator(4);cb.update_ticks()
    handles=[GradientLegend(*TYPE_RANGE[g],f'{g} 型') for g in 'ABC']
    handles += [Line2D([],[],c=INK,lw=1.6,ls='--',label='返航'),Line2D([],[],marker='*',ls='',ms=12,c=INK,label='调度中心')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.001),ncol=5,frameon=False,fontsize=14,columnspacing=2.4,handler_map={GradientLegend:GradientLegendHandler()})
    # 仅移动架次文字，节点与航线坐标保持原值；避开服务区标签。
    import re
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax in axs:
        route_texts=[t for t in ax.texts if re.fullmatch(r'R[0-9]+(?:/R[0-9]+)*',t.get_text())]
        for tx in route_texts:
            others=[t for t in ax.texts if t is not tx and t.get_text()]
            candidates=[tx.get_position(),(6,-15),(-28,8),(-28,-15),(10,17),(-30,17),(10,-24)]
            best=None
            for pos in candidates:
                tx.set_position(pos);bb=tx.get_window_extent(renderer).expanded(1.08,1.15)
                overlaps=sum(bb.overlaps(t.get_window_extent(renderer).expanded(1.05,1.1)) for t in others)
                if best is None or overlaps<best[0]:best=(overlaps,pos)
                if overlaps==0:break
            tx.set_position(best[1])
    path=OUT/'认证更新_图01_九面板运输地图.png';tmp=path.with_name(path.stem+'.tmp.png')
    fig.savefig(tmp,dpi=360,bbox_inches='tight',pad_inches=.07);plt.close(fig);tmp.replace(path)
    energy.to_csv(OUT/'认证更新_图01_地图航段平均能耗.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(wave_records).to_csv(OUT/'认证更新_图01_地图配送波次.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({'服务区':SITES,'需求kg':MASS.to_numpy(),'最早硬截止秒':HARD.to_numpy(),'最小硬时限余量min':[slack.get(s,np.nan) for s in SITES]}).to_csv(OUT/'认证更新_图01_地图节点数据.csv',index=False,encoding='utf-8-sig')
    assert sum(r['质量kg'] for r in wave_records)==758 and len(wave_records)==80
    report={'source_prefix':str(SOURCE),'sorties':len(F),'boxes':len(D),'energy_kwh':float(F['能耗kWh'].sum()),'all_return_s':int(F['返回秒'].max()),'multi_sorties':MULTI['架次编号'].tolist(),'waves':windows,'wave_box_counts':[sum(1 for r in wave_records if r['波次']==i+1) for i in range(3)],'dem':'q1_flat/最终工作DEM.tif','style':'沿用v6九面板、同一RdYlBu_r渐变；地图无热力矩阵；实际节点间直线'}
    (OUT/'认证更新_图01_绘图核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(path.name,flush=True)

if __name__=='__main__':
    assert len(D)==80
    main()
