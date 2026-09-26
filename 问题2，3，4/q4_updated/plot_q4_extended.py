"""Four additional Q4 evidence figures from the fixed Q3 sorties and Q4 partitions.

Figure 08: mandatory same-group graph and exhaustive legal partition choices.
Figure 09: every group's dedicated resource counts, stock and shortage causes.
Figure 10: actual occupancy intervals giving the B/C resource lower bounds.
Figure 11: partition overhead, unmatched idle stock, work and box shares.
No Q3 route, relay sortie, delivery time, or communication assignment is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle, Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import LightSource

ROOT = Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument("--data-dir",type=Path,default=ROOT/"data")
ap.add_argument("--q3-dir",type=Path,default=ROOT/"q3_source")
ap.add_argument("--output-dir",type=Path,default=ROOT/"results")
ap.add_argument("--dpi",type=int,default=300)
a=ap.parse_args()
a.output_dir.mkdir(parents=True,exist_ok=True)
font_manager.fontManager.addfont(str(a.q3_dir/"q1_flat"/"中文字体_绘图临时.otf"))
CN=font_manager.FontProperties(fname=str(a.q3_dir/"q1_flat"/"中文字体_绘图临时.otf")).get_name()
plt.rcParams.update({"font.family":["STIXGeneral",CN],"axes.unicode_minus":False,
  "pdf.fonttype":42,"font.size":13,"axes.labelsize":14,"axes.titlesize":17,
  "xtick.labelsize":11,"ytick.labelsize":11,"text.color":"#171717",
  "axes.labelcolor":"#171717","figure.facecolor":"white","savefig.facecolor":"white"})
COL={"G1":"#314F8C","G2":"#F07847","G3":"#A50026"}
DEM=plt.colormaps["RdYlBu_r"]

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plane(lon,lat):
    return np.column_stack(((np.asarray(lon)-lon0)*111.32*math.cos(math.radians(lat0)),
                            (np.asarray(lat)-lat0)*111.132))
def halo(t):t.set_path_effects([patheffects.withStroke(linewidth=2.8,foreground="white")])

def save(fig,stem):
    png=a.output_dir/(stem+".png");pdf=a.output_dir/(stem+".pdf")
    tmp_png=a.output_dir/(stem+".tmp.png");tmp_pdf=a.output_dir/(stem+".tmp.pdf")
    fig.savefig(tmp_png,dpi=a.dpi,bbox_inches="tight",pad_inches=.13)
    fig.savefig(tmp_pdf,bbox_inches="tight",pad_inches=.13)
    plt.close(fig)
    with Image.open(tmp_png) as im:im.verify()
    with Image.open(tmp_png) as im:im.load();sz=list(im.size)
    assert tmp_pdf.stat().st_size>5000
    tmp_png.replace(png);tmp_pdf.replace(pdf)
    return {"png":png.name,"pdf":pdf.name,"size_px":sz,"sha256":digest(png)}

nodes=pd.read_csv(a.data_dir/"nodes.csv").sort_values("site_id")
sites=nodes[nodes.site_id.ne("O01")].copy().reset_index(drop=True)
depot=nodes[nodes.site_id.eq("O01")].iloc[0]
lon0,lat0=float(depot.longitude),float(depot.latitude)
site_xy=plane(sites.longitude,sites.latitude)
node_xy={r.site_id:tuple(xy)for r,xy in zip(sites.itertuples(),site_xy)}
node_xy["O01"]=(0.,0.)
flights=pd.read_csv(a.data_dir/"sources"/"flights.csv")
relays=pd.read_csv(a.data_dir/"sources"/"relays.csv")
paths={r["架次编号"]:r["访问顺序"].split("→") for _,r in flights.iterrows()}
groups=pd.read_csv(a.data_dir/"site_groups.csv")
components=pd.read_csv(a.data_dir/"components.csv")
sites=sites.merge(components[["site_id","n_boxes"]],on="site_id",validate="one_to_one")
links=pd.read_csv(a.data_dir/"must_link_edges.csv")
scenarios=pd.read_csv(a.data_dir/"scenarios.csv").set_index("scenario_id")
grouptot=pd.read_csv(a.data_dir/"group_summary.csv")
groupres=pd.read_csv(a.data_dir/"group_resources.csv")
resources=pd.read_csv(a.data_dir/"scenario_resources.csv")
intervals=pd.read_csv(a.data_dir/"intervals.csv")
delivery=pd.read_csv(a.data_dir/"sources"/"deliveries.csv")
hexes=pd.read_csv(a.data_dir/"hex_distance_cells.csv")
assert len(sites)==15 and len(flights)==20 and len(relays)==3 and len(delivery)==80
assert set(scenarios.index)=={"K2_01","K2_02","K2_03","K3_01"}
assert len(hexes)==166 and len(links)==44
K2,K3="K2_02","K3_01"
lookup={sid:groups[groups.scenario_id.eq(sid)].set_index("site_id").group_id.to_dict()
        for sid in scenarios.index}
for sid, mapping in lookup.items():
    assert len(mapping)==15
    for seq in paths.values():
        assert len({mapping[n] for n in seq if n!="O01"})==1

def dem_crop():
    p=a.q3_dir/"q1_flat"/"最终工作DEM.tif"
    with Image.open(p) as image:
        z=np.asarray(image,dtype=np.float32).copy()
        scale=image.tag_v2[33550];tie=image.tag_v2[33922]
        geo=tuple(image.tag_v2[34735]);rt=None
        for i in range(4,len(geo),4):
            if geo[i]==1025:rt=geo[i+3]
    assert z.ndim==2 and np.isfinite(z).all() and rt in(1,2)
    dx,dy=float(scale[0]),float(scale[1])
    left=float(tie[3])-(dx/2 if rt==2 else 0)
    top=float(tie[4])+(dy/2 if rt==2 else 0)
    bounds=[site_xy[:,0].min()-1.55,site_xy[:,0].max()+1.55,
            site_xy[:,1].min()-1.45,site_xy[:,1].max()+1.45]
    lon_l=lon0+bounds[0]/(111.32*math.cos(math.radians(lat0)))
    lon_r=lon0+bounds[1]/(111.32*math.cos(math.radians(lat0)))
    lat_b=lat0+bounds[2]/111.132;lat_t=lat0+bounds[3]/111.132
    c0=max(0,int((lon_l-left)/dx));c1=min(z.shape[1],int((lon_r-left)/dx)+2)
    r0=max(0,int((top-lat_t)/dy));r1=min(z.shape[0],int((top-lat_b)/dy)+2)
    crop=z[r0:r1,c0:c1]
    assert crop.size>10000
    x=((left+np.array([c0,c1])*dx)-lon0)*111.32*math.cos(math.radians(lat0))
    y=((top-np.array([r1,r0])*dy)-lat0)*111.132
    ext=[x[0],x[1],y[0],y[1]]
    shade=LightSource(azdeg=313,altdeg=50).hillshade(crop,dx=30,dy=30,vert_exag=1.5)
    color=np.clip(DEM(Normalize(0,1150)(crop))[...,:3]*(.76+.24*shade[...,None]),0,1)
    return crop,color,ext,bounds,p

Z,RELIEF,EXT,BOUNDS,DEM_PATH=dem_crop()
radius=.51
verts=np.array([[radius*math.cos(j*math.pi/3),radius*math.sin(j*math.pi/3)] for j in range(6)])
poly=hexes[["center_x_km","center_y_km"]].to_numpy()[:,None,:]+verts[None,:,:]

def borders_for(labels):
    edge_map={}
    for idx,p in enumerate(poly):
        for lo,hi in zip(p,np.roll(p,-1,axis=0)):
            edge=tuple(sorted((tuple(np.round(lo,5)),tuple(np.round(hi,5)))))
            edge_map.setdefault(edge,[]).append(idx)
    return [edge for edge,idx in edge_map.items()if len(idx)==2 and labels[idx[0]]!=labels[idx[1]]]

def map_base(ax,title,num,contour=False):
    ax.imshow(RELIEF,extent=EXT,origin="upper",interpolation="nearest",zorder=0)
    if contour:
        xx=np.linspace(EXT[0],EXT[1],Z.shape[1]);yy=np.linspace(EXT[3],EXT[2],Z.shape[0])
        step=4
        ax.contour(xx[::step],yy[::step],Z[::step,::step],levels=[200,400,600,800,1000],
                   colors="#263949",linewidths=.55,alpha=.42,zorder=1)
    ax.set(xlim=BOUNDS[:2],ylim=BOUNDS[2:],aspect="equal")
    ax.set_title(f"{chr(97+num)}  {title}",loc="left",pad=9,fontweight="bold")
    if num%2==0:ax.set_ylabel("相对O01北向距离（km）")
    if num>=2:ax.set_xlabel("相对O01东向距离（km）")
    ax.tick_params(width=1.1,length=4)
    for s in ax.spines.values():s.set_linewidth(1.2)

def draw_sites(ax,mapping=None,all_labels=False,bubble=False):
    for r,xy in zip(sites.itertuples(),site_xy):
        c="white" if mapping is None else COL[mapping[r.site_id]]
        ax.scatter(*xy,s=64+8*r.n_boxes if bubble else 62,facecolor=c,
                   edgecolor="#272727" if mapping is None else "white",
                   linewidth=1.0,zorder=10)
        if all_labels or r.site_id in ("S006","S011","S008","S015"):
            ox=-6 if r.site_id in ("S005","S007","S009","S011") else 7
            oy=-8 if r.site_id in ("S006","S010","S011","S013") else 6
            t=ax.annotate(r.site_id,xy,xytext=(ox,oy),textcoords="offset points",
                          ha="right" if ox<0 else "left",fontsize=10.5,
                          fontweight="bold",zorder=12)
            halo(t)
    ax.scatter(0,0,s=220,marker="*",c="#111111",ec="white",lw=1.2,zorder=14)
    t=ax.annotate("O01",(0,0),xytext=(0,-21),textcoords="offset points",ha="center",
                  fontsize=11,fontweight="bold",zorder=15);halo(t)

def draw_routes(ax,mapping=None):
    drawn=set()
    for fid,seq in paths.items():
        if mapping is None:col="#273849"
        else:col=COL[mapping[next(s for s in seq if s!="O01")]]
        for p,q in zip(seq[:-1],seq[1:]):
            key=(tuple(sorted((p,q))),col)
            if key in drawn:continue
            drawn.add(key)
            xs=np.array([node_xy[p],node_xy[q]])
            ax.plot(xs[:,0],xs[:,1],color="white",lw=3.8,zorder=4,alpha=.86)
            ax.plot(xs[:,0],xs[:,1],color=col,lw=2.0,zorder=5,alpha=.95)
    return len(drawn)

def draw_relays(ax):
    coords=plane(relays["经度"],relays["纬度"])
    for row,xy in zip(relays.itertuples(),coords):
        ax.plot([0,xy[0]],[0,xy[1]],color="#212121",ls=(0,(4,3)),lw=1.45,zorder=6)
        ax.scatter(*xy,s=104,marker="D",facecolor="#A50026",edgecolor="white",lw=1,zorder=11)
        label=getattr(row,"中继架次")
        t=ax.annotate(label,xy,xytext=(6,9),textcoords="offset points",
                      fontsize=11,fontweight="bold",zorder=12);halo(t)

def add_group_sampling(ax,sid):
    labels=hexes["group_"+sid].to_numpy()
    ax.add_collection(PolyCollection(poly,facecolors=[COL[x] for x in labels],edgecolors="none",
                                   alpha=.20,zorder=2))
    ax.add_collection(LineCollection(borders_for(labels),colors="#303030",
                                     linewidths=1.0,alpha=.7,zorder=3))
    ax.add_patch(Circle((0,0),radius=.46,facecolor="white",edgecolor="none",alpha=.8,zorder=7))

def fig_terrain():
    fig,ax=plt.subplots(2,2,figsize=(19,12.9),gridspec_kw={
        "left":.060,"right":.987,"bottom":.180,"top":.953,"wspace":.09,"hspace":.17})
    titles=["30米修正DEM与15个服务区","固定问题三：运输航线与中继回传",
            "两组任务 · K2_02","三组任务 · K3_01"]
    for i,aa in enumerate(ax.flat):map_base(aa,titles[i],i,contour=i<2)
    draw_sites(ax[0,0],all_labels=True,bubble=True)
    draw_routes(ax[0,1]);draw_relays(ax[0,1]);draw_sites(ax[0,1],all_labels=False)
    for aa,sid in ((ax[1,0],K2),(ax[1,1],K3)):
        add_group_sampling(aa,sid);draw_routes(aa,lookup[sid]);draw_sites(aa,lookup[sid],all_labels=True)
    ax[0,0].annotate("北",(.055,.89),xycoords="axes fraction",ha="center",fontsize=13)
    ax[0,0].annotate("",xy=(.055,.875),xytext=(.055,.790),xycoords="axes fraction",
                      arrowprops=dict(arrowstyle="-|>",lw=1.4,color="black"))
    xx=BOUNDS[0]+.45;yy=BOUNDS[2]+.35
    ax[0,0].plot([xx,xx+2],[yy,yy],c="#202020",lw=2.1,zorder=16)
    ax[0,0].text(xx+1,yy+.17,"2 km",ha="center",fontsize=10.8,zorder=16)
    h=[Patch(fc=COL[g],label=f"第{i}组") for i,g in enumerate(COL,1)]
    h += [Line2D([],[],marker="*",color="#111111",ls="",markersize=12,label="O01"),
          Line2D([],[],marker="D",color="#A50026",ls="",markersize=7,label="中继悬停"),
          Line2D([],[],color="#252525",ls="--",lw=1.5,label="中继回传")]
    fig.legend(handles=h,ncol=6,loc="lower center",bbox_to_anchor=(.5,.109),frameon=False,fontsize=12)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1150),cmap=DEM),
                    cax=fig.add_axes([.364,.080,.27,.011]),orientation="horizontal")
    cb.set_ticks([0,300,600,900,1150]);cb.ax.tick_params(labelsize=10)
    fig.text(.5,.041,"工作DEM高程（m）",ha="center",fontsize=11)
    fig.text(.5,.010,"20个运输架次、3个中继架次均来自已核验的问题三；线路为平面投影。淡色范围仅由最近服务点蜂巢采样生成，并非行政或人口边界。",
             ha="center",fontsize=10.7)
    return save(fig,"问题四_图07_真实DEM地形与固定航线分组")

def forest_edges():
    parent={x:x for x in sites.site_id}
    def find(v):
        while parent[v]!=v:v=parent[v]
        return v
    kept=[]
    for r in links.sort_values("relation",ascending=False).itertuples():
        u,v=find(r.site_a),find(r.site_b)
        if u==v:continue
        parent[v]=u;kept.append(r)
    assert len(kept)==12
    return kept

def fig_partition():
    fig=plt.figure(figsize=(17.8,10.8))
    gs=fig.add_gridspec(2,2,left=.07,right=.98,bottom=.103,top=.949,wspace=.18,hspace=.31,
                        width_ratios=[1.16,1.0],height_ratios=[1.12,.88])
    aa=fig.add_subplot(gs[0,0]);bb=fig.add_subplot(gs[0,1]);cc=fig.add_subplot(gs[1,0]);dd=fig.add_subplot(gs[1,1])
    map_base(aa,"必须同组的关系图：主组件13区",0)
    for row in forest_edges():
        p,q=np.array(node_xy[row.site_a]),np.array(node_xy[row.site_b])
        relay=row.relation=="同一中继架次保障"
        aa.plot([p[0],q[0]],[p[1],q[1]],color="#A50026" if relay else "#414141",
                ls="--" if relay else "-",lw=1.9 if relay else 2.6,alpha=.78,zorder=6)
    comp=components.set_index("site_id").component_id.to_dict()
    labels={"C1":"#314F8C","C2":"#F07847","C3":"#A50026"}
    draw_sites(aa,{site:{"C1":"G1","C2":"G2","C3":"G3"}[comp[site]] for site in comp},all_labels=True)
    aa.legend(handles=[Line2D([],[],color="#333333",lw=2.3,label="同一运输架次"),
        Line2D([],[],color="#A50026",ls="--",lw=2,label="同一中继架次"),
        Patch(fc="#314F8C",label="C1 · 13区")],loc="upper left",frameon=True,framealpha=.85,fontsize=9)
    aa.set_xlabel("相对O01东向距离（km）")
    # Four exhaustive, unlabeled candidates expressed as a categorical ownership matrix.
    sidlist=list(scenarios.index)
    for col,sid in enumerate(sidlist):
        for row,site in enumerate(sites.site_id):
            g=lookup[sid][site]
            bb.add_patch(Rectangle((col-.43,row-.43),.86,.86,fc=COL[g],ec="white",lw=1.4))
            bb.text(col,row,g,ha="center",va="center",color="white",fontsize=9.3)
    bb.set(xlim=(-.52,3.52),ylim=(14.6,-.65),xticks=range(4),xticklabels=sidlist,
           yticks=range(15),yticklabels=list(sites.site_id))
    bb.xaxis.tick_top();bb.tick_params(length=0,labelsize=10)
    bb.set_title("b  全部合法分区：两组三种、三组一种",loc="left",fontweight="bold",pad=20)
    bb.spines[:].set_visible(False)
    for i,site in enumerate(sites.site_id):
        if comp[site] in ("C2","C3"):
            bb.text(3.52,i,comp[site],va="center",ha="left",fontsize=9.5,color="#333333")
    cc.set_title("c  不可拆分组件承载的货箱",loc="left",fontweight="bold")
    csum=components.groupby("component_id").agg(sites=("site_id","size"),boxes=("n_boxes","sum"))
    for i,cid in enumerate(["C1","C2","C3"]):
        n=int(csum.loc[cid,"boxes"])
        cc.barh(i,n,color=labels[cid],height=.53,edgecolor="white")
        cc.text(n+1.1,i,f"{n}箱 · {int(csum.loc[cid,'sites'])}区",va="center",fontsize=12)
    cc.set(yticks=range(3),yticklabels=["C1 · 主组件","C2 · S006","C3 · S011"],
           xlim=(0,87),xlabel="不可拆分货箱数")
    cc.invert_yaxis();cc.grid(axis="x",alpha=.14)
    dd.set_title("d  资源缺口与运输工时不平等",loc="left",fontweight="bold")
    for sid,row in scenarios.iterrows():
        dd.scatter(row.shortage_count,row.transport_workload_gini_normalized,s=120,
                   c="#A50026" if sid==K2 else "#314F8C",
                   marker="D" if row.n_groups==3 else "o",ec="white",zorder=3)
        o={"K2_01":(7,13),"K2_02":(7,-21),"K2_03":(-68,-13),"K3_01":(7,6)}[sid]
        dd.annotate(sid+(' ★' if sid==K2 else ''),(row.shortage_count,row.transport_workload_gini_normalized),
                    xytext=o,textcoords="offset points",fontsize=10.5)
    dd.set(xlim=(1.5,5),ylim=(.825,.97),xticks=[2,3,4],
           xlabel="需增配件数",ylabel="归一化运输工时Gini")
    dd.grid(alpha=.14)
    for ax in (cc,dd):ax.spines[["top","right"]].set_visible(False)
    fig.text(.5,.030,"同组边是任务和通信的逻辑约束，不表示新航线；13区主组件不能在保持问题三中继与运输任务不变的条件下拆开。",
             ha="center",fontsize=11.4)
    return save(fig,"问题四_图08_必须同组约束与完整候选分区")

RESOURCE_ORDER=["transport_drone_A","transport_drone_B","transport_drone_C",
                "transport_battery_A","transport_battery_B","transport_battery_C",
                "relay_drone_R","relay_component_R"]
RNAME={"transport_drone_A":"A型运输机","transport_drone_B":"B型运输机",
       "transport_drone_C":"C型运输机","transport_battery_A":"A型电池",
       "transport_battery_B":"B型电池","transport_battery_C":"C型电池",
       "relay_drone_R":"中继无人机","relay_component_R":"中继能源组件"}

def allocation(ax,sid,title):
    rows=groupres[groupres.scenario_id.eq(sid)]
    summary=resources[resources.scenario_id.eq(sid)].set_index("resource_key")
    for j,key in enumerate(RESOURCE_ORDER):
        frame=rows[rows.resource_key.eq(key)].sort_values("group_id")
        left=0
        for r in frame.itertuples():
            n=int(r.required_count)
            if n:
                ax.barh(j,n,left=left,height=.54,color=COL[r.group_id],edgecolor="white",lw=1,zorder=3)
                if n>=1:ax.text(left+n/2,j,str(n),ha="center",va="center",fontsize=10,color="white",fontweight="bold")
            left+=n
        stock=int(summary.loc[key,"inventory"]);short=int(summary.loc[key,"shortage"])
        ax.scatter(stock,j,marker="|",s=420,c="black",lw=2,zorder=5)
        ax.text(7.5,j,f"{left}/{stock}"+(f"  +{short}" if short else ""),
                ha="right",va="center",fontsize=10.5,color="#A50026" if short else "#202020")
        assert left==int(summary.loc[key,"sum_required"])
    ax.set(yticks=range(8),yticklabels=[RNAME[k] for k in RESOURCE_ORDER],
           xlim=(0,7.7),xticks=[0,2,4,6],xlabel="最低专属配置（件）")
    ax.invert_yaxis();ax.grid(axis="x",alpha=.15,zorder=0)
    ax.set_title(title,loc="left",fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

def fig_resource():
    fig=plt.figure(figsize=(18.5,11.2))
    gs=fig.add_gridspec(2,2,left=.105,right=.978,bottom=.11,top=.931,
                        wspace=.31,hspace=.29,height_ratios=[1.45,1])
    ax1=fig.add_subplot(gs[0,0]);ax2=fig.add_subplot(gs[0,1]);ax3=fig.add_subplot(gs[1,:])
    allocation(ax1,K2,"a  K2_02：各组专属配置与库存")
    allocation(ax2,K3,"b  K3_01：各组专属配置与库存")
    key_order=["transport_drone_B","transport_battery_B","transport_drone_C","transport_battery_C"]
    color_by={key:"#E97850" if key.endswith("_B") else "#A50026" for key in key_order}
    for i,sid in enumerate(scenarios.index):
        f=resources[resources.scenario_id.eq(sid)].set_index("resource_key")
        left=0
        for key in key_order:
            n=int(f.loc[key,"shortage"])
            if n:
                ax3.barh(i,n,left=left,height=.51,fc=color_by[key],ec="white",hatch="//" if "battery" in key else None)
                left+=n
        ax3.text(left+.12,i,f"缺{left}件  /  专属配置{int(scenarios.loc[sid,'total_required'])}件",
                 va="center",fontsize=12)
        assert left==int(scenarios.loc[sid,"shortage_count"])
    ax3.set(yticks=range(4),yticklabels=list(scenarios.index),xlim=(0,7),
            xticks=[0,1,2,3,4],xlabel="按类别分别核算的新增资源（件）")
    ax3.invert_yaxis();ax3.spines[["top","right"]].set_visible(False)
    ax3.grid(axis="x",alpha=.15)
    ax3.set_title("c  所有候选方案的分类缺口与专属配置规模",loc="left",fontweight="bold")
    handles=[Patch(fc=COL[g],label=f"第{j}组") for j,g in enumerate(COL,1)]
    handles += [Line2D([],[],color="black",marker="|",ls="",ms=19,label="现有库存"),
                Patch(fc="#E97850",label="B型机/电池缺口"),Patch(fc="#A50026",label="C型机/电池缺口")]
    fig.legend(handles=handles,ncol=6,loc="upper center",bbox_to_anchor=(.54,.997),
               frameon=False,fontsize=11)
    fig.text(.5,.027,"分组后资源不得跨组调配；K2_02需增配C型机和电池各1件，K3_01还需增配B型机和电池各1件。电池与无人机不能互相抵扣。",
             ha="center",fontsize=11.4)
    return save(fig,"问题四_图09_专属资源配置与分类库存缺口")

def draw_gantt(ax,sid,key,title,letter):
    frame=intervals[(intervals.scenario_id.eq(sid))&(intervals.resource_key.eq(key))].copy()
    ids=sorted(frame.assigned_id.unique(),key=lambda x:(x.split("-")[0],x))
    assert len(ids)==int(resources[(resources.scenario_id.eq(sid))&
                                   (resources.resource_key.eq(key))].sum_required.iloc[0])
    flights_end=flights.set_index("架次编号")["返回秒"].to_dict()
    for i,rid in enumerate(ids):
        selected=frame[frame.assigned_id.eq(rid)]
        g=str(selected.group_id.iloc[0]);c=COL[g]
        for row in selected.itertuples():
            x=row.start_s/3600;width=(row.end_s-row.start_s)/3600
            if "battery" in key:
                end=min(float(flights_end[row.task_id]),row.end_s)/3600
                ax.barh(i,end-x,left=x,height=.59,color=c,edgecolor="white",lw=.45,zorder=3)
                if row.end_s/3600>end:
                    ax.barh(i,row.end_s/3600-end,left=end,height=.59,color=c,
                            edgecolor="white",lw=.55,hatch="////",alpha=.53,zorder=3)
            else:
                ax.barh(i,width,left=x,height=.59,color=c,edgecolor="white",lw=.55,zorder=3)
            if width>.12:
                ax.text(x+width/2,i,row.task_id,ha="center",va="center",fontsize=8.3,
                        color="white" if "battery" not in key else "#101010",zorder=4)
        if i and ids[i-1].split("-")[0]!=g:ax.axhline(i-.5,c="#636363",ls=":",lw=1)
    ax.set(yticks=range(len(ids)),yticklabels=[i.split("-")[0]+"-"+i.rsplit("-",1)[-1] for i in ids],
           xlim=(0,2.43),xticks=[0,.5,1,1.5,2],xlabel="距任务开始（小时）")
    ax.invert_yaxis();ax.grid(axis="x",alpha=.13)
    ax.spines[["top","right"]].set_visible(False)
    need=len(ids);stock=int(resources[(resources.scenario_id.eq(sid))&
                                  (resources.resource_key.eq(key))].inventory.iloc[0])
    ax.set_title(f"{letter}  {title}  {need}/{stock}",loc="left",fontweight="bold")

def fig_intervals():
    fig,ax=plt.subplots(2,2,figsize=(17.6,10.8),gridspec_kw={
        "left":.082,"right":.981,"bottom":.125,"top":.91,"wspace":.16,"hspace":.33})
    specs=[(K2,"transport_drone_C","K2_02 · C型运输机"),
           (K2,"transport_battery_C","K2_02 · C型共享电池"),
           (K3,"transport_drone_B","K3_01 · B型运输机"),
           (K3,"transport_battery_B","K3_01 · B型共享电池")]
    for i,(sid,key,title) in enumerate(specs):draw_gantt(ax.flat[i],sid,key,title,chr(97+i))
    handles=[Patch(fc=COL[g],label=f"第{j}组")for j,g in enumerate(COL,1)]
    handles+=[Patch(fc="#787878",hatch="////",alpha=.5,label="返航后充电")]
    fig.legend(handles=handles,ncol=4,loc="upper center",bbox_to_anchor=(.5,.98),
               frameon=False,fontsize=12.2)
    fig.text(.5,.029,"横条对应原问题三的20个运输架次；同色每行是一件组内专属资源。G1/G2/G3的峰值不必同时出现，仍须分别预留设备及电池。",
             ha="center",fontsize=11.3)
    return save(fig,"问题四_图10_资源独占占用与充电时序证据")

def fig_overhead():
    fig,axs=plt.subplots(2,2,figsize=(17.7,10.5),gridspec_kw={
        "left":.085,"right":.978,"bottom":.115,"top":.887,"wspace":.22,"hspace":.31})
    sc=scenarios.loc[["K2_01","K2_02","K2_03","K3_01"]]
    shared=int(resources[resources.scenario_id.eq(K2)].global_peak.sum())
    inventory_total=int(resources[resources.scenario_id.eq(K2)].inventory.sum())
    assert shared==26 and inventory_total==30
    ax=axs[0,0]
    for j,row in enumerate(sc.itertuples()):
        ax.barh(j,shared,fc="#314F8C",height=.57,label="不分组的逐类峰值之和" if j==0 else None)
        ax.barh(j,row.partition_overhead,left=shared,fc="#F07847",height=.57,
                label="专属配置新增" if j==0 else None)
        ax.text(row.total_required+.25,j,str(row.total_required)+"件",va="center",fontsize=11)
    ax.axvline(inventory_total,c="#A50026",lw=1.7,ls="--",label="分类库存件数合计")
    ax.set(yticks=range(4),yticklabels=sc.index.tolist(),xlim=(0,34),
           xticks=[0,10,20,26,28,30],xlabel="所需专属配置数量（件）")
    ax.invert_yaxis();ax.grid(axis="x",alpha=.13)
    ax.set_title("a  不分组共享峰值与分区后新增配置",loc="left",fontweight="bold")
    ax=axs[0,1]
    yy=np.arange(4)
    ax.barh(yy-.18,sc.shortage_count,height=.32,fc="#A50026",label="分类资源缺口")
    ax.barh(yy+.18,sc.unused_inventory,height=.32,fc="#A6BDD0",label="不能抵扣的闲置库存")
    for j,row in enumerate(sc.itertuples()):
        ax.text(row.shortage_count+.10,j-.18,str(row.shortage_count),va="center",fontsize=10.4)
        ax.text(row.unused_inventory+.10,j+.18,str(row.unused_inventory),va="center",fontsize=10.4)
    ax.set(yticks=yy,yticklabels=sc.index.tolist(),xlim=(0,5.5),xticks=[0,1,2,3,4,5],
           xlabel="设备或电池件数")
    ax.invert_yaxis();ax.legend(frameon=False,loc="lower right",fontsize=10.5)
    ax.grid(axis="x",alpha=.13)
    ax.set_title("b  缺口与闲置必须分别计数",loc="left",fontweight="bold")
    styles=[("transport_occupancy_h","运输机","#31538C"),
            ("battery_occupancy_h","共享电池","#7E9FC2"),
            ("relay_occupancy_h","中继机","#F07847"),
            ("component_occupancy_h","中继组件","#A50026")]
    for col,sid in enumerate((K2,K3)):
        aa=axs[1,col]
        frame=grouptot[grouptot.scenario_id.eq(sid)].sort_values("group_id")
        for j,r in enumerate(frame.itertuples()):
            left=0
            for key,label,c in styles:
                amount=float(getattr(r,key))
                aa.barh(j,amount,left=left,height=.54,color=c,edgecolor="white",
                        label=label if j==0 else None)
                left+=amount
            aa.text(left+.45,j,f"{r.n_boxes}箱  {left:.1f}小时",va="center",fontsize=10.8)
        aa.set(yticks=range(len(frame)),yticklabels=[f"第{i}组" for i in range(1,len(frame)+1)],
               xlim=(0,45),xlabel="设备累计占用工时（小时）")
        aa.invert_yaxis();aa.grid(axis="x",alpha=.14)
        aa.set_title(f"{chr(99+col)}  {sid}：组间任务负担与货箱",loc="left",fontweight="bold")
        if col==0:aa.legend(frameon=False,ncol=2,loc="lower right",fontsize=9.5)
    for ax in axs.flat:ax.spines[["top","right"]].set_visible(False)
    h,l=axs[0,0].get_legend_handles_labels()
    fig.legend(h,l,ncol=3,loc="upper center",bbox_to_anchor=(.50,.985),
               frameon=False,fontsize=11)
    fig.text(.5,.035,"虽然全部分类库存合计为30件，B/C型机和电池仍存在无法跨类别抵消的缺口；组间设备占用小时不等于货箱等待时间。",
             ha="center",fontsize=11.5)
    return save(fig,"问题四_图11_分区资源冗余与组间负担分解")

out=[fig_partition(),fig_resource(),fig_intervals(),fig_overhead()]
assert len(out)==4
report={"status":"PASS","transport_sorties":len(flights),"relay_sorties":len(relays),
 "boxes":len(delivery),"dem_sha256":digest(DEM_PATH),"same_group_spanning_edges":len(forest_edges()),
 "candidate_count":{"2":3,"3":1},"all_tasks_fixed":True,"relay_not_duplicated":True,
 "known_shortages":{sid:int(r.shortage_count) for sid,r in scenarios.iterrows()},
 "outputs":out}
(a.output_dir/"问题四_扩展图件核验.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
