"""Q4 spatial distance and inequality supplement for the fixed, certified Q3 plan.

Run after compute_q4.py: python plot_q4_equity_hex.py
The hexagons are a sampling grid, NOT inhabited cells, administrative polygons,
road distances, extra delivery nodes, or additional Q3 tasks.
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
from matplotlib import patheffects
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch, Circle
from matplotlib.colors import LightSource
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import ConvexHull, cKDTree

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
parser.add_argument("--q3-dir", type=Path, default=ROOT / "q3_source")
parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
parser.add_argument("--dpi", type=int, default=300)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

font_path = args.q3_dir / "q1_flat" / "中文字体_绘图临时.otf"
from matplotlib import font_manager
font_manager.fontManager.addfont(str(font_path))
CN = font_manager.FontProperties(fname=str(font_path)).get_name()
plt.rcParams.update({
    "font.family": ["STIXGeneral", CN], "axes.unicode_minus": False,
    "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 14,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "pdf.fonttype": 42,
    "text.color": "#151515", "axes.labelcolor": "#151515",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
GROUP_COLORS = {"G1": "#304B86", "G2": "#F07445", "G3": "#A50026"}
DIST_CMAP = LinearSegmentedColormap.from_list(
    "distance", ["#E8F4FA", "#A6D7DC", "#F8CD8B", "#E76B45", "#A50026"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gini(values, weights=None):
    """Weighted Gini on a nonnegative burden; 0 means uniform burden."""
    x = np.asarray(values, float)
    w = np.ones(len(x), float) if weights is None else np.asarray(weights, float)
    assert len(x) and np.all(x >= 0) and np.all(w > 0) and x.sum() > 0
    order = np.argsort(x, kind="stable"); x = x[order]; w = w[order]
    cw = np.r_[0., np.cumsum(w) / w.sum()]
    cx = np.r_[0., np.cumsum(x * w) / np.dot(x, w)]
    return float(1 - np.sum((cx[1:] + cx[:-1]) * np.diff(cw)))


def lorenz(values):
    x = np.sort(np.asarray(values, float))
    return np.arange(len(x) + 1) / len(x), np.r_[0, np.cumsum(x) / x.sum()]


def local_km(lon, lat, lon0, lat0):
    # Small-area tangent plane, solely for drawing the regular hexagon mesh.
    return np.column_stack(((np.asarray(lon) - lon0) * 111.32 * math.cos(math.radians(lat0)),
                            (np.asarray(lat) - lat0) * 111.132))


def from_local(x, y, lon0, lat0):
    return lon0 + x / (111.32 * math.cos(math.radians(lat0))), lat0 + y / 111.132


def read_terrain(path, lon0, lat0, lim, colored=False):
    with Image.open(path) as im:
        z = np.asarray(im, dtype=np.float32)
        scale = im.tag_v2[33550]; tie = im.tag_v2[33922]
        geokeys = tuple(im.tag_v2[34735]); raster_type = None
        for i in range(4, len(geokeys), 4):
            if geokeys[i] == 1025: raster_type = geokeys[i + 3]
    assert raster_type in (1, 2) and z.ndim == 2 and np.isfinite(z).all()
    dx, dy = float(scale[0]), float(scale[1])
    left = float(tie[3]) - (dx / 2 if raster_type == 2 else 0.)
    top = float(tie[4]) + (dy / 2 if raster_type == 2 else 0.)
    lon_l, lat_b = from_local(lim[0] - .4, lim[2] - .4, lon0, lat0)
    lon_r, lat_t = from_local(lim[1] + .4, lim[3] + .4, lon0, lat0)
    c0 = max(0, int((lon_l - left) / dx)); c1 = min(z.shape[1], int((lon_r - left) / dx) + 2)
    r0 = max(0, int((top - lat_t) / dy)); r1 = min(z.shape[0], int((top - lat_b) / dy) + 2)
    part = z[r0:r1, c0:c1]
    assert part.shape[0] > 10 and part.shape[1] > 10
    shade = LightSource(azdeg=310, altdeg=52).hillshade(part, dx=30, dy=30, vert_exag=1.25)
    if colored:
        rgb = np.clip(plt.colormaps["RdYlBu_r"](Normalize(0, 1150)(part))[..., :3]
                      * (.76 + .24 * shade[..., None]), 0, 1)
    else:
        # Background only; the colors on top measure distance or task ownership.
        rgb = np.repeat((.89 + .11 * shade)[..., None], 3, axis=2)
    ll, bb = from_local(0, 0, lon0, lat0)
    xe = ((left + np.array([c0, c1]) * dx) - ll) * 111.32 * math.cos(math.radians(lat0))
    ye = ((top - np.array([r1, r0]) * dy) - bb) * 111.132
    return rgb, [xe[0], xe[1], ye[0], ye[1]]


def hex_grid(xy, radius=.51, hull_buffer=1.03):
    """Keep hex centers at most hull_buffer km from the sites' convex hull."""
    hull = xy[ConvexHull(xy).vertices]
    polygon = MplPath(hull, closed=True)
    step_y = np.sqrt(3) * radius; step_x = 1.5 * radius
    xcols = np.arange(xy[:, 0].min() - 2, xy[:, 0].max() + 2.01, step_x)
    centers = []
    for j, x in enumerate(xcols):
        y0 = math.floor((xy[:, 1].min() - 2) / step_y) * step_y
        for y in np.arange(y0, xy[:, 1].max() + 2.01, step_y):
            p = np.array([x, y + (j % 2) * step_y / 2])
            edge_dist = min(float(np.linalg.norm(p - (a + np.clip(np.dot(p-a, b-a) /
                       np.dot(b-a, b-a), 0, 1) * (b-a)))) for a, b in zip(hull, np.roll(hull, -1, axis=0)))
            if polygon.contains_point(p) or edge_dist <= hull_buffer:
                centers.append(p)
    centers = np.asarray(centers)
    verts = np.array([[radius * math.cos(k * math.pi / 3), radius * math.sin(k * math.pi / 3)]
                      for k in range(6)])
    return centers, centers[:, None, :] + verts[None, :, :]


def polygon_group_boundaries(polys, group_labels):
    adjacency = {}
    for i, poly in enumerate(polys):
        for p, q in zip(poly, np.roll(poly, -1, axis=0)):
            edge = tuple(sorted((tuple(np.round(p, 5)), tuple(np.round(q, 5)))))
            adjacency.setdefault(edge, []).append(i)
    return [edge for edge, idx in adjacency.items()
            if len(idx) == 2 and group_labels[idx[0]] != group_labels[idx[1]]]


def save(fig, stem):
    p = args.output_dir / stem
    tmp_png = args.output_dir / (stem + ".tmp.png")
    tmp_pdf = args.output_dir / (stem + ".tmp.pdf")
    fig.savefig(tmp_png, dpi=args.dpi, bbox_inches="tight", pad_inches=.12)
    fig.savefig(tmp_pdf, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)
    with Image.open(tmp_png) as im: im.verify()
    with Image.open(tmp_png) as im: im.load(); dims = list(im.size)
    assert tmp_pdf.stat().st_size > 5000
    tmp_png.replace(p.with_suffix(".png"))
    tmp_pdf.replace(p.with_suffix(".pdf"))
    return {"png": p.with_suffix(".png").name, "pdf": p.with_suffix(".pdf").name,
            "size_px": dims, "png_sha256": sha(p.with_suffix(".png"))}


nodes = pd.read_csv(args.data_dir / "nodes.csv")
groups = pd.read_csv(args.data_dir / "site_groups.csv")
summaries = pd.read_csv(args.data_dir / "group_summary.csv")
scenarios = pd.read_csv(args.data_dir / "scenarios.csv")
deliveries = pd.read_csv(args.data_dir / "sources" / "deliveries.csv")
legs = pd.read_csv(args.data_dir / "sources" / "legs.csv")
stations = nodes[nodes.site_id.ne("O01")].sort_values("site_id").copy()
assert len(stations) == 15 and stations.site_id.is_unique
assert len(deliveries) == 80 and deliveries["货箱编号"].is_unique
assert len(scenarios) == 4 and len(summaries) == 9
two = scenarios[(scenarios.n_groups.eq(2)) &
                (scenarios.recommended.astype(str).str.lower().eq("true"))].scenario_id.item()
three = scenarios[scenarios.n_groups.eq(3)].scenario_id.item()
assert two == "K2_02" and three == "K3_01"

# An exact horizontal O01 -> Si flight-leg length is already in the frozen Q3 model.
outbound = legs[legs["起点"].eq("O01")].groupby("终点")["距离m"]
assert set(outbound.groups) == set(stations.site_id)
assert (outbound.max() - outbound.min()).max() < 1e-6
stations["base_to_site_km"] = stations.site_id.map(outbound.first()) / 1000.
visit = deliveries.groupby("服务区")["送达秒"].agg(["min", "median", "max", "mean", "count"])
stations["first_delivery_s"] = stations.site_id.map(visit["min"])
stations["last_delivery_s"] = stations.site_id.map(visit["max"])
stations["mean_delivery_s"] = stations.site_id.map(visit["mean"])
stations["n_boxes"] = stations.site_id.map(visit["count"]).astype(int)
assert int(stations.n_boxes.sum()) == 80
stations["component_id"] = stations.site_id.map(
    pd.read_csv(args.data_dir / "components.csv").set_index("site_id").component_id)
for sid in (two, three):
    mask = groups.scenario_id.eq(sid)
    stations[f"group_{sid}"] = stations.site_id.map(groups[mask].set_index("site_id").group_id)
assert stations[[f"group_{two}", f"group_{three}"]].notna().all().all()

origin = nodes[nodes.site_id.eq("O01")].iloc[0]
lon0, lat0 = float(origin.longitude), float(origin.latitude)
xy = local_km(stations.longitude, stations.latitude, lon0, lat0)
centers, polys = hex_grid(xy)
nearest_d, near_idx = cKDTree(xy).query(centers)
near = stations.iloc[near_idx].reset_index(drop=True)
assert len(centers) > 120 and nearest_d.max() < 4
assert np.allclose(near.base_to_site_km.to_numpy(), stations.base_to_site_km.to_numpy()[near_idx])
hx_lon, hx_lat = from_local(centers[:, 0], centers[:, 1], lon0, lat0)
hexes = pd.DataFrame({
    "hex_id": [f"H{i:03d}" for i in range(1, len(centers)+1)],
    "center_x_km": centers[:, 0], "center_y_km": centers[:, 1],
    "center_lon": hx_lon, "center_lat": hx_lat,
    "nearest_site_id": near.site_id, "center_to_nearest_site_km": nearest_d,
    "base_to_nearest_site_km": near.base_to_site_km,
    f"group_{two}": near[f"group_{two}"], f"group_{three}": near[f"group_{three}"],
})
assert hexes.hex_id.is_unique and not hexes.isna().any().any()
hexes.to_csv(args.data_dir / "hex_distance_cells.csv", index=False, encoding="utf-8-sig")
stations.to_csv(args.data_dir / "site_access_equity.csv", index=False, encoding="utf-8-sig")

ineq = []
for s in scenarios.itertuples():
    grp = summaries[summaries.scenario_id.eq(s.scenario_id)].sort_values("group_id")
    assert len(grp) == s.n_groups
    ineq.append({
        "scenario_id": s.scenario_id, "n_groups": int(s.n_groups),
        "shortage_count": int(s.shortage_count), "total_required": int(s.total_required),
        "group_workload_gini": gini(grp.total_workload_h),
        "group_workload_gini_normalized": gini(grp.total_workload_h)/(1-1/s.n_groups),
        "transport_workload_gini": gini(grp.transport_workload_h),
        "transport_workload_gini_normalized": gini(grp.transport_workload_h)/(1-1/s.n_groups),
        "relay_workload_gini_normalized": gini(grp.relay_workload_h)/(1-1/s.n_groups),
        "group_workload_cv": float(s.workload_cv),
        "group_box_count_gini": gini(grp.n_boxes),
        "site_first_delivery_gini_FIXED": gini(stations.first_delivery_s),
        "box_delivery_gini_FIXED": gini(deliveries["送达秒"]),
        "recommended_resource_first": bool(s.recommended),
    })
ineq = pd.DataFrame(ineq)
ineq.to_csv(args.data_dir / "scenario_equity.csv", index=False, encoding="utf-8-sig")

group_detail = []
for row in summaries.itertuples():
    subset = stations[stations[f"group_{row.scenario_id}"].eq(row.group_id)] if row.scenario_id in (two, three) else \
        stations[stations.site_id.isin(row.sites.split("、"))]
    assert len(subset) == row.n_sites and subset.n_boxes.sum() == row.n_boxes
    ds = deliveries[deliveries["服务区"].isin(subset.site_id)]
    group_detail.append({"scenario_id": row.scenario_id, "group_id": row.group_id,
        "n_sites": len(subset), "n_boxes": int(len(ds)),
        "workload_h": row.total_workload_h, "workload_per_box_h": row.total_workload_h / len(ds),
        "mean_box_delivery_min": ds["送达秒"].mean() / 60,
        "mean_direct_distance_km": float(np.average(subset.base_to_site_km, weights=subset.n_boxes)),
        "site_range_km": float(np.max(np.linalg.norm(local_km(subset.longitude,subset.latitude,lon0,lat0)
                  [:,None,:]-local_km(subset.longitude,subset.latitude,lon0,lat0)[None,:,:],axis=-1))),
    })
pd.DataFrame(group_detail).to_csv(args.data_dir / "group_equity.csv", index=False, encoding="utf-8-sig")


def plot_hex():
    lim = [min(centers[:, 0].min()-.68, xy[:, 0].min()-1.1),
           max(centers[:, 0].max()+.68, xy[:, 0].max()+1.1),
           min(centers[:, 1].min()-.62, xy[:, 1].min()-1.0),
           max(centers[:, 1].max()+.62, xy[:, 1].max()+1.0)]
    terrain, terrain_ext = read_terrain(args.q3_dir / "q1_flat" / "最终工作DEM.tif",
                                        lon0, lat0, lim)
    relief, _ = read_terrain(args.q3_dir / "q1_flat" / "最终工作DEM.tif",
                             lon0, lat0, lim, colored=True)
    fig, axs = plt.subplots(3, 2, figsize=(17.5, 18.2), gridspec_kw={
        "left": .064, "right": .982, "bottom": .083, "top": .965,
        "wspace": .12, "hspace": .23})
    titles = ["修正DEM与实际服务点", "蜂巢到最近服务点的距离",
              "O01到最近服务点的水平航段", "最近服务点的首箱送达时间",
              f"两组任务归属 · {two}", f"三组任务归属 · {three}"]
    vmax_near = float(np.ceil(nearest_d.max() * 2) / 2)
    vmax_base = float(np.ceil(stations.base_to_site_km.max()))
    wait_cmap = LinearSegmentedColormap.from_list("wait", ["#D5EDF1", "#FAD599", "#EF8A5E", "#A50026"])
    raster_maps = {1:(nearest_d, Normalize(0, vmax_near), DIST_CMAP, "距离（km）"),
                   2:(near.base_to_site_km.to_numpy(), Normalize(0,vmax_base), DIST_CMAP, "距离（km）"),
                   3:(near.first_delivery_s.to_numpy()/60, Normalize(0,120), wait_cmap, "首箱送达（分钟）")}
    for i, ax in enumerate(axs.flat):
        ax.imshow(relief if i==0 else terrain, extent=terrain_ext,
                  origin="upper", interpolation="nearest", zorder=0)
        if i in raster_maps:
            value, norm, cm, _ = raster_maps[i]
            colors = cm(norm(value)); colors[:, -1] = .86
            ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors="#394044",
                                             linewidths=.30, alpha=.88, zorder=2))
        elif i >= 4:
            labels = near[f"group_{two if i == 4 else three}"].to_numpy()
            colors = [GROUP_COLORS[g] for g in labels]
            ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors="#394044",
                                             linewidths=.30, alpha=.57, zorder=2))
            borders = polygon_group_boundaries(polys, labels)
            ax.add_collection(LineCollection(borders, colors="#171717", linewidths=1.35,
                                              alpha=.88, zorder=8))
            # O01 is a common external dispatch centre, not a member of a task group.
            ax.add_patch(Circle((0,0), radius=.56, facecolor="white", edgecolor="#303030",
                                linewidth=1.2, alpha=.91, zorder=9))
        for j, site in stations.reset_index(drop=True).iterrows():
            x, y = xy[j]
            color = GROUP_COLORS[site[f"group_{two if i == 4 else three}"]] if i >= 4 else "white"
            ax.scatter(x, y, s=(39+site.n_boxes*8) if i==0 else (44 if i<4 else 58),
                       facecolor=color, edgecolor="#222222" if i<4 else "white",
                       linewidth=1.0, zorder=11)
            ox = -7 if site.site_id in ("S005", "S007", "S009", "S011") else 7
            oy = -11 if site.site_id in ("S006", "S010", "S011", "S013", "S014") else 7
            if i == 2: text = f"{site.site_id} {site.base_to_site_km:.1f}"
            elif i == 3: text = f"{site.site_id} {site.first_delivery_s/60:.0f}"
            elif i == 1: text = site.site_id if site.site_id in ("S003","S006","S008","S011","S015") else ""
            else: text = site.site_id
            if not text: continue
            anno = ax.annotate(text, (x,y), xytext=(ox,oy), textcoords="offset points",
                               ha="right" if ox < 0 else "left", va="center",
                               fontsize=10.4, fontweight="bold", zorder=12, clip_on=False)
            anno.set_path_effects([patheffects.withStroke(linewidth=2.7, foreground="white")])
        ax.scatter(0, 0, s=195, marker="*", facecolor="#161616", edgecolor="white",
                   linewidth=1.0, zorder=13)
        mark = ax.annotate("O01", (0,0), xytext=(0,-20), textcoords="offset points",
                           ha="center", fontsize=12, fontweight="bold", zorder=14)
        mark.set_path_effects([patheffects.withStroke(linewidth=3.2, foreground="white")])
        ax.set(xlim=lim[:2], ylim=lim[2:], aspect="equal")
        ax.set_title(f"{chr(97+i)}  {titles[i]}", loc="left", pad=10, fontweight="bold")
        if i % 2 == 0: ax.set_ylabel("相对O01的北向距离（km）")
        if i >= 4: ax.set_xlabel("相对O01的东向距离（km）")
        ax.tick_params(length=4, width=1.1)
        for spine in ax.spines.values(): spine.set_linewidth(1.2)
    axs[0,0].annotate("北", (.067,.88), xycoords="axes fraction", ha="center", fontsize=13)
    axs[0,0].annotate("", xy=(.067,.865), xytext=(.067,.785), xycoords="axes fraction",
                      arrowprops={"arrowstyle":"-|>", "color":"black", "lw":1.4})
    for idx in (0,1,2,3):
        ax=axs.flat[idx]
        if idx==0:
            norm,cm,title=Normalize(0,1150),plt.colormaps["RdYlBu_r"],"地形高程（m）"
        else:
            _,norm,cm,title=raster_maps[idx]
        bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm), ax=ax,
                         orientation="horizontal",pad=.052,fraction=.031,aspect=26)
        bar.set_label(title,fontsize=11.5,labelpad=2)
    handles = [Patch(facecolor=c, edgecolor="white", label=f"第{j}组")
               for j, (g,c) in enumerate(GROUP_COLORS.items(),1)]
    fig.legend(handles=handles + [Line2D([],[],ls="", marker="*", ms=13,
                    color="#161616",label="调度中心")], ncol=4,
               loc="lower center", bbox_to_anchor=(.52,.045), frameon=False, fontsize=13)
    fig.text(.5, .014, "蜂巢仅表示按最近服务点计算的空间采样；格内人口与货箱未知。O01为共有调度中心，服务点距离、首箱时刻均继承问题三。",
             ha="center", va="center", fontsize=11.8)
    return save(fig,"问题四_图05_服务距离与任务分组蜂巢图")


def plot_equity():
    fig, axs = plt.subplots(2,2,figsize=(15.7,10.5), gridspec_kw={
        "left":.095,"right":.963,"bottom":.11,"top":.92,"wspace":.31,"hspace":.35})
    ax = axs[0,0]
    for row in ineq.itertuples():
        marker = "D" if row.n_groups == 3 else "o"
        color = "#A50026" if row.scenario_id == two else "#405F9A"
        ax.scatter(row.shortage_count, row.transport_workload_gini_normalized, s=125,
                   marker=marker, c=color, ec="white", lw=1.2, zorder=4)
        offset = {"K2_01":(9,12),"K2_02":(9,-22),
                  "K2_03":(-88,-14),"K3_01":(9,7)}[row.scenario_id]
        ax.annotate(row.scenario_id + (" ★" if row.scenario_id==two else ""),
                    (row.shortage_count, row.transport_workload_gini_normalized),
                    xytext=offset, textcoords="offset points", fontsize=11.5, color="black")
    ax.set(xlim=(1.5,5),ylim=(.825,.97),xticks=[2,3,4],
           xlabel="较现有库存需追加的设备/电池件数", ylabel="归一化运输工时 Gini")
    ax.set_title("a  分类缺口与运输工作量不平等",loc="left",fontweight="bold")
    ax.grid(axis="both",alpha=.16)
    ax = axs[0,1]
    yy=[]; labs=[]; demand=[]; work=[]; colors=[]; pos=0
    for sid in (two,three):
        frame = summaries[summaries.scenario_id.eq(sid)].sort_values("group_id")
        for r in frame.itertuples():
            yy.append(pos);labs.append(sid.replace("_0", "-") + " / " + r.group_id)
            demand.append(r.n_boxes/80*100)
            work.append(r.transport_workload_h/frame.transport_workload_h.sum()*100)
            colors.append(GROUP_COLORS[r.group_id]);pos+=1
        pos+=.45
    for y,d,w,col in zip(yy,demand,work,colors):
        ax.plot([d,w],[y,y],color=col,lw=2.1,zorder=2)
        ax.scatter(d,y,marker="o",s=79,facecolor="white",edgecolor=col,lw=2,zorder=3)
        ax.scatter(w,y,marker="s",s=63,facecolor=col,edgecolor="white",lw=.8,zorder=4)
    ax.set(yticks=yy,yticklabels=labs,xlim=(-2,103),xlabel="占全部货箱或运输工时（%）")
    ax.invert_yaxis();ax.grid(axis="x",alpha=.16)
    ax.legend([Line2D([],[],marker="o",ls="",mfc="white",mec="#404040",ms=8),
               Line2D([],[],marker="s",ls="",mfc="#404040",mec="#404040",ms=7)],
              ["货箱份额","运输工时份额"],loc="lower right",frameon=False,fontsize=10.5)
    ax.set_title("b  服务需求与运输负担的组间份额",loc="left",fontweight="bold")
    ax = axs[1,0]
    for row in stations.itertuples():
        color=GROUP_COLORS[getattr(row, f"group_{two}")]
        xx=row.base_to_site_km; yy=row.first_delivery_s/60
        ax.scatter(xx,yy,s=33+13*row.n_boxes,c=color,edgecolor="white",linewidth=1,zorder=4)
        dx=-8 if row.site_id in ("S002","S007","S012","S013") else 7
        dy=-11 if row.site_id in ("S011","S014","S006","S005") else 5
        ax.annotate(row.site_id,(xx,yy),xytext=(dx,dy),textcoords="offset points",
                    fontsize=9.8,ha="right" if dx<0 else "left",color="black")
    ax.set(xlabel="O01至服务点水平航段（km）",ylabel="首箱送达（分钟）",
           xlim=(2.25,8.55),ylim=(3,120))
    ax.grid(alpha=.16);ax.set_title("c  距离与首次保障时刻（15个服务区）",loc="left",fontweight="bold")
    ax = axs[1,1]
    xb,yb=lorenz(deliveries["送达秒"])
    xs,ys=lorenz(stations.first_delivery_s)
    ax.plot([0,1],[0,1],ls="--",color="#999999",lw=1.5,label="完全均等")
    ax.plot(xb,yb,c="#304B86",lw=2.6,label=f"逐箱送达时间  Gini={gini(deliveries['送达秒']):.3f}")
    ax.plot(xs,ys,c="#A50026",lw=2.6,label=f"服务区首箱时刻  Gini={gini(stations.first_delivery_s):.3f}")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="从早到晚排列的累计箱/服务区占比",
           ylabel="累计送达等待时间占比")
    ax.grid(alpha=.14);ax.legend(loc="upper left",frameon=False,fontsize=10.8)
    ax.set_title("d  固定调度的等待时间分布",loc="left",fontweight="bold")
    for a in axs.flat:
        a.spines[["top","right"]].set_visible(False)
        a.tick_params(width=1.1,length=4)
    fig.text(.51,.035,"分组仅调整资源归属：80箱送达与总能耗不变。运输 Gini 按组数上限归一化；中继工时、八类利用率另行展示。",
             ha="center",va="center",fontsize=11.5)
    return save(fig,"问题四_图06_资源不平等与服务等待诊断")


outputs=[plot_hex(),plot_equity()]
max_sample_dist = float(nearest_d.max())
assert all(np.isclose(ineq.box_delivery_gini_FIXED, gini(deliveries["送达秒"])))
assert all(np.isclose(ineq.site_first_delivery_gini_FIXED, gini(stations.first_delivery_s)))
assert all(scenarios.set_index("scenario_id").loc[ineq.scenario_id,"shortage_count"].to_numpy()
           == ineq.shortage_count.to_numpy())
report={"status":"PASS", "input_q3_unchanged":True,
    "q3_transport_sorties":20,"q3_relay_sorties":3,"delivered_boxes":80,
    "valid_partition_count":{"2":3,"3":1},
    "reference_grouping":{"two_groups":two,"three_groups":three},
    "hex_count":len(hexes),"hex_radius_km":.51,"convex_hull_buffer_km":1.03,
    "max_center_to_nearest_site_km":max_sample_dist,
    "hex_distance_definition":"hex centroid to nearest of 15 service sites, local tangent plane",
    "base_distance_definition":"frozen Q3 O01->Si horizontal flight leg, not road or actual multi-stop route length",
    "demand_in_hex_cells":False,"time_or_route_rescheduled":False,
    "group_workload_gini":ineq.set_index("scenario_id").group_workload_gini.to_dict(),
    "normalized_group_workload_gini":ineq.set_index("scenario_id").group_workload_gini_normalized.to_dict(),
    "normalized_transport_workload_gini":ineq.set_index("scenario_id").transport_workload_gini_normalized.to_dict(),
    "unchanged_box_delivery_gini":gini(deliveries["送达秒"]),
    "unchanged_site_first_delivery_gini":gini(stations.first_delivery_s),
    "source_sha256":{"dem":sha(args.q3_dir/"q1_flat"/"最终工作DEM.tif"),
      "legs":sha(args.data_dir/"sources"/"legs.csv"),
      "deliveries":sha(args.data_dir/"sources"/"deliveries.csv"),
      "groups":sha(args.data_dir/"site_groups.csv")},
    "outputs":outputs}
(args.output_dir/"问题四_空间公平核验.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
