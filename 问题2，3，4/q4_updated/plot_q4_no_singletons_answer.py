"""Q4 visual answer: original Q3 3-D DEM and graphical resource comparison.

The V9 flights, relays, intervals, deliveries and eight-category inventory are
fixed. The precise per-group numbers remain in the companion Markdown/CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--data-dir', type=Path, default=ROOT/'no_singletons')
ap.add_argument('--q3-dir', type=Path, default=ROOT.parent/'q3_source')
ap.add_argument('--output-dir', type=Path, default=ROOT/'no_singletons'/'figures')
ap.add_argument('--dpi', type=int, default=230)
a = ap.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=True)
os.environ['Q2_CERTIFIED_PREFIX'] = str(a.q3_dir/'问题三_时间主方案_运输')
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(a.q3_dir))

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import pandas as pd
from PIL import Image
from 问题三_V3_基础绘图 import load_v3
from 问题三_V7_六面板三维山地航线 import Scene, CM, INK

plt.rcParams.update({'pdf.fonttype': 42, 'axes.unicode_minus': False,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                     'font.size': 18, 'axes.labelsize': 17,
                     'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK})
DEFICIT = CM(.985)
GROUP_COLORS = {'G1': CM(.035), 'G2': CM(.78), 'G3': CM(.985)}
KEYS = ['transport_drone_A', 'transport_drone_B', 'transport_drone_C',
        'transport_battery_A', 'transport_battery_B', 'transport_battery_C',
        'relay_drone_R', 'relay_component_R']
LABELS = ['A型运输机', 'B型运输机', 'C型运输机', 'A型电池',
          'B型电池', 'C型电池', '中继无人机', '中继能源组件']

group_resources = pd.read_csv(a.data_dir/'group_resources.csv')
scenario_resources = pd.read_csv(a.data_dir/'scenario_resources.csv')
sites = pd.read_csv(a.data_dir/'site_groups.csv')
summary = pd.read_csv(a.data_dir/'group_summary.csv')
cert = json.loads((a.data_dir/'results.json').read_text())
assert cert['checks']['cp_sat_optimal']
SID = ('R2_不设单站组', 'R3_不设单站组')

f, boxes, relays, comms, intervals, stations, provenance = load_v3(
    str(a.q3_dir/'问题三_时间主方案'),
    str(a.q3_dir/'问题三_时间主方案_精确'),
    str(a.q3_dir/'问题三_V3_站点.csv'))
assert len(f) == 22 and len(boxes) == 80 and len(relays) == 4 and len(intervals)>0
scene = Scene(relays)
stations = stations.set_index('station_id')
used = stations.loc[relays.station_id.unique()]
scene.zlim = (0., float(np.ceil(max(scene.zlim[1], intervals.p0_z.max(),
                                   intervals.p1_z.max(), relays.z.max(),
                                   used.cruise_z.max())/200)*200))


def assignment(sid):
    sub = sites[sites.scenario_id.eq(sid)].set_index('site_id').group_id.to_dict()
    assert len(sub) == 15
    route_group = {}
    for _, flight in f.iterrows():
        visited = set(flight['访问顺序'].split('→')) - {'O01'}
        groups = {sub[x] for x in visited}
        assert len(groups) == 1, (flight['架次编号'], groups)
        route_group[flight['架次编号']] = groups.pop()
    assert len(route_group) == 22
    return sub, route_group


def terrain(ax, sid, letter, title):
    node_groups, route_groups = assignment(sid)
    scene.base(ax, letter, title, axis_labels=False)
    ax.collections[0].set_rasterized(True)
    flight = intervals.copy()
    flight['group'] = flight.route.map(route_groups)
    assert flight.group.notna().all()
    p, q = scene.xyz(flight, 'p0'), scene.xyz(flight, 'p1')
    moving = np.linalg.norm((q-p)*[1, 1, .001], axis=1) > 1e-9
    pieces = np.stack([p[moving], q[moving]], axis=1)
    ax.add_collection3d(Line3DCollection(pieces, colors='white',
                                         linewidths=4.25, zorder=7))
    ax.add_collection3d(Line3DCollection(
        pieces, colors=[GROUP_COLORS[x] for x in flight.loc[moving, 'group']],
        linewidths=3.1, zorder=8))
    site_ids = sorted(node_groups)
    xyz = np.stack([scene.node[x] for x in site_ids])
    ax.scatter(*xyz.T, s=81, c=[GROUP_COLORS[node_groups[x]] for x in site_ids],
               edgecolors='white', linewidths=1.2, depthshade=False, zorder=14)
    for site in ('S006',):
        point = scene.node[site]
        ax.text(point[0]+.12, point[1], point[2]+70, site, fontsize=15,
                color=INK, weight='bold', zorder=20)
    # One geographic relay template can have one dedicated physical copy in each group.
    scene.relay(ax, relays, stations, flight=True)
    for mark in ax.texts:
        if mark.get_text().startswith('J06'):
            mark.set_text('J06');mark.set_fontsize(12)
    return {'transport_flights': len(route_groups), 'relay_flights': len(relays),
            'site_groups': {g: sum(x == g for x in node_groups.values())
                            for g in sorted(set(node_groups.values()))}}


def resources(ax, sid):
    sub = group_resources[group_resources.scenario_id.eq(sid)]
    res = scenario_resources[scenario_resources.scenario_id.eq(sid)].set_index('resource_key')
    groups = sorted(summary[summary.scenario_id.eq(sid)].group_id)
    assert len(groups) == (2 if sid == SID[0] else 3)
    by = sub.set_index(['group_id', 'resource_key'])
    ax.set_xlim(0, 8.30)
    ax.set_ylim(-.83, 7.72)
    ax.set_yticks(range(7, -1, -1), LABELS)
    ax.set_xticks((0, 2, 4, 6))
    ax.tick_params(axis='x', labelsize=13.5, length=0, pad=7, colors=INK)
    ax.tick_params(axis='y', labelsize=15.3, length=0, pad=12, colors=INK)
    ax.set_xlabel('独立配置（架／组）', fontsize=15.0,
                  color=INK, labelpad=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='x', linewidth=.8, color='#d8dee4', alpha=.75, zorder=0)

    for i, key in enumerate(KEYS):
        y = 7-i
        inv = int(res.loc[key, 'inventory'])
        total = int(res.loc[key, 'sum_required'])
        short = int(res.loc[key, 'shortage'])
        assert total == sum(int(by.loc[(g, key), 'required_count']) for g in groups)
        assert short == max(0, total-inv)
        # The pale full-width rail encodes available stock; one fixed scale
        # is used for both recommended partitions and all eight resources.
        ax.barh(y, inv, height=.56, color='#edf0f2', zorder=1)
        start = 0
        for group in groups:
            value = int(by.loc[(group, key), 'required_count'])
            if value:
                ax.barh(y, value, left=start, height=.56,
                        color=GROUP_COLORS[group], edgecolor='white',
                        linewidth=.6, zorder=3)
                ax.text(start+value/2, y, str(value), ha='center', va='center',
                        fontsize=13.4, color=INK if group == 'G2' else 'white',
                        weight='bold', zorder=5)
            start += value
        # A narrow black stock threshold stays distinct from group colours.
        ax.plot([inv, inv], [y-.39, y+.39], c=INK, lw=2.0,
                solid_capstyle='round', zorder=6)
        if short:
            ax.add_patch(Rectangle((inv, y-.29), short, .58,
                                   fill=False, edgecolor=DEFICIT,
                                   linewidth=2.15, zorder=7))
            ax.text(total+.17, y, f'+{short}', fontsize=13.5,
                    weight='bold', color=DEFICIT, va='center', zorder=8)
    return {'resource_total': int(res.sum_required.sum()),
            'shortage': int(res.shortage.sum()),
            'stock': {key: int(res.loc[key, 'inventory']) for key in KEYS}}


fig = plt.figure(figsize=(20.9, 14.4))
fig.legend(handles=[Line2D([0], [0], color=GROUP_COLORS[g], lw=3.3,
                           label=f'第{g[1]}组') for g in GROUP_COLORS]
                   + [Line2D([0], [0], color=INK, lw=2.8, label='中继航迹')],
           loc='upper center', bbox_to_anchor=(.5, 1.004), ncol=4,
           frameon=False, fontsize=17, columnspacing=1.8, handlelength=2)

audit = {}
for x, sid, letter, name in ((.008, SID[0], 'a', '两组 · 8区 / 7区'),
                             (.505, SID[1], 'b', '三组 · 4区 / 6区 / 5区')):
    ax = fig.add_axes([x, .480, .487, .475], projection='3d')
    audit[sid] = terrain(ax, sid, letter, name)

bar = fig.colorbar(ScalarMappable(norm=scene.norm, cmap=CM),
                   cax=fig.add_axes([.358, .026, .260, .012]), orientation='horizontal')
bar.set_ticks((100, 400, 700, 1000))
bar.ax.tick_params(labelsize=14, pad=2, length=3)
bar.outline.set_linewidth(.7)
fig.text(.63, .032, '地形海拔（米）', fontsize=16, va='center', color=INK)

for x, sid, letter, heading in (
    (.085, SID[0], 'c', '两组 · 配置35 · 缺5'),
    (.580, SID[1], 'd', '三组 · 配置39 · 缺9')):
    ax = fig.add_axes([x, .142, .355, .272])
    audit[sid].update(resources(ax, sid))
    ax.set_title(f'{letter}  {heading}', loc='left', pad=9,
                 fontsize=20, color=INK)

fig.legend(handles=[Line2D([0], [0], marker='|', markersize=14,
                     color=INK, ls='none', markeredgewidth=2,
                     label='现有库存'),
              Patch(facecolor='none', edgecolor=DEFICIT,
                    linewidth=2, label='库存缺口')],
           loc='lower center', bbox_to_anchor=(.5, .080), ncol=2,
           frameon=False, fontsize=15, columnspacing=3, handlelength=1.6)

assert audit[SID[0]]['resource_total'] == 35 and audit[SID[0]]['shortage'] == 5
assert audit[SID[1]]['resource_total'] == 39 and audit[SID[1]]['shortage'] == 9
assert all(audit[sid]['transport_flights'] == 22 for sid in SID)
for sid in SID:
    cert_row = next(x for x in cert['extended'] if x['scenario_id'] == sid)
    assert audit[sid]['resource_total'] == sum(cert_row['required'].values())
    assert audit[sid]['shortage'] == cert_row['shortage']

stem = '问题四_图03_原版三维地形与八类资源缺口'
tmp_png = a.output_dir/(stem+'.tmp.png')
tmp_pdf = a.output_dir/(stem+'.tmp.pdf')
png, pdf = a.output_dir/(stem+'.png'), a.output_dir/(stem+'.pdf')
fig.savefig(tmp_png, dpi=a.dpi, bbox_inches='tight', pad_inches=.12)
fig.savefig(tmp_pdf, dpi=170, bbox_inches='tight', pad_inches=.12)
plt.close(fig)
with Image.open(tmp_png) as im:
    im.verify()
with Image.open(tmp_png) as im:
    im.load()
    pixels = list(im.size)
assert tmp_pdf.stat().st_size > 5000
tmp_png.replace(png)
tmp_pdf.replace(pdf)
qa = {'status': 'PASS', 'source': 'fixed_revised_Q3_original_DEM',
      'figure_type': 'two_original_3d_terrain_maps_and_two_stacked_resource_bar_charts',
      'scenario_audit': audit, 'size_px': pixels,
      'png': png.name, 'pdf': pdf.name,
      'sha256_png': hashlib.sha256(png.read_bytes()).hexdigest()}
(a.output_dir/'问题四_最终答案图核验.json').write_text(
    json.dumps(qa, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(qa, ensure_ascii=False))
