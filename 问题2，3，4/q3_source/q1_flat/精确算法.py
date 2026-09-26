"""问题一精确算法内核（只含计算，不生成图件）。
由原始已核验求解器、三目标精确前沿程序及收敛核验程序合并。
主入口：python 问题一_完整计算.py
所有物理参数从同目录三张原始 Excel 读取；算法保持精确动态规划。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


CATEGORIES = ("医疗物资", "饮用水", "应急食品", "生活卫生用品")


def utm49(lon: float, lat: float) -> tuple[float, float]:
    """WGS84 横轴墨卡托 49 带；题目工作区经度约 109°E。"""
    a, e2, k0 = 6378137.0, 0.0066943799901413165, 0.9996
    phi, dl = math.radians(lat), math.radians(lon - 111.0)
    ep2 = e2 / (1 - e2)
    s, c, t = math.sin(phi), math.cos(phi), math.tan(phi) ** 2
    C, A = ep2 * c * c, c * dl
    N = a / math.sqrt(1 - e2 * s * s)
    M = a * ((1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * phi)
             + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * phi)
             - (35 * e2**3 / 3072) * math.sin(6 * phi))
    x = 500000 + k0 * N * (A + (1 - t + C) * A**3 / 6
                           + (5 - 18 * t + t * t + 72 * C - 58 * ep2) * A**5 / 120)
    y = k0 * (M + N * math.tan(phi) * (A * A / 2
                                      + (5 - t + 9 * C + 4 * C * C) * A**4 / 24
                                      + (61 - 58 * t + t * t + 600 * C - 330 * ep2) * A**6 / 720))
    return x, y


def touched_cells(x0: float, y0: float, x1: float, y1: float,
                  width: int, height: int) -> set[tuple[int, int]]:
    """直线切过或恰好触及的所有 DEM 像元，网格线两侧都计入。"""
    dx, dy = x1 - x0, y1 - y0
    times = [0.0, 1.0]
    for start, delta in ((x0, dx), (y0, dy)):
        if abs(delta) > 1e-14:
            for edge in range(math.floor(min(start, start + delta)),
                              math.ceil(max(start, start + delta)) + 1):
                t = (edge - start) / delta
                if 0 < t < 1:
                    times.append(t)
    times.sort()
    unique = [times[0]]
    for t in times[1:]:
        if t - unique[-1] > 1e-12:
            unique.append(t)
    result = set()
    for t in unique + [(a + b) / 2 for a, b in zip(unique, unique[1:])]:
        x, y = x0 + t * dx, y0 + t * dy
        for col in {math.floor(x - 1e-9), math.floor(x + 1e-9)}:
            for row in {math.floor(y - 1e-9), math.floor(y + 1e-9)}:
                if 0 <= col < width and 0 <= row < height:
                    result.add((row, col))
    return result


def read_dem(path: Path) -> tuple[np.ndarray, float, float, float, float]:
    with Image.open(path) as image:
        dem = np.asarray(image, dtype=np.float32).copy()
        scale = image.tag_v2[33550]
        point = image.tag_v2[33922]
        keys = tuple(image.tag_v2[34735])
        raster_type = None
        for i in range(4, len(keys), 4):
            if keys[i] == 1025:
                raster_type = keys[i + 3]
        dx, dy = float(scale[0]), float(scale[1])
        assert dem.ndim == 2 and np.isfinite(dem).all() and raster_type in (1, 2)
        # GeoTIFF PixelIsArea: tiepoint 在左上角；PixelIsPoint: 在首像元中心。
        left = float(point[3]) - (dx / 2 if raster_type == 2 else 0)
        top = float(point[4]) + (dy / 2 if raster_type == 2 else 0)
        return dem, left, top, dx, dy


@dataclass(frozen=True)
class Pattern:
    model: str
    counts: tuple[int, int, int, int]
    mass: int
    volume_l: int
    energy: float
    duration: float
    battery: float


def leg_data(nodes: pd.DataFrame, dem_path: Path) -> pd.DataFrame:
    dem, left, top, dx, dy = read_dem(dem_path)
    origin = nodes.loc["O01"]
    o_utm = utm49(origin.longitude_deg, origin.latitude_deg)
    x0 = (origin.longitude_deg - left) / dx
    y0 = (top - origin.latitude_deg) / dy
    records = []
    for site in nodes.index.drop("O01"):
        dest = nodes.loc[site]
        u = utm49(dest.longitude_deg, dest.latitude_deg)
        d = math.dist(o_utm, u)
        x1, y1 = (dest.longitude_deg - left) / dx, (top - dest.latitude_deg) / dy
        cells = touched_cells(x0, y0, x1, y1, dem.shape[1], dem.shape[0])
        if not cells:
            raise ValueError(f"航线超出 DEM：{site}")
        peak = max(float(dem[r, c]) for r, c in cells)
        H = peak + 50
        z0, z1 = float(origin.operating_altitude_m), float(dest.operating_altitude_m)
        records.append(dict(服务区=site, 水平单程米=d, 沿线最高地面米=peak,
                            巡航海拔米=H, 去程爬升米=max(0, H-z0),
                            去程下降米=max(0, H-z1), 返程爬升米=max(0, H-z1),
                            返程下降米=max(0, H-z0), 沿线涉及像元数=len(cells)))
    return pd.DataFrame(records).set_index("服务区")


def flight(site: pd.Series, model: pd.Series, load: float) -> tuple[float, float]:
    """附录航程 + 两项能耗闭合；返程载货量为零。"""
    qmax = float(model.max_payload_kg)
    assert -1e-9 <= load <= qmax + 1e-9
    rng = (model.empty_range_m - (model.empty_range_m-model.full_range_m)
           * (load/qmax)**1.5)
    B, d = float(model.battery_kwh), float(site.水平单程米)
    horizontal = B*d*(1/rng+1/model.empty_range_m)
    ascent = (9.80665 / (3.6e6 * model.climb_efficiency)
              * ((model.empty_mass_kg+load)*site.去程爬升米
                 + model.empty_mass_kg*site.返程爬升米))
    seconds = (2*d/model.cruise_speed_m_s
               + (site.去程爬升米+site.返程爬升米)/model.climb_speed_m_s
               + (site.去程下降米+site.返程下降米)/model.descent_speed_m_s)
    return float(horizontal+ascent), float(seconds)


def safe_mass(site: pd.Series, model: pd.Series, reserve: float) -> float:
    upper = float(model.max_payload_kg)
    budget = float(model.battery_kwh)*(1-reserve)
    if flight(site, model, 0)[0] > budget+1e-10:
        return 0.0
    if flight(site, model, upper)[0] <= budget+1e-10:
        return upper
    lo, hi = 0.0, upper
    for _ in range(70):
        mid = (lo+hi)/2
        if flight(site, model, mid)[0] <= budget:
            lo = mid
        else:
            hi = mid
    return lo


def site_patterns(site: pd.Series, local: pd.DataFrame, models: pd.DataFrame,
                  reserve: float, unit: tuple[tuple[int, int], ...]):
    demand = tuple(int((local.material_type == c).sum()) for c in CATEGORIES)
    patterns = []
    safes = {}
    for name, model in models.iterrows():
        safe = safe_mass(site, model, reserve)
        safes[name] = safe
        for c in itertools.product(*(range(v + 1) for v in demand)):
            if not sum(c):
                continue
            mass = sum(c[i]*unit[i][0] for i in range(4))
            vol = sum(c[i]*unit[i][1] for i in range(4))
            if mass > min(safe, float(model.max_payload_kg))+1e-9:
                continue
            if vol > model.max_volume_m3*1000+1e-8:
                continue
            energy, flying = flight(site, model, mass)
            if energy > float(model.battery_kwh)*(1-reserve)+1e-9:
                continue
            n = sum(c)
            seconds = (flying + model.fixed_prep_s + model.handoff_base_s
                       + n*(model.load_per_box_s+model.handoff_per_box_s))
            patterns.append(Pattern(name, c, mass, vol, energy,
                                    float(seconds), float(model.battery_kwh)))
    return demand, patterns, safes


def solve_site(demand: tuple[int, ...], patterns: list[Pattern],
               priority: str) -> dict[int, tuple[float, float, tuple[Pattern, ...]]]:
    """完全枚举需求状态和候选模式；对每个精确架次数保存最优能耗或时间。"""
    def key(en: float, seconds: float):
        return (en, seconds) if priority == "energy" else (seconds, en)

    @lru_cache(maxsize=None)
    def recurse(rem: tuple[int, ...]):
        if not any(rem):
            return {0: (0.0, 0.0, ())}
        best = {}
        for p in patterns:
            if any(c > r for c, r in zip(p.counts, rem)):
                continue
            next_rem = tuple(r-c for r, c in zip(rem, p.counts))
            for k, (en, sec, plan) in recurse(next_rem).items():
                trial = (en+p.energy, sec+p.duration, (p,)+plan)
                old = best.get(k+1)
                if old is None or key(trial[0], trial[1]) < key(old[0], old[1]):
                    best[k+1] = trial
        return best

    return recurse(demand)


def combine(all_frontiers: dict[str, dict], priority: str):
    def key(en, t):
        return (en, t) if priority == "energy" else (t, en)
    current = {0: (0., 0., {})}
    for site, options in all_frontiers.items():
        after = {}
        for n0, (en0, t0, chosen) in current.items():
            for n1, (en1, t1, plan) in options.items():
                n = n0+n1
                trial = (en0+en1, t0+t1, {**chosen, site: plan})
                old = after.get(n)
                if old is None or key(*trial[:2]) < key(*old[:2]):
                    after[n] = trial
        current = after
    return current




@dataclass(frozen=True)
class Label:
    sorties: int
    energy: float
    duration: float
    plan: tuple[object, ...]


def _same(a: float, b: float, tol: float = 1e-10) -> bool:
    if not (math.isfinite(a) and math.isfinite(b)):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def pareto_prune(labels: list[Label]) -> tuple[Label, ...]:
    """Return exact nondominated labels for three minimization objectives.

    Energy and duration are calculated deterministically from a small discrete
    pattern set.  Tolerances only collapse round-off duplicates; they do not
    aggregate or approximate the frontier.
    """
    if not labels:
        return ()

    # First collapse objective-identical labels caused only by permutation of
    # the same sortie patterns.  Rounding is far below the reported precision.
    unique: dict[tuple[int, float, float], Label] = {}
    for lab in labels:
        key = (lab.sorties, round(lab.energy, 12), round(lab.duration, 9))
        unique.setdefault(key, lab)

    by_n: dict[int, list[Label]] = {}
    for lab in unique.values():
        by_n.setdefault(lab.sorties, []).append(lab)

    accepted: list[Label] = []
    for n in sorted(by_n):
        # Exact 2-D frontier at fixed n: after sorting by energy, retain only
        # strictly improving durations.
        fixed: list[Label] = []
        best_t = math.inf
        for lab in sorted(by_n[n], key=lambda x: (x.energy, x.duration)):
            if lab.duration < best_t and not _same(lab.duration, best_t):
                fixed.append(lab)
                best_t = lab.duration

        # Any accepted label has fewer sorties.  Such a label dominates the
        # candidate iff it is no worse in both continuous objectives.
        for lab in fixed:
            dominated = any(
                old.energy <= lab.energy + 1e-10
                and old.duration <= lab.duration + 1e-8
                for old in accepted
            )
            if not dominated:
                accepted.append(lab)

    return tuple(sorted(accepted, key=lambda x: (x.sorties, x.energy, x.duration)))


def prune_fixed_n(labels: list[Label]) -> tuple[Label, ...]:
    """2-D energy--duration frontier when all labels have the same sortie count."""
    if not labels:
        return ()
    unique: dict[tuple[float, float], Label] = {}
    for lab in labels:
        unique.setdefault((round(lab.energy, 12), round(lab.duration, 9)), lab)
    accepted: list[Label] = []
    best_t = math.inf
    for lab in sorted(unique.values(), key=lambda x: (x.energy, x.duration)):
        if lab.duration < best_t and not _same(lab.duration, best_t):
            accepted.append(lab)
            best_t = lab.duration
    return tuple(accepted)


def solve_site_pareto(
    demand: tuple[int, ...], patterns: list[Pattern]
) -> tuple[Label, ...]:
    """Exact state-space DP retaining every nondominated local label."""

    # Dominated single-sortie patterns having the same delivered counts can be
    # removed safely before the DP.  This is exact because their state change is
    # identical and both energy and duration are additive.
    grouped: dict[tuple[int, ...], list[Pattern]] = {}
    for pattern in patterns:
        grouped.setdefault(pattern.counts, []).append(pattern)
    efficient_patterns: list[Pattern] = []
    for group in grouped.values():
        candidates = []
        for p in group:
            if not any(
                q.energy <= p.energy + 1e-10
                and q.duration <= p.duration + 1e-8
                and (q.energy < p.energy - 1e-10 or q.duration < p.duration - 1e-8)
                for q in group
            ):
                candidates.append(p)
        efficient_patterns.extend(candidates)

    @lru_cache(maxsize=None)
    def recurse(rem: tuple[int, ...]) -> tuple[Label, ...]:
        if not any(rem):
            return (Label(0, 0.0, 0.0, ()),)
        candidates: list[Label] = []
        for p in efficient_patterns:
            if any(c > r for c, r in zip(p.counts, rem)):
                continue
            next_rem = tuple(r - c for r, c in zip(rem, p.counts))
            for child in recurse(next_rem):
                candidates.append(
                    Label(
                        child.sorties + 1,
                        child.energy + p.energy,
                        child.duration + p.duration,
                        (p,) + child.plan,
                    )
                )
        return pareto_prune(candidates)

    return recurse(demand)


def solve_site_conditional(
    demand: tuple[int, ...], patterns: list[Pattern]
) -> dict[int, tuple[Label, ...]]:
    """Exact energy--time frontiers conditional on every exact sortie count."""
    @lru_cache(maxsize=None)
    def recurse(rem: tuple[int, ...]) -> tuple[tuple[int, tuple[Label, ...]], ...]:
        if not any(rem):
            return ((0, (Label(0, 0.0, 0.0, ()),)),)
        candidates: dict[int, list[Label]] = {}
        for p in patterns:
            if any(c > r for c, r in zip(p.counts, rem)):
                continue
            next_rem = tuple(r - c for r, c in zip(rem, p.counts))
            for k, labels in recurse(next_rem):
                for child in labels:
                    lab = Label(
                        k + 1,
                        child.energy + p.energy,
                        child.duration + p.duration,
                        (p,) + child.plan,
                    )
                    candidates.setdefault(k + 1, []).append(lab)
        result = {k: prune_fixed_n(v) for k, v in candidates.items()}
        return tuple(sorted(result.items()))

    return dict(recurse(demand))


def combine_conditional(
    site_frontiers: dict[str, dict[int, tuple[Label, ...]]]
) -> dict[int, tuple[Label, ...]]:
    """Minkowski sum retaining the 2-D conditional frontier for every N."""
    current: dict[int, tuple[Label, ...]] = {0: (Label(0, 0.0, 0.0, ()),)}
    for site, local in site_frontiers.items():
        candidates: dict[int, list[Label]] = {}
        for n0, lefts in current.items():
            for n1, rights in local.items():
                n = n0 + n1
                for left in lefts:
                    for right in rights:
                        candidates.setdefault(n, []).append(
                            Label(
                                n,
                                left.energy + right.energy,
                                left.duration + right.duration,
                                left.plan + tuple((site, p) for p in right.plan),
                            )
                        )
        current = {n: prune_fixed_n(v) for n, v in candidates.items()}
        print(
            f"{site}: conditional N={len(current):3d}, labels={sum(map(len, current.values())):6d}",
            flush=True,
        )
    return current


def combine_frontiers(
    site_frontiers: dict[str, tuple[Label, ...]]
) -> tuple[Label, ...]:
    """Exact Minkowski sum of local Pareto sets with dominance pruning."""
    current = (Label(0, 0.0, 0.0, ()),)
    for site, local in site_frontiers.items():
        candidates = [
            Label(
                left.sorties + right.sorties,
                left.energy + right.energy,
                left.duration + right.duration,
                left.plan + tuple((site, p) for p in right.plan),
            )
            for left in current
            for right in local
        ]
        current = pareto_prune(candidates)
        print(f"{site}: local={len(local):4d}, combined={len(current):5d}", flush=True)
    return current


def instrument_dp(demand, patterns):
    """复制正式求解器的递归 DP，只额外记录缓存状态与标签数。"""
    @lru_cache(maxsize=None)
    def rec(rem):
        if not any(rem):
            return {0: (0.0, 0.0)}
        best = {}
        for p in patterns:
            if any(c > r for c, r in zip(p.counts, rem)):
                continue
            nxt = tuple(r - c for r, c in zip(rem, p.counts))
            for k, (en, sec) in rec(nxt).items():
                trial = (en + p.energy, sec + p.duration)
                old = best.get(k + 1)
                if old is None or trial < old:
                    best[k + 1] = trial
        return best

    frontier = rec(tuple(demand))
    # 在缓存内的每个需求状态上，对每个精确架次仅保留一个最优标签。
    # Python 不暴露 cache 字典，因此标签总数用同一个有向无环状态网再计数。
    all_states = sorted(itertools.product(*(range(v + 1) for v in demand)), key=lambda x: (sum(x), x))
    labels = {tuple(0 for _ in demand): {0: (0.0, 0.0)}}
    for delivered in all_states:
        if delivered not in labels:
            continue
        for p in patterns:
            nxt = tuple(delivered[i] + p.counts[i] for i in range(len(demand)))
            if any(nxt[i] > demand[i] for i in range(len(demand))):
                continue
            dst = labels.setdefault(nxt, {})
            for k, (en, sec) in labels[delivered].items():
                trial = (en + p.energy, sec + p.duration)
                old = dst.get(k + 1)
                if old is None or trial < old:
                    dst[k + 1] = trial
    return frontier, rec.cache_info().currsize, sum(len(v) for v in labels.values())


def canonical_partitions(demand, patterns, sorties):
    """返回无顺序重复的装载划分；架次在问题一中没有发车先后语义。"""
    rows = []
    for idxs in itertools.combinations_with_replacement(range(len(patterns)), sorties):
        selected = [patterns[i] for i in idxs]
        counts = tuple(sum(p.counts[j] for p in selected) for j in range(len(demand)))
        if counts != tuple(demand):
            continue
        rows.append({
            "energy": sum(p.energy for p in selected),
            "time": sum(p.duration for p in selected),
            "indices": idxs,
            "models": "+".join(p.model for p in selected),
        })
    return rows


def dominates(a, b, tol=1e-10):
    return (a[0] <= b[0] + tol and a[1] <= b[1] + tol
            and (a[0] < b[0] - tol or a[1] < b[1] - tol))


def pareto_prune_2d(rows):
    kept = []
    for r in rows:
        point = (r["energy"], r["time"])
        if any(dominates((k["energy"], k["time"]), point) for k in rows if k is not r):
            continue
        # 同目标值仅保留一个代表性装载方案。
        if not any(abs(k["energy"] - r["energy"]) < 1e-10 and abs(k["time"] - r["time"]) < 1e-7
                   for k in kept):
            kept.append(r)
    return kept


def prune_objectives(points):
    unique = {}
    for n, e, t in points:
        unique[(n, round(e, 10), round(t, 7))] = (int(n), float(e), float(t))
    values = list(unique.values())
    out = []
    for x in values:
        if any(y[0] <= x[0] and y[1] <= x[1] + 1e-10 and y[2] <= x[2] + 1e-7
               and (y[0] < x[0] or y[1] < x[1] - 1e-10 or y[2] < x[2] - 1e-7)
               for y in values if y is not x):
            continue
        out.append(x)
    return sorted(out)


def local_multiobjective_front(demand, patterns):
    states = {tuple(0 for _ in demand): [(0, 0.0, 0.0)]}
    all_states = sorted(itertools.product(*(range(v + 1) for v in demand)), key=lambda x: (sum(x), x))
    for delivered in all_states:
        labs = states.get(delivered)
        if not labs:
            continue
        for p in patterns:
            nxt = tuple(delivered[i] + p.counts[i] for i in range(len(demand)))
            if any(nxt[i] > demand[i] for i in range(len(demand))):
                continue
            candidates = states.get(nxt, []) + [
                (n + 1, e + p.energy, t + p.duration) for n, e, t in labs
            ]
            states[nxt] = prune_objectives(candidates)
    return prune_objectives(states[tuple(demand)])




def cell_interval(x0, y0, x1, y1, row, col):
    """计算线段与闭像元的参数交集；保留仅触角/触边的像元。"""
    lo, hi = 0., 1.
    for start, delta, lower, upper in (
        (x0, x1-x0, col, col+1), (y0, y1-y0, row, row+1),
    ):
        if abs(delta) < 1e-14:
            if not lower-1e-9 <= start <= upper+1e-9:
                return None
        else:
            a, b = sorted(((lower-start)/delta, (upper-start)/delta))
            lo, hi = max(lo, a), min(hi, b)
    if hi < lo-1e-10:
        return None
    return float(np.clip(lo, 0, 1)), float(np.clip(max(lo, hi), 0, 1))



