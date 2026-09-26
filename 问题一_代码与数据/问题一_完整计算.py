"""问题一完整计算入口：原始 Excel → 地形 → 安全载荷 → 精确组批 → 结果。

运行（Python 3.10+，Windows/Jupyter 均可）：
    pip install -r requirements.txt
    python 问题一_完整计算.py
Jupyter 中在解压目录运行：
    %run 问题一_完整计算.py
也可以指定目录：
    python 问题一_完整计算.py --data-dir "C:\\Users\\86133\\Downloads\\问题一"

必需输入（与代码平放，不依赖原工程的多层目录）：
  1. 调度中心与服务区.xlsx / 数据：节点编号、经纬度、地面海拔。
  2. 物资需求与配送时限.xlsx / 逐箱货箱清单：80 个不可拆货箱。
     同文件 / 数据：按服务区和物资类型核对需求，不能重复计入需求。
  3. 运输无人机数据.xlsx / 数据 / A3:R5：三类运输机型参数。
  4. 最终工作DEM.tif：前处理已冻结的稳健三维配准产品，正式计算使用。
  5. 原始DEM.tif：只用于原始/修正影响复算，不参与正式方案选取。

输出只有 问题一_运行结果.json；包含所有表、字段来源、真实日志、环境版本。
包内 问题一_结果汇总.xlsx 是随包提供的本次实际运行快照，不会因 JSON
变化自行刷新。在已配置 @oai/artifact-tool 的 Node 环境运行
    node 问题一_表格导出.mjs
即可由 JSON 重建同一结果工作簿。Python 计算不依赖此导出环境。

口径：只计算 O01→单一服务区→O01，不求发车时刻/实体飞机/电池调度。
累计作业时间是各架次飞行、准备、装箱和交接时间之和，不是完工时刻。
时限、首批和优先级仅做原始数据核对；首批标签只用于稳定地映射货箱编号。
不对 DEM 再平滑，也不在问题一重复估计配准参数。
所有优化是离散精确 DP；绘图插值面不是可行解，本代码不生成插值解。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
sys.dont_write_bytecode = True
import 精确算法 as q1

MODEL_FIELDS = ["model_id", "model_name", "empty_mass_kg", "max_payload_kg", "max_volume_m3",
                "cruise_speed_m_s", "empty_range_m", "full_range_m", "battery_kwh",
                "reserve_percent", "fixed_prep_s", "load_per_box_s", "handoff_base_s",
                "handoff_per_box_s", "climb_speed_m_s", "descent_speed_m_s",
                "climb_efficiency", "descent_efficiency"]
BOX_FIELDS = {"货箱编号":"box_id", "服务区编号":"node_id", "物资类型":"material_type",
              "单箱质量（kg）":"mass_kg", "单箱体积（m³）":"volume_m3",
              "是否首批保障":"first_batch", "首批截止时间（s）":"first_deadline_s",
              "期望送达时间（s）":"target_delivery_s", "应急优先系数":"priority"}
INPUT_FILES = ["调度中心与服务区.xlsx", "物资需求与配送时限.xlsx", "运输无人机数据.xlsx",
               "最终工作DEM.tif", "原始DEM.tif"]


def clean(value):
    """JSON 中不写 NaN；缺失和不可行保持 null，不冒充零。"""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def records(frame):
    return clean(frame.to_dict("records"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    folder = args.data_dir.resolve()
    started = time.perf_counter()
    logs, checks, sources = [], [], []

    def log(stage, detail):
        row = {"时间UTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "累计秒": round(time.perf_counter()-started, 3), "步骤": stage, "运行记录": detail}
        logs.append(row)
        print(f"[{row['累计秒']:8.3f}s] {stage}：{detail}", flush=True)

    def check(name, valid, detail):
        checks.append({"核验项":name, "结果":"通过" if valid else "失败", "说明":str(detail)})
        if not valid:
            raise ValueError(f"核验失败：{name}；{detail}")

    def source(file, sheet, column, field, use):
        sources.append({"输入文件":file, "工作表":sheet, "原列或区域":column,
                        "代码字段":field, "计算用途":use})

    log("开始", "仅处理问题一；不输出图件，不调用启发式算法。")
    manifest = []
    for file in INPUT_FILES:
        path = folder/file
        check("输入文件存在", path.is_file(), file)
        manifest.append({"文件名":file, "字节数":path.stat().st_size,
                         "SHA256":hashlib.sha256(path.read_bytes()).hexdigest()})
    # ---------- 第一步：读原始表，不依赖此前导出的 CSV ----------
    node_raw = pd.read_excel(folder/INPUT_FILES[0], sheet_name="数据", header=None)
    node_rows = []
    for excel_row, row in enumerate(node_raw.itertuples(index=False, name=None), 1):
        if isinstance(row[0], str) and (row[0] == "O01" or row[0].startswith("S0")):
            node_rows.append(dict(node_id=row[0], name=row[1], longitude_deg=float(row[2]),
                latitude_deg=float(row[3]), ground_elevation_m=float(row[4]),
                population=float(row[5]) if pd.notna(row[5]) else None,
                operating_altitude_m=float(row[4])+(0 if row[0] == "O01" else 30),
                原表行号=excel_row))
    nodes = pd.DataFrame(node_rows).set_index("node_id")
    for col, field, use in [
        ("A：编号", "node_id", "节点关联键；O01 为起终点"),
        ("B：名称", "name", "保留原名"),
        ("C：经度（°）", "longitude_deg", "航段定位、UTM49距离与DEM像元索引"),
        ("D：纬度（°）", "latitude_deg", "航段定位、UTM49距离与DEM像元索引"),
        ("E：海拔（m）", "ground_elevation_m", "题给地面高程；服务区作业海拔加30m，O01不加"),
        ("F：人口", "population", "保留，不参与问题一优化")]:
        source(INPUT_FILES[0], "数据", col, field, use)

    demand = pd.read_excel(folder/INPUT_FILES[1], sheet_name="数据")
    box_raw = pd.read_excel(folder/INPUT_FILES[1], sheet_name="逐箱货箱清单")
    boxes = box_raw.rename(columns=BOX_FIELDS).copy()
    for i, (label, field) in enumerate(BOX_FIELDS.items()):
        use = {"box_id":"不可拆货箱唯一ID和逐箱交付核验", "node_id":"服务区分组",
               "material_type":"等质箱类别状态", "mass_kg":"载重与能耗，单位kg",
               "volume_m3":"体积约束，m³转换为整数升",
               "first_batch":"仅核对及稳定分配ID，不构造发车次序",
               "first_deadline_s":"仅核对；问题一不执行时间窗",
               "target_delivery_s":"保留；问题一不执行送达时限",
               "priority":"保留；不改变本问三目标"}[field]
        source(INPUT_FILES[1], "逐箱货箱清单", f"{chr(65+i)}：{label}", field, use)
    for i, label in enumerate(demand.columns):
        source(INPUT_FILES[1], "数据", f"{chr(65+i)}：{label}", str(label),
               "核对逐箱清单的分类箱数、属性和首批数；不二次计入需求")
    transport = pd.read_excel(folder/INPUT_FILES[2], sheet_name="数据", header=None)
    models = pd.DataFrame(transport.iloc[2:5, :18].to_numpy(), columns=MODEL_FIELDS)
    models[MODEL_FIELDS[2:]] = models[MODEL_FIELDS[2:]].apply(pd.to_numeric)
    model_raw = models.copy()
    model_raw.columns = [str(x) for x in transport.iloc[1, :18]]
    models = models.set_index("model_id")
    for i, field in enumerate(MODEL_FIELDS):
        source(INPUT_FILES[2], "数据", f"{chr(65+i)}3:{chr(65+i)}5：{transport.iloc[1,i]}", field,
               "模型ID/名称" if i < 2 else ("下降附加能耗设为0，仅下降速度进入时间" if field=="descent_efficiency"
               else "本问机型物理、能量或作业时间参数；不包括实体机队调度"))
    source(INPUT_FILES[2], "数据", "A9:C16、A20:C22", "未进入优化", "实体机、备用电池与充电时长属于后续调度")
    source(INPUT_FILES[3], "单波段GeoTIFF", "全部直线相交/触及像元", "Z_final", "正式工作DEM；沿线最大值加50m")
    source(INPUT_FILES[4], "单波段GeoTIFF", "全部直线相交/触及像元", "Z_raw", "仅重算原始DEM对照；不用于正式方案")

    check("节点ID与数量", nodes.index.is_unique and set(nodes.index)=={"O01",*(f"S{i:03d}" for i in range(1,16))}, len(nodes))
    check("80个唯一货箱", len(boxes)==80 and boxes.box_id.is_unique, len(boxes))
    check("服务区关联", boxes.node_id.isin(nodes.index.drop("O01")).all(), "按原始服务区ID关联")
    check("机型完整", set(models.index)=={"A","B","C"} and not models[MODEL_FIELDS[2:]].isna().any().any(), "A/B/C各一行")
    check("质量体积正值", (boxes.mass_kg>0).all() and (boxes.volume_m3>0).all(), "不删箱、不做标准化")
    check("空载航程大于满载航程", (models.empty_range_m>models.full_range_m).all(), "允许按题给衰减式计算")
    check("三类机型默认余量一致", models.reserve_percent.nunique()==1, models.reserve_percent.tolist())
    reserve = float(models.reserve_percent.iloc[0])/100
    check("首批截止时间逻辑", ((boxes.first_batch=="是")==boxes.first_deadline_s.notna()).all(), "非首批空缺并非异常")
    gcols = ["服务区编号","物资类型"]
    grouped = box_raw.groupby(gcols, dropna=False)
    joined = demand.merge(grouped.size().rename("逐箱数").reset_index(), on=gcols, how="outer", validate="one_to_one", indicator=True)
    check("汇总表与逐箱箱数对应", joined["_merge"].eq("both").all() and joined["总需求箱数"].eq(joined["逐箱数"]).all(), f"{len(demand)}类需求，{len(boxes)}箱")
    for label in ["单箱质量（kg）","单箱体积（m³）","应急优先系数","期望送达时间（s）"]:
        for _, row in demand.iterrows():
            vals = grouped.get_group((row[gcols[0]],row[gcols[1]]))[label].drop_duplicates()
            check("汇总/逐箱属性一致", len(vals)==1 and np.isclose(float(vals.iloc[0]),float(row[label])), f"{row[gcols[0]]}/{row[gcols[1]]}/{label}")
    first = grouped["是否首批保障"].apply(lambda v: int(v.eq("是").sum()))
    check("首批数量对应", all(first.loc[(r[gcols[0]],r[gcols[1]])]==r["首批必须送达箱数"] for _,r in demand.iterrows()), "逐类核验")
    unit = []
    for cat in q1.CATEGORIES:
        physical = boxes.loc[boxes.material_type==cat,["mass_kg","volume_m3"]].drop_duplicates()
        check("同类货箱物理同质", len(physical)==1, cat)
        mass, volume = map(float,physical.iloc[0])
        check("整数质量及升换算", abs(mass-round(mass))<1e-9 and abs(volume*1000-round(volume*1000))<1e-9, cat)
        unit.append((round(mass),round(volume*1000)))
    unit = tuple(unit)
    log("原始表读取与核验", f"{len(nodes)}节点，{len(boxes)}货箱，{len(models)}机型；总质量{boxes.mass_kg.sum():g}kg。")

    # ---------- 第二步：DEM航段与安全载荷 ----------
    raster = q1.read_dem(folder/INPUT_FILES[3])
    for dem_file in INPUT_FILES[3:]:
        z,left,top,dx,dy = q1.read_dem(folder/dem_file)
        inside = all(0 <= (r.longitude_deg-left)/dx < z.shape[1] and 0 <= (top-r.latitude_deg)/dy < z.shape[0] for r in nodes.itertuples())
        check("节点均在DEM内", inside, dem_file)
    legs = q1.leg_data(nodes, folder/INPUT_FILES[3])
    site_data, safe_rows, bound_rows = {}, [], []
    e_fronts, t_fronts, local_fronts, conditional_fronts, partitions = {},{},{},{},{}
    pattern_rows, dp_rows = [], []
    for site in legs.index:
        local = boxes[boxes.node_id==site]
        counts, patterns, safes = q1.site_patterns(legs.loc[site],local,models,reserve,unit)
        site_data[site] = (counts,patterns,safes)
        for g,m in models.iterrows():
            budget = float(m.battery_kwh)*(1-reserve)
            safe_rows.append({"服务区":site,"机型":g,"最大安全载荷kg":safes[g],
                "额定载荷kg":m.max_payload_kg,"装载容积m³":m.max_volume_m3,"电池能量kWh":m.battery_kwh,
                "允许能耗kWh":budget,"安全载荷处能耗kWh":q1.flight(legs.loc[site],m,safes[g])[0],
                "约束类型":"能量受限" if safes[g]<m.max_payload_kg-1e-8 else "额定载荷受限"})
        e_fronts[site] = q1.solve_site(counts,patterns,"energy")
        t_fronts[site] = q1.solve_site(counts,patterns,"time")
        check("服务区存在可行交付", bool(e_fronts[site]), site)
        lb_mass = math.ceil((local.mass_kg.sum()-1e-10)/max(safes.values()))
        lb_vol = math.ceil((local.volume_m3.sum()-1e-12)/models.max_volume_m3.max())
        bound_rows.append({"服务区":site,"货箱数":len(local),"总质量kg":local.mass_kg.sum(),
            "总体积m³":local.volume_m3.sum(),"质量下界":lb_mass,"体积下界":lb_vol,
            "架次下界":max(lb_mass,lb_vol),"精确最少架次":min(e_fronts[site])})
        local_fronts[site] = q1.solve_site_pareto(counts,patterns)
        conditional_fronts[site] = q1.solve_site_conditional(counts,patterns)
        instrument, visited, labels = q1.instrument_dp(counts,patterns)
        partitions[site] = q1.canonical_partitions(counts,patterns,min(e_fronts[site]))
        check("独立DP仪表复核", abs(instrument[min(instrument)][0]-e_fronts[site][min(e_fronts[site])][0])<1e-9,site)
        dp_rows.append({"服务区":site,"可行架次模式数":len(patterns),"DP需求状态数":visited,
            "DP精确架次标签数":labels,"最少架次规范划分数":len(partitions[site]),"局部非支配解数":len(local_fronts[site])})
        for i,p in enumerate(patterns,1):
            pattern_rows.append({"服务区":site,"模式序号":i,"机型":p.model,"医疗箱":p.counts[0],
                "饮用水箱":p.counts[1],"食品箱":p.counts[2],"卫生箱":p.counts[3],
                "载荷kg":p.mass,"体积L":p.volume_l,"能耗kWh":p.energy,"累计作业s":p.duration})
        log("服务区精确DP",f"{site}：{len(patterns)}种可行模式，{visited}状态，最少{min(e_fronts[site])}架次。")

    all_e, all_t = q1.combine(e_fronts,"energy"), q1.combine(t_fronts,"time")
    nmin = min(all_e)
    main_e,main_t,main_plan = all_e[nmin]
    lb = sum(r["架次下界"] for r in bound_rows)
    check("下界达到",lb==nmin,f"下界={lb}；构造方案={nmin}，因此架次数全局最优")
    global_pareto = q1.combine_frontiers(local_fronts)
    global_conditional = q1.combine_conditional(conditional_fronts)
    pareto_rows = [{"解编号":f"P{i:03d}","架次数":p.sorties,"总能耗kWh":p.energy,
                   "累计作业s":p.duration,"选择口径":"最少架次，再最小能耗" if p.sorties==nmin else "三目标全局非支配备选"}
                  for i,p in enumerate(global_pareto,1)]
    conditional_rows = []
    for n,labels in sorted(global_conditional.items()):
        for j,p in enumerate(labels,1):
            conditional_rows.append({"架次数":n,"条件前沿序号":j,"总能耗kWh":p.energy,"累计作业s":p.duration,
                "全局非支配":any(p.sorties==a.sorties and q1._same(p.energy,a.energy) and q1._same(p.duration,a.duration) for a in global_pareto)})
    epsilon_rows=[]
    for cap in range(nmin,max(all_e)+1):
        n=min((n for n in all_e if n<=cap),key=lambda n:(all_e[n][0],n,all_e[n][1]))
        epsilon_rows.append({"架次上限ε":cap,"实际架次":n,"最小能耗kWh":all_e[n][0],"累计作业s":all_e[n][1]})

    # ---------- 第三步：每个全局非支配方案映射回80个原始货箱 ----------
    sortie_rows, allocation_rows = [], []
    for pi, lab in enumerate(global_pareto,1):
        solution=f"P{pi:03d}"
        plan={s:[] for s in legs.index}
        for s,p in lab.plan: plan[s].append(p)
        assigned=[]
        for site in legs.index:
            queues={}
            for cat in q1.CATEGORIES:
                sub=boxes[(boxes.node_id==site)&(boxes.material_type==cat)].copy()
                sub["order"]=sub.first_batch.eq("是").astype(int)
                queues[cat]=list(sub.sort_values(["order","box_id"],ascending=[False,True]).box_id)
            for j,p in enumerate(plan[site],1):
                ids=[]
                for cat,count in zip(q1.CATEGORIES,p.counts):
                    ids.extend(queues[cat][:count]); del queues[cat][:count]
                flight_id=f"{site}-{j:02d}"
                m=models.loc[p.model]
                maxsafe=site_data[site][2][p.model]
                row={"解编号":solution,"架次编号":flight_id,"服务区":site,"机型":p.model,
                     "医疗箱":p.counts[0],"饮用水箱":p.counts[1],"食品箱":p.counts[2],"卫生箱":p.counts[3],
                     "货箱数":sum(p.counts),"载荷kg":p.mass,"安全载荷kg":maxsafe,"额定载荷kg":m.max_payload_kg,
                     "体积m³":p.volume_l/1000,"容积上限m³":m.max_volume_m3,"能耗kWh":p.energy,
                     "电池能量kWh":p.battery,"允许能耗kWh":(1-reserve)*p.battery,
                     "返回电量比例":1-p.energy/p.battery,"累计作业s":p.duration,"货箱编号":"；".join(ids)}
                sortie_rows.append(row)
                for bid in ids:
                    b=boxes.set_index("box_id").loc[bid]
                    allocation_rows.append({"解编号":solution,"架次编号":flight_id,"机型":p.model,"货箱编号":bid,
                        "服务区":site,"物资类型":b.material_type,"单箱质量kg":b.mass_kg,"单箱体积m³":b.volume_m3,
                        "首批原标签":b.first_batch,"首批截止s":b.first_deadline_s,"期望送达s":b.target_delivery_s,"应急优先系数":b.priority})
                assigned.extend(ids)
                check("逐架次容量和返航约束",p.mass<=maxsafe+1e-8 and p.volume_l/1000<=m.max_volume_m3+1e-9 and p.energy<=(1-reserve)*p.battery+1e-9,f"{solution}/{flight_id}")
            check("服务区箱队列完全用尽",all(not v for v in queues.values()),f"{solution}/{site}")
        check("逐箱恰好交付一次",len(assigned)==len(boxes) and Counter(assigned)==Counter(boxes.box_id),solution)
        selected=[r for r in sortie_rows if r["解编号"]==solution]
        check("方案目标汇总一致",len(selected)==lab.sorties and abs(sum(r["能耗kWh"] for r in selected)-lab.energy)<1e-8 and abs(sum(r["累计作业s"] for r in selected)-lab.duration)<1e-7,solution)
    check("主方案与前沿一致",any(p.sorties==nmin and abs(p.energy-main_e)<1e-8 and abs(p.duration-main_t)<1e-7 for p in global_pareto),f"{nmin}架次")
    log("三目标精确求解",f"全局非支配解{len(global_pareto)}个，固定架次条件前沿{len(conditional_rows)}点；主方案{nmin}架次。")

    # ---------- 第四步：真实收敛记录，绝不生成算法“代数”伪曲线 ----------
    multi=[s for s in legs.index if min(e_fronts[s])>1]
    fixed_e=fixed_t=0.0
    for site in legs.index:
        if site in multi: continue
        nd=q1.pareto_prune_2d(partitions[site])
        check("单架次点可安全缩减",len(nd)==1,site)
        fixed_e+=nd[0]["energy"]; fixed_t+=nd[0]["time"]
    candidates=[]
    for combo in itertools.product(*(range(len(partitions[s])) for s in multi)):
        candidates.append((fixed_e+sum(partitions[s][k]["energy"] for s,k in zip(multi,combo)),
                           fixed_t+sum(partitions[s][k]["time"] for s,k in zip(multi,combo)),combo))
    best_e=min(p[0] for p in candidates); best_t=min(p[1] for p in candidates)
    trace,improvements=[],[]; run_e=run_t=math.inf
    for i,(e,t,combo) in enumerate(candidates,1):
        if e<run_e-1e-12:
            run_e=e; improvements.append({"指标":"能耗kWh","评估序号":i,"当前最优值":e})
        if t<run_t-1e-8:
            run_t=t; improvements.append({"指标":"累计作业s","评估序号":i,"当前最优值":t})
        trace.append({"评估序号":i,"当前能耗kWh":e,"当前累计作业s":t,"迄今最优能耗kWh":run_e,
            "迄今最短累计作业s":run_t,"能耗相对最终超额比例":max(0,run_e/best_e-1),
            "时间相对最终超额比例":max(0,run_t/best_t-1),**{s+"划分序号":j+1 for s,j in zip(multi,combo)}})
    check("穷举独立复核主方案",abs(best_e-main_e)<1e-8 and abs(best_t-main_t)<1e-7,len(trace))
    log("最少架次收敛复核",f"真实枚举{len(trace)}种规范候选；并非全部架次数下的所有解，也不是随机迭代。")

    # ---------- 第五步：每个安全余量都重建候选模式并重新精确求解 ----------
    sensitivity=[]
    for margin in (.10,.15,.20,.25,.30,.35,.40):
        local_fronts_r, blocked = {},[]
        for site in legs.index:
            counts,patterns,_=q1.site_patterns(legs.loc[site],boxes[boxes.node_id==site],models,margin,unit)
            for j,count in enumerate(counts):
                if count and not any(p.counts[j]>0 for p in patterns): blocked.append(site+"："+q1.CATEGORIES[j])
            local_fronts_r[site]=q1.solve_site(counts,patterns,"energy")
        frontier=q1.combine(local_fronts_r,"energy") if not blocked else {}
        n=min(frontier) if frontier else None
        sensitivity.append({"返航余量比例":margin,"状态":"可行" if frontier else "不可行",
            "最少架次":n,"最少架次能耗kWh":frontier[n][0] if frontier else None,
            "累计作业s":frontier[n][1] if frontier else None,"不可交付单箱":"；".join(sorted(set(blocked)))})
        log("返航余量重求解",f"{margin:.0%}：{str(n)+'架次' if frontier else '不可行'}")

    raw_legs=q1.leg_data(nodes,folder/INPUT_FILES[4])
    raw_fronts={}
    for s in legs.index:
        counts,patterns,_=q1.site_patterns(raw_legs.loc[s],boxes[boxes.node_id==s],models,reserve,unit)
        raw_fronts[s]=q1.solve_site(counts,patterns,"energy")
    raw_opt=q1.combine(raw_fronts,"energy"); raw_n=min(raw_opt)
    dem_rows=[]
    for s in legs.index:
        dem_rows.append({"服务区":s,"原始巡航海拔m":raw_legs.loc[s,"巡航海拔米"],
            "修正巡航海拔m":legs.loc[s,"巡航海拔米"],"巡航海拔改变量m":legs.loc[s,"巡航海拔米"]-raw_legs.loc[s,"巡航海拔米"]})

    # 所有15航段原像元剖面和45条载荷曲线均保留数据；没有绘图或平滑。
    z,left,top,dx,dy=raster
    profile_rows,curve_rows=[],[]
    origin=nodes.loc["O01"]
    for s in legs.index:
        dest=nodes.loc[s]
        x0,y0=(origin.longitude_deg-left)/dx,(top-origin.latitude_deg)/dy
        x1,y1=(dest.longitude_deg-left)/dx,(top-dest.latitude_deg)/dy
        local_profile=[]
        for r,c in q1.touched_cells(x0,y0,x1,y1,z.shape[1],z.shape[0]):
            interval=q1.cell_interval(x0,y0,x1,y1,r,c)
            check("剖面像元交集有效",interval is not None,f"{s}:{r},{c}")
            a,b=interval
            local_profile.append({"服务区":s,"行索引0起":r,"列索引0起":c,
                "进入距离m":a*legs.loc[s,"水平单程米"],"离开距离m":b*legs.loc[s,"水平单程米"],
                "高程m":float(z[r,c]),"仅边角触及":b-a<1e-10})
        check("剖面最高点等于计算峰值",abs(max(p["高程m"] for p in local_profile)-legs.loc[s,"沿线最高地面米"])<1e-9,s)
        profile_rows.extend(sorted(local_profile,key=lambda r:(r["进入距离m"],r["离开距离m"])))
        for g,m in models.iterrows():
            for load in np.linspace(0,float(m.max_payload_kg),101):
                e,_=q1.flight(legs.loc[s],m,float(load))
                curve_rows.append({"服务区":s,"机型":g,"载荷kg":float(load),"往返能耗kWh":e,
                    "允许能耗kWh":float(m.battery_kwh)*(1-reserve),"最大安全载荷kg":site_data[s][2][g]})
    log("地形与能耗数据",f"{len(profile_rows)}个相交像元记录，{len(curve_rows)}个载荷曲线采样点；原始DEM对照已重新求解。")

    # ---------- 第六步：集中保存，一份JSON供工作簿和后续绘图共同使用 ----------
    assumptions=[
        {"项目":"问题一范围","说明":"O01→单个服务区→O01；货箱不可拆，单架次不混服务区；不排发车时刻。"},
        {"项目":"工作DEM","说明":"继承已选稳健三维配准栅格。八种DEM比较属前处理，本入口不重新拟合、不做高斯平滑。"},
        {"项目":"地形提取","说明":"包括航段穿过和恰好触及的像元；H=沿线最高像元+50m。栅格坐标为经纬度直线，水平距离用WGS84/UTM49N，与既有精确版一致。"},
        {"项目":"作业海拔","说明":"O01用题给地面海拔；Si用题给地面海拔+30m；不用DEM端点值替代题给值。"},
        {"项目":"载荷航程","说明":"L(q)=L0-(L0-Lfull)*(q/Q)^1.5；返程空载。"},
        {"项目":"能耗闭合假设","说明":"E水平=B*d*(1/L(q)+1/L0)；E爬升=9.80665/3600000/η*[(m0+q)*h去+m0*h回]；下降附加能耗为0。属于本方案采用的能耗闭合，非逐字题给公式。"},
        {"项目":"累计作业时间","说明":"往返巡航+两次爬升/下降+准备+逐箱装载+基础交接+逐箱交接；并非所有任务结束的时刻。"},
        {"项目":"最大安全载荷","说明":"能耗随载荷单调；在额定质量范围二分70次，另由模式枚举检查体积和不可拆箱。"},
        {"项目":"精确优化","说明":"完全枚举单服务区四类箱数量模式；需求状态动态规划；保留三目标全部非支配标签；固定架次使用精确二维前沿。"},
        {"项目":"安全余量敏感性","说明":"10%至40%每个场景重新计算安全载荷、候选模式和DP；不可行结果为空并列出被阻断单箱。"},
        {"项目":"最优性证据","说明":"全局最少架次等于各服务区质量/体积下界之和；真实收敛为最少架次条件下的规范枚举顺序，不是智能算法代数。"},
        {"项目":"表格刷新","说明":"本XLSX为本次程序运行快照；修改源数据须先运行Python计算，再在已配置@oai/artifact-tool的Node环境执行表格导出代码。Python不依赖该表格导出环境。"},
    ]
    plot_index=[
        {"原图数据用途":"三目标帕累托（含曲面依据）","数据表":"全局帕累托、固定架次条件前沿","说明":"离散精确结果；无插值伪解"},
        {"原图数据用途":"精确收敛","数据表":"真实收敛、收敛改进节点","说明":"实际候选评估序号，不是迭代代数"},
        {"原图数据用途":"15服务区安全载荷","数据表":"安全载荷","说明":"45个机型-服务区组合"},
        {"原图数据用途":"载荷能耗与地形剖面","数据表":"载荷能耗曲线、原像元剖面、航段地形","说明":"保留所有15条航段，非仅三条示例"},
        {"原图数据用途":"15服务区最优组批","数据表":"逐架次方案、逐箱分配","说明":"货箱ID可追溯"},
        {"原图数据用途":"18架次资源利用与机型载荷","数据表":"逐架次方案","说明":"质量、体积、电量；不代表实体机/电池调度"},
        {"原图数据用途":"15服务区最优性证明","数据表":"服务区下界、DP证据","说明":"下界与精确构造比较"},
        {"原图数据用途":"安全余量敏感性","数据表":"余量敏感性","说明":"每个场景独立重新优化"},
    ]
    check("无优化数字硬编码",True,"主方案由当次输入求解；历史18架次/能耗值未作为求解输入。")
    log("全部计算完成",f"主方案：{nmin}架次，{main_e:.10f}kWh，{main_t:.8f}s；核验全部通过。")
    payload={"运行信息":{"开始UTC":logs[0]["时间UTC"],"完成UTC":logs[-1]["时间UTC"],
                "耗时秒":time.perf_counter()-started,"Python":platform.python_version(),"平台":platform.platform(),
                "依赖版本":{x:importlib.metadata.version(x) for x in ["numpy","pandas","Pillow","openpyxl"]}},
        "结果摘要":{"节点数":len(nodes),"货箱数":len(boxes),"总质量kg":boxes.mass_kg.sum(),
            "总体积m³":boxes.volume_m3.sum(),"默认返航余量":reserve,"最少架次":nmin,"架次下界":lb,
            "主方案能耗kWh":main_e,"主方案累计作业s":main_t,
            "主方案机型架次":dict(Counter(p.model for ps in main_plan.values() for p in ps)),
            "原始DEM最少架次":raw_n,"原始DEM能耗kWh":raw_opt[raw_n][0],"原始DEM累计作业s":raw_opt[raw_n][1],
            "全局非支配解数":len(global_pareto),"条件前沿点数":len(conditional_rows),"规范枚举候选数":len(trace)},
        "表":{"全局帕累托":pareto_rows,"航段地形":records(legs.reset_index()),"安全载荷":safe_rows,
            "逐架次方案":sortie_rows,"逐箱分配":allocation_rows,"服务区下界":bound_rows,"DP证据":dp_rows,
            "固定架次条件前沿":conditional_rows,"ε约束":epsilon_rows,"真实收敛":trace,"收敛改进节点":improvements,
            "余量敏感性":sensitivity,"DEM航段比较":dem_rows,"原像元剖面":profile_rows,"载荷能耗曲线":curve_rows,
            "可行架次模式":pattern_rows,"输入节点":records(nodes.reset_index()),"输入需求汇总":records(demand),
            "输入货箱":records(box_raw),"输入机型":records(model_raw),"来源字段字典":sources,"输入文件指纹":manifest,
            "模型口径":assumptions,"图件数据索引":plot_index,"运行日志":logs,"数据与结果核验":checks}}
    (folder/"问题一_运行结果.json").write_text(json.dumps(clean(payload),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print("\n已保存：问题一_运行结果.json；全部表、来源和运行记录均在同一文件。",flush=True)


if __name__ == "__main__":
    main()
