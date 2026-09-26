/** 问题一结果表导出。输入：问题一_运行结果.json；输出：问题一_结果汇总.xlsx。
 * 依赖：Node.js 及已配置的 @oai/artifact-tool（不属于 Python requirements）。
 * 运行：node 问题一_表格导出.mjs [代码与数据所在目录]
 * 该依赖在普通 Python/Node 安装中不自带。未配置时，可直接使用随包已运行的
 * XLSX，并用 Python 完整复算 JSON。优化绝不由 Excel 单元格伪装重求解。
 */
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { Workbook, SpreadsheetFile, FileBlob } from "@oai/artifact-tool";

const args = process.argv.slice(2);
const dataDir = args[0] && !args[0].startsWith("--") ? path.resolve(args[0]) : path.dirname(fileURLToPath(import.meta.url));
const qaIndex = args.indexOf("--qa-dir");
const qaDir = qaIndex >= 0 ? args[qaIndex+1] : null;
if (qaDir) await fs.mkdir(qaDir, { recursive: true });
const priorIndex = args.indexOf("--prior");
if (priorIndex >= 0 && qaDir) {
  const old = await SpreadsheetFile.importXlsx(await FileBlob.load(args[priorIndex+1]));
  console.log((await old.inspect({kind:"sheet",include:"id,name"})).ndjson);
  const preview = await old.render({sheetName:"表1_三目标结果",range:"A1:K9",scale:1,format:"png"});
  await fs.writeFile(path.join(qaDir,"previous.png"),new Uint8Array(await preview.arrayBuffer()));
}
const data = JSON.parse(await fs.readFile(path.join(dataDir,"问题一_运行结果.json"),"utf8"));
const t = data["表"], summary = data["结果摘要"];
const wb = Workbook.create();
const FONT="Droid Sans Fallback", INK="#1C2531", BLUE="#334F70", PALE="#EDF2F7", RULE="#C6D1DC";
const sheetNames=["结果汇总","航段载荷","逐架次方案","逐箱分配","最优性证据","精确前沿","真实收敛","剖面数据","能耗曲线","可行模式","输入与来源","运行与核验"];
const sheets=Object.fromEntries(sheetNames.map(name=>[name,wb.worksheets.add(name)]));
const loc={};
function col(n){let out="";for(;n>0;n=Math.floor((n-1)/26))out=String.fromCharCode(65+(n-1)%26)+out;return out;}
function title(sh,text,note,last="H"){
  sh.showGridLines=false;
  sh.getRange(`A1:${last}3`).format.font={name:FONT,size:12,color:INK};
  sh.getRange("A2").values=[[text]];
  sh.getRange("A2").format.font={name:FONT,size:17,bold:true,color:INK};
  sh.getRange("A2").format.rowHeight=32;
  sh.getRange("A3").values=[[note]];
  sh.getRange(`A3:${last}3`).format.rowHeight=28;
  sh.getRange(`A3:${last}3`).format.borders={bottom:{style:"thin",color:RULE}};
  sh.freezePanes.freezeRows(5);
}
function section(sh,row,text,last="H"){
  const r=sh.getRange(`A${row}:${last}${row}`);
  r.format.font={name:FONT,size:12,bold:true,color:INK};
  r.format.rowHeight=30;
  sh.getRange(`A${row}`).values=[[text]];
}
function table(sh,headerRow,rows,headers=null,widths=null,key=null){
  headers??=Object.keys(rows[0]??{});
  const n=rows.length, last=col(headers.length), start=headerRow+1, end=headerRow+n;
  const all=sh.getRange(`A${headerRow}:${last}${Math.max(headerRow,end)}`);
  all.format.font={name:FONT,size:12,color:INK};
  all.format.verticalAlignment="center";
  all.format.rowHeight=24;
  const head=sh.getRange(`A${headerRow}:${last}${headerRow}`);
  head.values=[headers];
  head.format={fill:BLUE,font:{name:FONT,size:12,bold:true,color:"#FFFFFF"},
    horizontalAlignment:"center",verticalAlignment:"center",wrapText:true,rowHeight:42,
    borders:{top:{style:"medium",color:BLUE},bottom:{style:"thin",color:BLUE},insideVertical:{style:"thin",color:"#FFFFFF"}}};
  if(n){
    const values=rows.map(r=>headers.map(h=>r[h]??null));
    sh.getRange(`A${start}:${last}${end}`).values=values;
    sh.getRange(`A${end}:${last}${end}`).format.borders={bottom:{style:"thin",color:RULE}};
    headers.forEach((h,j)=>{
      const r=sh.getRange(`${col(j+1)}${start}:${col(j+1)}${end}`);
      const sample=rows.map(row=>row[h]).find(x=>x!==null&&x!==undefined);
      if(typeof sample==="number"){
        r.format.horizontalAlignment="right";
        const integers=rows.filter(row=>row[h]!==null&&row[h]!==undefined).every(row=>Number.isInteger(row[h]));
        r.setNumberFormat(/比例|利用率/.test(h)?"0.00%":/kWh|kg|m³|累计作业|距离|海拔|地面|爬升|下降|高程|体积|能耗/.test(h)?"0.0000":integers?"0":"0.0000");
      } else r.format.horizontalAlignment="left";
      if(h==="时间UTC")r.setNumberFormat('yyyy-mm-dd hh:mm:ss" UTC"');
    });
  }
  headers.forEach((h,j)=>sh.getRange(`${col(j+1)}:${col(j+1)}`).format.columnWidth=(widths?.[j]??Math.max(14,Math.min(27,h.length*1.8+4))));
  if(key)loc[key]={sheet:sh.name,start,end,headers};
  return end+4;
}
function formulas(sh,column,start,list){sh.getRange(`${column}${start}:${column}${start+list.length-1}`).formulas=list.map(s=>[s]);}

// 一张工作簿内保留正文表及全部计算记录，无图件。
{
 const sh=sheets["逐架次方案"];
 title(sh,"逐架次精确方案","P001 为主方案；P002 为节能备选。架次编号不表示发车顺序。","W");
 const rows=t["逐架次方案"].map(r=>({...r,"安全载荷利用率":null,"额定载荷利用率":null,"体积利用率":null}));
 table(sh,5,rows,null,[12,17,13,10,11,11,11,11,12,15,16,16,15,17,17,17,17,17,19,82,18,18,16],"sorties");
 const a=6,b=a+rows.length-1;
 formulas(sh,"I",a,rows.map((r,i)=>`=SUM(E${a+i}:H${a+i})`));
 formulas(sh,"R",a,rows.map((r,i)=>`=1-O${a+i}/P${a+i}`));
 formulas(sh,"U",a,rows.map((r,i)=>`=J${a+i}/K${a+i}`));
 formulas(sh,"V",a,rows.map((r,i)=>`=J${a+i}/L${a+i}`));
 formulas(sh,"W",a,rows.map((r,i)=>`=M${a+i}/N${a+i}`));
 sh.getRange(`R${a}:R${b}`).setNumberFormat("0.00%");
 sh.getRange(`U${a}:W${b}`).setNumberFormat("0.00%");
}
{
 const sh=sheets["结果汇总"];sh.tabColor=BLUE;
 title(sh,"问题一｜精确优化结果","80 个不可拆货箱；15 个服务区；返航余量 20%；累计作业时间不等于完工时刻。","H");
 const rows=t["全局帕累托"].map(r=>({"解编号":r["解编号"],"方案":r["选择口径"],"架次数":null,
   "总能耗kWh":null,"累计作业h":null,"B型架次":null,"C型架次":null,"最低返回电量比例":null}));
 let next=table(sh,5,rows,null,[13,34,14,19,19,15,15,25]);
 const sr=loc.sorties;
 t["全局帕累托"].forEach((r,i)=>{
   const rr=6+i, id=r["解编号"];
   const selected=t["逐架次方案"].map((p,j)=>({...p,row:6+j})).filter(p=>p["解编号"]===id);
   sh.getRange(`C${rr}`).formulas=[[`=COUNTIF('逐架次方案'!A${sr.start}:A${sr.end},A${rr})`]];
   sh.getRange(`D${rr}`).formulas=[[`=SUMIF('逐架次方案'!A${sr.start}:A${sr.end},A${rr},'逐架次方案'!O${sr.start}:O${sr.end})`]];
   sh.getRange(`E${rr}`).formulas=[[`=SUMIF('逐架次方案'!A${sr.start}:A${sr.end},A${rr},'逐架次方案'!S${sr.start}:S${sr.end})/3600`]];
   for(const [c,g] of [["F","B"],["G","C"]])sh.getRange(`${c}${rr}`).formulas=[[`=COUNTIFS('逐架次方案'!A${sr.start}:A${sr.end},A${rr},'逐架次方案'!D${sr.start}:D${sr.end},"${g}")`]];
   sh.getRange(`H${rr}`).formulas=[[`=MIN(${selected.map(p=>`'逐架次方案'!R${p.row}`).join(",")})`]];
 });
 sh.getRange("C6:C7").setNumberFormat("0");sh.getRange("F6:G7").setNumberFormat("0");
 sh.getRange("D6:E7").setNumberFormat("0.0000");sh.getRange("H6:H7").setNumberFormat("0.00%");
 sh.getRange("C6:H7").format.horizontalAlignment="right";
 section(sh,next,"返航余量敏感性｜每个场景独立重新优化");
 const sensitivity=t["余量敏感性"].map(r=>({...r,"不可交付单箱":r["不可交付单箱"]?`${r["不可交付单箱"].split("；").length}类（核验页明细）`:"—"}));
 next=table(sh,next+1,sensitivity,null,[13,34,14,19,19,57]);
 section(sh,next,"DEM 影响｜相同模型分别重新求解");
 const comps=[{"DEM":"原始DEM","最少架次":summary["原始DEM最少架次"],"能耗kWh":summary["原始DEM能耗kWh"],"累计作业s":summary["原始DEM累计作业s"]},
   {"DEM":"最终工作DEM","最少架次":summary["最少架次"],"能耗kWh":summary["主方案能耗kWh"],"累计作业s":summary["主方案累计作业s"]},
   {"DEM":"修正后减原始","最少架次":null,"能耗kWh":null,"累计作业s":null}];
 const start=next+2;next=table(sh,next+1,comps,null,[25,19,19,24]);
 for(const c of ["B","C","D"])sh.getRange(`${c}${start+2}`).formulas=[[`=${c}${start+1}-${c}${start}`]];
 section(sh,next,"运行摘要");
 table(sh,next+1,[{"核验指标":"最少架次理论下界","数值":summary["架次下界"],"说明":"各服务区质量/体积下界之和；已构造同架次可行方案"},
   {"核验指标":"精确全局非支配解数","数值":summary["全局非支配解数"],"说明":"保留离散解，不把插值曲面作为新解"},
   {"核验指标":"条件前沿点数","数值":summary["条件前沿点数"],"说明":"所有可行精确架次数的能耗—时间二维前沿"},
   {"核验指标":"真实规范候选评估数","数值":summary["规范枚举候选数"],"说明":"18架次条件下，经安全支配删减后完全枚举"}],null,[34,19,78]);
 // 同列有多个表时，保留首表紧凑布局，较长说明允许向右空白单元格延伸。
 [25,35,18,20,21,18,18,27].forEach((w,i)=>sh.getRange(`${col(i+1)}:${col(i+1)}`).format.columnWidth=w);
 sh.getRange(`C${next+2}:C${next+5}`).format.wrapText=false;
}
{
 const sh=sheets["航段载荷"];
 title(sh,"航段地形与安全载荷","巡航海拔＝相交像元最高地面＋50 m；连续安全载荷另受组批体积与不可拆箱约束。","J");
 let next=table(sh,5,t["航段地形"],null,[14,20,23,20,20,20,20,20,24]);
 section(sh,next,"三机型×15服务区：连续最大安全载荷","J");
 next=table(sh,next+1,t["安全载荷"],null,[14,12,21,19,20,20,22,26,25]);
 section(sh,next,"原始与修正 DEM 的航段巡航海拔");
 const start=next+2;table(sh,next+1,t["DEM航段比较"],null,[14,23,23,25]);
 formulas(sh,"D",start,t["DEM航段比较"].map((_,i)=>`=C${start+i}-B${start+i}`));
}
for(const [sheet,key,note] of [
 ["逐箱分配","逐箱分配","每个方案均含80行，分别核验全部货箱恰好交付一次；原时限保留，但本问不排到达时刻。"],
 ["剖面数据","原像元剖面","全部15条航段的原像元高程；保留触边/触角像元，未平滑。距离单位 m，索引从0起。"],
 ["能耗曲线","载荷能耗曲线","15个服务区×3种机型×101个载荷采样点；连续能耗函数采样不是离散组批解。"],
 ["可行模式","可行架次模式","逐服务区完全枚举单架次模式；用于精确DP，四类物资顺序和原清单一致。"]]){
 const sh=sheets[sheet];title(sh,sheet,note,col(Object.keys(t[key][0]).length));table(sh,5,t[key]);
}
{
 const sh=sheets["最优性证据"];title(sh,"最优性证据","理论下界与精确构造一致，证明最少架次；DP状态数和规范划分数来自当次运行。","H");
 const rows=t["服务区下界"].map(r=>({...r,"下界差":null}));
 let next=table(sh,5,rows,null,[15,14,18,19,15,15,16,18,15]);
 formulas(sh,"I",6,rows.map((r,i)=>`=H${6+i}-G${6+i}`));
 section(sh,next,"动态规划状态与候选模式");next=table(sh,next+1,t["DP证据"]);
 section(sh,next,"真实收敛改进节点");table(sh,next+1,t["收敛改进节点"]);
}
{
 const sh=sheets["精确前沿"];title(sh,"精确前沿与 ε 约束","前813行是固定架次数的二维前沿；全局非支配另用标记区分。下方是架次数上限的最小能耗。","F");
 let next=table(sh,5,t["固定架次条件前沿"],null,[17,20,22,24,23]);
 section(sh,next,"ε约束结果：给定最多架次数，再最小化能耗");table(sh,next+1,t["ε约束"]);
}
{
 const sh=sheets["真实收敛"];title(sh,"真实候选评估记录","限定最少架次；行号是规范装载候选评估次序，不是进化算法代数。最优值首次出现不等于证明完成。","J");
 table(sh,5,t["真实收敛"],null,[16,23,25,27,28,29,29,22,22,22]);
}
{
 const sh=sheets["输入与来源"];title(sh,"输入快照与字段来源","原始文件、工作表、列、代码字段和用途均可追溯；无外部工作簿链接，原表保留在同目录。","I");
 let next=table(sh,5,t["来源字段字典"],null,[43,23,62,28,93]);
 section(sh,next,"输入文件指纹：用于核对版本");next=table(sh,next+1,t["输入文件指纹"],null,[43,23,82]);
 section(sh,next,"节点快照：作业海拔由题给地面海拔计算");next=table(sh,next+1,t["输入节点"]);
 section(sh,next,"运输机型原始参数：数据!A3:R5","R");next=table(sh,next+1,t["输入机型"]);
 section(sh,next,"需求汇总原表：只核对，不与逐箱清单重复计数","I");next=table(sh,next+1,t["输入需求汇总"]);
 section(sh,next,"逐箱清单原表：80个不可拆货箱","I");next=table(sh,next+1,t["输入货箱"]);
 // 宽表为可追溯数据，不强行压到一页；原字段与数值完整保留。
 sh.getRange("A:A").format.columnWidth=42;sh.getRange("B:B").format.columnWidth=26;
 sh.getRange("C:C").format.columnWidth=58;sh.getRange("D:D").format.columnWidth=32;sh.getRange("E:E").format.columnWidth=90;
}
{
 const sh=sheets["运行与核验"];title(sh,"运行记录、核验与模型口径","所有优化结果由本次代码实际运行产生；修改输入须重跑，Excel 是计算快照而非优化求解器。","E");
 let next=table(sh,5,t["运行日志"],null,[32,17,28,100]);
 section(sh,next,"环境版本与运行时间");
 const env=[...Object.entries(data["运行信息"]).filter(([k])=>k!=="依赖版本"),...Object.entries(data["运行信息"]["依赖版本"])]
   .map(([key,value])=>({"环境项":key,"值":value}));
 const envStart=next+2;
 next=table(sh,next+1,env,null,[35,105]);
 for(let i=0;i<env.length;i++)if(/UTC$/.test(env[i]["环境项"]))sh.getRange(`B${envStart+i}`).setNumberFormat('yyyy-mm-dd hh:mm:ss" UTC"');
 section(sh,next,"数据与结果核验：按项目汇总，逐条明细在运行结果JSON中");
 const groups=new Map();
 for(const r of t["数据与结果核验"]){const k=r["核验项"];if(!groups.has(k))groups.set(k,{"核验项":k,"检查次数":0,"失败次数":0,"说明示例":r["说明"]});const a=groups.get(k);a["检查次数"]++;if(r["结果"]!=="通过")a["失败次数"]++;}
 next=table(sh,next+1,[...groups.values()],null,[40,17,17,100]);
 section(sh,next,"不可行场景：被能量约束阻断的不可拆单箱");
 const blocked=t["余量敏感性"].flatMap(r=>r["不可交付单箱"]?r["不可交付单箱"].split("；").map(item=>({"返航余量比例":r["返航余量比例"],"服务区与物资类型":item})):[]);
 next=table(sh,next+1,blocked,null,[35,105]);
 section(sh,next,"模型与计算口径");next=table(sh,next+1,t["模型口径"],null,[35,140]);
 section(sh,next,"原图件的数据索引：本包只交付数据，不输出图件");table(sh,next+1,t["图件数据索引"],null,[44,66,90]);
 sh.getRange("A:A").format.columnWidth=43;sh.getRange("B:B").format.columnWidth=36;
 sh.getRange("C:C").format.columnWidth=35;sh.getRange("D:D").format.columnWidth=110;
 // 两列表的长说明向右空白区域展开，保留全部可读文字，不缩小字号。
}
wb.recalculate();
const inspect=await wb.inspect({kind:"table",range:"结果汇总!A5:H7",include:"values,formulas",tableMaxRows:4,tableMaxCols:8});
console.log(inspect.ndjson);
const errors=await wb.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",options:{useRegex:true,maxResults:50}});
console.log(errors.ndjson);
if(qaDir){
 await fs.writeFile(path.join(qaDir,"formula_check.json"),JSON.stringify({summary:inspect,errors},null,2));
 for(const [name,range] of [["结果汇总","A1:H19"],["航段载荷","A1:I21"],["逐架次方案","A1:S12"],["输入与来源","A1:E12"],["运行与核验","A1:D14"]]){
  const im=await wb.render({sheetName:name,range,scale:1,format:"png"});
  await fs.writeFile(path.join(qaDir,name+".png"),new Uint8Array(await im.arrayBuffer()));
 }
}
const out=await SpreadsheetFile.exportXlsx(wb);
const resultPath=path.join(dataDir,"问题一_结果汇总.xlsx");
await out.save(resultPath);
// 工具生成的诊断侧文件不是交付内容，移到临时审阅区，保持代码数据目录平整。
const diagnostic=resultPath+".inspect.ndjson";
try {await fs.access(diagnostic); await fs.rename(diagnostic,path.join(qaDir??os.tmpdir(),`q1_export_${Date.now()}.inspect.ndjson`));}
catch(error){if(error.code!=="ENOENT")throw error;}
console.log("已导出：问题一_结果汇总.xlsx；12个工作表，无图件。");
