"""复算固定 V9 问题三方案的两组/三组分区，导出十八张问题四图。"""
from pathlib import Path
import subprocess,sys,os,json,time,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
TASKS={
 'compute':{'script':'compute_q4.py','args':['--source',str(ROOT/'data'/'sources'),'--output',str(ROOT/'data')],'images':[]},
 'maps':{'script':'plot_q4_maps.py','args':['--q3-dir',str(ROOT/'q3_source'),'--data-dir',str(ROOT/'data'),'--output-dir',str(OUT),'--dpi','340'],'images':['问题四_图01_六面板分组空间比较.png','问题四_图02_六面板三维分组航线.png']},
 'statistics':{'script':'plot_q4_statistics.py','args':['--q3-dir',str(ROOT/'q3_source'),'--data-dir',str(ROOT/'data'),'--output-dir',str(OUT),'--dpi','320'],'images':['问题四_图03_分组配置与任务负载.png','问题四_图04_分类资源占用与分组门槛.png']},
 'equity_hex':{'script':'plot_q4_equity_hex.py','args':['--q3-dir',str(ROOT/'q3_source'),'--data-dir',str(ROOT/'data'),'--output-dir',str(OUT),'--dpi','300'],'images':['问题四_图05_服务距离与任务分组蜂巢图.png','问题四_图06_资源不平等与服务等待诊断.png']},
 'compute_spatial':{'script':'compute_q4_spatial.py','args':['--q3-dir',str(ROOT/'q3_source'),'--data-dir',str(ROOT/'data')],'images':[]},
 'cpsat':{'script':'compute_q4_cpsat.py','args':['--data-dir',str(ROOT/'data')],'images':[]},
 'extended':{'script':'plot_q4_extended.py','args':['--q3-dir',str(ROOT/'q3_source'),'--data-dir',str(ROOT/'data'),'--output-dir',str(OUT),'--dpi','300'],'images':['问题四_图08_必须同组约束与完整候选分区.png','问题四_图09_专属资源配置与分类库存缺口.png','问题四_图10_资源独占占用与充电时序证据.png','问题四_图11_分区资源冗余与组间负担分解.png']},
 'terrain_ellipses':{'script':'plot_q4_terrain_ellipses.py','args':['--q3-dir',str(ROOT/'q3_source'),'--data-dir',str(ROOT/'data'),'--output-dir',str(OUT),'--dpi','300'],'images':['问题四_图07_真实DEM地形与固定航线分组.png','问题四_图12_任务组件及全部两组候选_三维地形航线.png','问题四_图13_全部候选分组的空间质心与标准差椭圆.png']},
 'spatial_main':{'script':'plot_q4_spatial_main.py','args':['--q3-dir',str(ROOT/'q3_source'),'--data-dir',str(ROOT/'data'),'--output-dir',str(OUT),'--dpi','300'],'images':['问题四_图14_DEM参考飞行时间与分组负担蜂巢图.png','问题四_图15_八类资源配置与B_C型占用阶梯.png','问题四_图16_资源代价_运输公平与空间紧凑性取舍.png','问题四_图17_超图任务块_公平下界与资源共享代价.png']},
 'final_answer':{'script':'plot_q4_final_answer.py','args':['--q3-dir',str(ROOT/'q3_source'),'--data-dir',str(ROOT/'data'),'--output-dir',str(OUT),'--dpi','300'],'images':['问题四_图18_最终答案_两组三组分区与八类资源配置.png']}}
RUNS=[]
def begin():
 global RUNS
 RUNS=[];OUT.mkdir(exist_ok=True)
 (OUT/'run_log.txt').write_text('问题四实际复算与重绘记录\n',encoding='utf-8')
 return {'输入':'与上传V9逐字节比对的20运输、3中继、80箱；包内修正DEM','流程':'运输通信超图收缩→CP-SAT分区和资源峰值→枚举交叉核验→DEM参考阻抗与公平下界→十八图及三维地形资源对比图','输出目录':str(OUT)}
def run_step(key):
 spec=TASKS[key];start=time.perf_counter();env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MPLBACKEND='Agg');env.pop('Q2_CERTIFIED_PREFIX',None)
 record={'step':key,'started_utc':datetime.now(timezone.utc).isoformat(),'command':['python',spec['script']]+spec['args'],'status':'RUNNING'}
 RUNS.append(record)
 try:
  p=subprocess.run([sys.executable,str(ROOT/spec['script'])]+spec['args'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
  record.update(returncode=p.returncode,stdout=(p.stdout or b'').decode('utf-8',errors='replace'),stderr=(p.stderr or b'').decode('utf-8',errors='replace'))
  with (OUT/'run_log.txt').open('a',encoding='utf-8') as f:f.write('\n'+key+'\n'+record['stdout']+'\n'+record['stderr'])
  if p.returncode:raise RuntimeError(record['stderr'] or record['stdout'])
  images=[]
  from PIL import Image
  for name in spec['images']:
   path=OUT/name
   with Image.open(path) as im:im.load();size=list(im.size)
   images.append({'file':name,'size_px':size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
  record.update(status='SUCCESS',images=images)
 except Exception as e:record.update(status='FAILED',error=str(e));raise
 finally:
  record['elapsed_s']=round(time.perf_counter()-start,3)
  (OUT/'run_records.json').write_text(json.dumps(RUNS,ensure_ascii=False,indent=2),encoding='utf-8')
 return {'步骤':key,'结果':record['status'],'耗时秒':record['elapsed_s'],'图片':spec['images']}
def main():
 print(begin())
 for key in TASKS:print(run_step(key))
if __name__=='__main__':main()
