"""Repeated explicit-device sparse network scale test; CPU fallback is forbidden."""
import argparse, json, platform, subprocess, time, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from dumbocracy.campaign import CampaignAction
from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.engine import GameEngine
from decimal import Decimal

def revision():
    dirty=subprocess.run(['git','status','--porcelain'],capture_output=True,text=True).stdout.strip()
    return 'uncommitted' if dirty else subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda:0'); p.add_argument('--nodes',type=int,nargs='+',default=[1000,10000,100000]); p.add_argument('--seed',type=int,default=20261019); p.add_argument('--rounds',type=int,default=5); p.add_argument('--repeats',type=int,default=3); p.add_argument('--output',default='reports/results/week5/gpu_scale.jsonl'); a=p.parse_args()
    if not a.device.startswith('cuda') or not torch.cuda.is_available(): raise SystemExit('Explicit CUDA device is required; CUDA unavailable (no CPU fallback).')
    started_at=datetime.now(timezone.utc).isoformat();command=' '.join([sys.executable,*sys.argv]);dev=torch.device(a.device); index=dev.index or 0
    if index>=torch.cuda.device_count(): raise SystemExit(f'CUDA device {a.device} does not exist')
    torch.cuda.set_device(index)
    driver=subprocess.run(['nvidia-smi','--query-gpu=driver_version,memory.total,memory.free','--format=csv,noheader','-i',str(index)],capture_output=True,text=True)
    driver_text=driver.stdout.strip() if driver.returncode==0 else f'nvidia-smi exit {driver.returncode}: {driver.stderr.strip()}'
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); rows=[]
    for n in a.nodes:
        for repeat in range(a.repeats):
            cfg=GameConfig(SeedConfig(a.seed),GraphSpec('erdos_renyi',n,min(1,8/max(1,n-1))),a.rounds,Decimal('20.00'),DynamicsConfig(.5),ActionConfig(.05,.02))
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter(); engine=GameEngine(a.device); state=engine.initialize(cfg); init=time.perf_counter()-t; step_times=[]
            for _ in range(a.rounds):
                action=CampaignAction('broadcast') if state.ledger.remaining_budget>=cfg.actions.broadcast_cost and not state.broadcast_used else CampaignAction('no_op')
                torch.cuda.synchronize(); t=time.perf_counter(); state=engine.step(state,action).state; torch.cuda.synchronize(); step_times.append(time.perf_counter()-t)
            rows.append({'n':n,'repeat':repeat,'seed':a.seed,'backend':state.backend,'device':torch.cuda.get_device_name(index),'cuda_runtime':torch.version.cuda,'torch':torch.__version__,'python':platform.python_version(),'driver_memory':driver_text,'graph':{'edges':len(state.graph.edges),'density':state.graph_summary.density,'mean_degree':state.graph_summary.mean_degree,'components':state.graph_summary.component_count},'initialization_seconds':init,'step_seconds':step_times,'episode_seconds':init+sum(step_times),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'result':state.result.to_public_dict(),'command':command,'started_at_utc':started_at,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'exit_status':0,'revision':revision()})
    with path.open('w') as f:
        for row in rows: f.write(json.dumps(row)+'\n')
    for row in rows: print(json.dumps(row))

if __name__=='__main__': main()
