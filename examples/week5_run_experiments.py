"""Reproducible Week 5 policy, budget and topology experiments (aggregate only)."""
import argparse, json, random, statistics, subprocess, time, platform, sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from dumbocracy.campaign import CampaignAction
from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.engine import GameEngine
from dumbocracy.graph import erdos_renyi_graph
from dumbocracy.graph_experiments import ring_lattice_graph
from dumbocracy.strategies import (FirstLegalStrategy, GreedyPublicModelStrategy, HistoryEntry,
    NoOpStrategy, PublicActionSummary, SeededRandomLegalStrategy, UniformBudgetStrategy)

POLICIES={'no_op':NoOpStrategy,'first_legal':FirstLegalStrategy,'seeded_random_legal':SeededRandomLegalStrategy,
          'uniform_budget':UniformBudgetStrategy,'greedy_public_model':GreedyPublicModelStrategy}

def revision():
    dirty=subprocess.run(['git','status','--porcelain'],capture_output=True,text=True).stdout.strip()
    return 'uncommitted' if dirty else subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()

def wilson(successes,total,z=1.96):
    if total==0:return [None,None]
    p=successes/total; den=1+z*z/total; center=(p+z*z/(2*total))/den
    margin=z*((p*(1-p)/total+z*z/(4*total*total))**.5)/den
    return [max(0,center-margin),min(1,center+margin)]

def episode(n,horizon,seed,budget,backend,policy_name,topology='er'):
    probability=min(1,8/max(1,n-1)); graph_started=time.perf_counter(); graph=erdos_renyi_graph(n,probability,SeedConfig(seed).graph) if topology=='er' else ring_lattice_graph(n,min(8,n-1 if (n-1)%2==0 else n-2),SeedConfig(seed).graph);graph_seconds=time.perf_counter()-graph_started
    cfg=GameConfig(SeedConfig(seed),GraphSpec('edge_list',n),horizon,Decimal(budget),DynamicsConfig(.5),ActionConfig(.05,.02))
    engine=GameEngine(backend); init_started=time.perf_counter();state=engine.initialize(cfg,graph);init_seconds=time.perf_counter()-init_started; policy_class=POLICIES[policy_name]
    policy=policy_class(horizon) if policy_name=='uniform_budget' else policy_class(seed) if policy_name=='greedy_public_model' else policy_class()
    rng=random.Random(cfg.seeds.strategy); history=[]; actions=[]; trajectory=[engine.observe(state).aggregate_support]; step_times=[]; started=time.perf_counter()
    while not state.terminal:
        obs=engine.observe(state); proposal=policy.choose(obs,tuple(history),engine.legal_actions(state),rng); step_started=time.perf_counter();step=engine.step(state,proposal);step_times.append(time.perf_counter()-step_started)
        actions.append({'action_id':proposal.action_id,'target_ids':list(proposal.target_ids),'accepted':step.accepted,'error_code':step.error_code})
        history.append(HistoryEntry(obs,PublicActionSummary(state.round_index,proposal.action_id,step.accepted,step.error_code))); state=step.state; trajectory.append(engine.observe(state).aggregate_support)
    try:
        import torch; torch_version=torch.__version__;cuda_runtime=torch.version.cuda
    except ImportError: torch_version='not-installed';cuda_runtime=None
    return {'schema_version':'1.0','seed':seed,'topology':topology,'graph_probability':probability,'strategy':policy_name,'backend':state.backend,'node_count':n,'edge_count':len(state.graph.edges),'density':state.graph_summary.density,'mean_degree':state.graph_summary.mean_degree,'component_count':state.graph_summary.component_count,'horizon':horizon,'alpha':cfg.dynamics.alpha,'delta_target':cfg.actions.delta_target,'delta_broadcast':cfg.actions.delta_broadcast,'initial_budget':budget,'remaining_budget':str(state.ledger.remaining_budget),'budget_used':str(state.ledger.spent),'score':state.result.score,'victory':state.result.victory,'actions':actions,'aggregate_support_trajectory':trajectory,'graph_generation_seconds':graph_seconds,'engine_initialization_seconds':init_seconds,'step_seconds':step_times,'episode_seconds':time.perf_counter()-started,'python':platform.python_version(),'numpy':__import__('numpy').__version__,'torch':torch_version,'cuda_runtime':cuda_runtime,'revision':revision(),'information_profile':'aggregate_v1'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--experiment',choices=['strategies','budget','topology','all'],default='all'); p.add_argument('--backend',default='cpu'); p.add_argument('--nodes',type=int,nargs='+',default=[1000]); p.add_argument('--seeds',type=int,nargs='+',default=[11,23,37,41,53]); p.add_argument('--rounds',type=int,default=5); p.add_argument('--strategies',nargs='+',default=['no_op','first_legal','seeded_random_legal','uniform_budget','greedy_public_model']); p.add_argument('--output-dir',default='reports/results/week5'); a=p.parse_args()
    started_at=datetime.now(timezone.utc).isoformat();out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); kinds=['strategies','budget','topology'] if a.experiment=='all' else [a.experiment]; rows=[]
    for kind in kinds:
      for n in a.nodes:
       for seed in a.seeds:
        if kind=='strategies': conditions=[(s,'20.00','er') for s in a.strategies]
        elif kind=='budget': conditions=[('greedy_public_model',factor,'er') for factor in ('10.00','20.00','40.00')]
        else: conditions=[('greedy_public_model','20.00',topo) for topo in ('er','ring_lattice')]
        for strategy,budget,topology in conditions:
          row=episode(n,a.rounds,seed,budget,a.backend,strategy,topology); row['experiment']=kind; rows.append(row)
    path=out/f'{a.experiment}.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    grouped={}
    for row in rows: grouped.setdefault((row['experiment'],row['strategy'],row['topology'],row['initial_budget']),[]).append(row)
    aggregates=[]
    baseline={(r['seed'],r['node_count']):r['score'] for r in rows if r['experiment']=='strategies' and r['strategy']=='no_op'}
    for k,v in grouped.items():
        count=len(v); victories=sum(r['victory'] for r in v); spend=statistics.mean(float(r['budget_used']) for r in v)
        deltas=[(r['score']-baseline[(r['seed'],r['node_count'])])/float(r['budget_used']) for r in v if k[0]=='strategies' and k[1]!='no_op' and float(r['budget_used'])>0 and (r['seed'],r['node_count']) in baseline]
        aggregates.append({'experiment':k[0],'strategy':k[1],'topology':k[2],'budget':k[3],'sample_count':count,'failed':0,'mean_score':statistics.mean(r['score'] for r in v),'score_population_sd':statistics.pstdev(r['score'] for r in v),'victory_rate':victories/count,'victory_wilson_95':wilson(victories,count),'mean_budget_used':spend,'mean_score_delta_per_spend_vs_no_op':statistics.mean(deltas) if deltas else None})
    manifest={'command':' '.join([sys.executable,*sys.argv]),'started_at_utc':started_at,'ended_at_utc':datetime.now(timezone.utc).isoformat(),'exit_status':0,'experiment':a.experiment,'backend':a.backend,'nodes':a.nodes,'seeds':a.seeds,'rounds':a.rounds,'revision':revision(),'information_profile':'aggregate_v1','row_count':len(rows),'failed':0,'skipped':0,'results_file':str(path)}
    (out/f'{a.experiment}_aggregate.json').write_text(json.dumps(aggregates,indent=2)+'\n'); (out/f'{a.experiment}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps({'manifest':manifest,'aggregates':aggregates},indent=2))

if __name__=='__main__': main()
