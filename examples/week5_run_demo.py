"""Run a deterministic full episode using an observation-only Week 5 policy."""
import argparse, json, platform, random, subprocess, sys
from decimal import Decimal
from pathlib import Path

def revision():
    dirty=subprocess.run(['git','status','--porcelain'],capture_output=True,text=True).stdout.strip()
    return 'uncommitted' if dirty else subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()

from dumbocracy.campaign import CampaignAction
from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.engine import GameEngine
from dumbocracy.logic import CanonicalTrace, evaluate_diagnostics
from dumbocracy.verification import verify_episode
from dumbocracy.strategies import (GreedyPublicModelStrategy, HistoryEntry, NoOpStrategy,
                                   PublicActionSummary, SeededRandomLegalStrategy, UniformBudgetStrategy)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--backend',default='cpu'); parser.add_argument('--nodes',type=int,default=1000); parser.add_argument('--seed',type=int,default=20261019); parser.add_argument('--rounds',type=int,default=5); parser.add_argument('--strategy',choices=['no_op','seeded_random_legal','uniform_budget','greedy_public_model'],default='greedy_public_model'); parser.add_argument('--output',default='reports/results/week5/demo_episode.json'); args=parser.parse_args()
    config=GameConfig(SeedConfig(args.seed),GraphSpec('erdos_renyi',args.nodes,min(1,8/max(1,args.nodes-1))),args.rounds,Decimal('20.00'),DynamicsConfig(.5),ActionConfig(.05,.02))
    engine=GameEngine(args.backend); state=engine.initialize(config); policy={'no_op':NoOpStrategy(),'seeded_random_legal':SeededRandomLegalStrategy(),'uniform_budget':UniformBudgetStrategy(args.rounds),'greedy_public_model':GreedyPublicModelStrategy(args.seed)}[args.strategy]; rng=random.Random(config.seeds.strategy); history=[]; observations=[engine.observe(state).to_public_dict()]; actions=[]
    while not state.terminal:
        obs=engine.observe(state); action=policy.choose(obs,tuple(history),engine.legal_actions(state),rng); step=engine.step(state,action); actions.append({'action_id':action.action_id,'target_ids':list(action.target_ids),'accepted':step.accepted,'error_code':step.error_code}); history.append(HistoryEntry(obs,PublicActionSummary(state.round_index,action.action_id,step.accepted,step.error_code))); state=step.state; observations.append(engine.observe(state).to_public_dict())
    result={'schema_version':'1.0','seed':args.seed,'backend':state.backend,'strategy':policy.name,'node_count':args.nodes,'horizon':args.rounds,'graph_summary':{'node_count':state.graph_summary.node_count,'edge_count':state.graph_summary.edge_count,'density':state.graph_summary.density,'mean_degree':state.graph_summary.mean_degree,'component_count':state.graph_summary.component_count},'actions':actions,'aggregate_support_trajectory':[o['aggregate_support'] for o in observations],'remaining_budget_trajectory':[o['remaining_budget'] for o in observations],'result':state.result.to_public_dict(),'trace_diagnostics':{k:v.to_public_dict() for k,v in evaluate_diagnostics(CanonicalTrace.from_game_state(state)).items()},'verification':verify_episode(state,engine),'revision':revision(),'command':' '.join([sys.executable,*sys.argv]),'python':platform.python_version(),'information_profile':'aggregate_v1'}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'output':str(path),'backend':state.backend,'result':result['result'],'actions':actions},indent=2))

if __name__=='__main__': main()
