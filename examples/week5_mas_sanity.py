"""Run named MAS/action sanity schedules and write aggregate-only JSONL evidence."""
import json,time
from decimal import Decimal
from pathlib import Path
from dumbocracy.campaign import CampaignAction
from dumbocracy.config import ActionConfig,DynamicsConfig,GameConfig,GraphSpec,SeedConfig
from dumbocracy.engine import GameEngine

def main():
 seed=20261019;n=1000;horizon=5;p=min(1,8/(n-1)); schedules={
  'no_campaign_peer_dynamics':[CampaignAction('no_op') for _ in range(horizon)],
  'single_broadcast_then_noop':[CampaignAction('broadcast')]+[CampaignAction('no_op') for _ in range(horizon-1)],
  'repeated_broadcast':[CampaignAction('broadcast') for _ in range(horizon)],
  'repeated_public_target':[CampaignAction('targeted_contact',('v0',)) for _ in range(horizon)],
  'peer_only_zero_campaign_effect':[CampaignAction('no_op') for _ in range(horizon)],
  'combined_broadcast_and_peer':[CampaignAction('broadcast'),CampaignAction('targeted_contact',('v0',))]+[CampaignAction('no_op') for _ in range(horizon-2)]}
 rows=[]
 for name,actions in schedules.items():
  zero_effect=name=='peer_only_zero_campaign_effect';cfg=GameConfig(SeedConfig(seed),GraphSpec('erdos_renyi',n,p),horizon,Decimal('30.00'),DynamicsConfig(.5),ActionConfig(0 if zero_effect else .05,0 if zero_effect else .02));engine=GameEngine('cpu');t=time.perf_counter();state=engine.initialize(cfg);init=time.perf_counter()-t;trajectory=[engine.observe(state).aggregate_support];sequence=[]
  for action in actions:
   step=engine.step(state,action);sequence.append({'action_id':action.action_id,'accepted':step.accepted,'error_code':step.error_code});state=step.state;trajectory.append(engine.observe(state).aggregate_support)
  rows.append({'scenario':name,'seed':seed,'backend':state.backend,'node_count':n,'edge_count':state.graph_summary.edge_count,'density':state.graph_summary.density,'mean_degree':state.graph_summary.mean_degree,'component_count':state.graph_summary.component_count,'horizon':horizon,'alpha':cfg.dynamics.alpha,'delta_target':cfg.actions.delta_target,'delta_broadcast':cfg.actions.delta_broadcast,'graph_initialization_seconds':init,'aggregate_support_trajectory':trajectory,'accepted_actions':sequence,'score':state.result.score,'player_votes':state.result.player_votes,'victory':state.result.victory,'budget_used':str(state.ledger.spent),'remaining_budget':str(state.ledger.remaining_budget),'convergence_claim':'none; fixed horizon only'})
 path=Path('reports/results/week5/mas_sanity.jsonl');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(row)+'\n' for row in rows));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
