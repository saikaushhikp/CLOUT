"""Save hand-constructed finite-world epistemic scenarios (not universal proofs)."""
import json
from decimal import Decimal
from pathlib import Path
from dumbocracy.config import ActionConfig,DynamicsConfig,GameConfig,GraphSpec,SeedConfig
from dumbocracy.engine import GameEngine
from dumbocracy.epistemic import check_finite_worlds
from dumbocracy.graph import ElectorateGraph
from dumbocracy.mas import VoterState

def main():
 graph=ElectorateGraph(('a','b'),()); cfg=GameConfig(SeedConfig(3),GraphSpec('edge_list',2),1,Decimal('1.00'),DynamicsConfig(0),ActionConfig(.02,.1)); engine=GameEngine()
 high=engine.initialize(cfg,graph,(VoterState('a',.8,1),VoterState('b',.9,1)))
 low_cfg=GameConfig(SeedConfig(4),GraphSpec('edge_list',2),1,Decimal('0.00'),DynamicsConfig(0),ActionConfig(.02,.1));low=engine.initialize(low_cfg,graph,(VoterState('a',.1,1),VoterState('b',.2,1)))
 a=engine.initialize(cfg,graph,(VoterState('a',.49,1),VoterState('b',.9,1)));b=engine.initialize(cfg,graph,(VoterState('a',.9,1),VoterState('b',.49,1)))
 scenarios={'possible_and_guaranteed':(high,),'inevitable_defeat':(low,),'mixed_indistinguishable':(a,b)}
 outputs={name:check_finite_worlds(engine,worlds).to_public_dict() for name,worlds in scenarios.items()}
 path=Path('reports/results/week5/model_check.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(outputs,indent=2)+'\n');print(json.dumps(outputs,indent=2))
if __name__=='__main__':main()
