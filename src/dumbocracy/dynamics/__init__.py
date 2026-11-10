"""Social aggregation, campaign (P8 party workers + IC cascade, media) and the preference update."""
from .social import aggregate, aggregate_degroot, aggregate_wmsr
from .update import step_preferences
from .campaign import (
    Allocation, CampaignOutcome, WorkerState, apply_assignments, campaign_round, combine_intensities,
    conversion_chance, eligible_counts, eligible_workers, ic_hop, media_cost, media_exposure, media_intensity,
    pick_workers, spreader_edge_counts, worker_cost,
)
