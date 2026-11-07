"""Social aggregation, campaign exposure (workers + SIR cascade, media) and the preference update (P2)."""
from .social import aggregate, aggregate_degroot, aggregate_wmsr
from .update import step_preferences
from .campaign import (
    Allocation, CampaignOutcome, CascadeResult, campaign_round, combine_intensities, media_cost, media_exposure,
    media_intensity, sir_cascade, word_of_mouth_intensity, worker_contacts, worker_cost, worker_intensity,
)
