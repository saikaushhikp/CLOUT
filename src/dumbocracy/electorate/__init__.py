"""Electorate generation: regions, ER stochastic block model, preferences, attributes (spec §2, P1)."""
from .electorate import Electorate, generate_electorate
from .sbm import Graph

__all__ = ["Electorate", "Graph", "generate_electorate"]
