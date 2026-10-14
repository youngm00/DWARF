"""DWARF: evolutionary discretization for accurate and interpretable depth-limited decision trees."""
from .candidates import build_candidate_bounds
from .feature_selection import onehot_and_group_importance, select_features
from .ga import Chromosome, Gene, discrete_df, evolve
from .pipeline import run_fold

__all__ = ["build_candidate_bounds", "onehot_and_group_importance", "select_features",
           "Chromosome", "Gene", "discrete_df", "evolve", "run_fold"]
__version__ = "1.0.0"
