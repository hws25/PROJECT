"""Helpers for the EV trip energy consumption project (MMA3001).

The package predicts the energy (kWh) used on an electric-vehicle trip.

* :mod:`ev_energy.data` - load, check and clean the Chalmers trip dataset.
* :mod:`ev_energy.modelling` - vehicle-grouped splitting, metrics, timing,
  feature importance and residual analysis.

Everything is re-exported here, so notebooks can simply write
``from ev_energy import load_data, clean_data, split_by_vehicle``.
"""
from .data import load_data, quality_report, clean_data, PLAUSIBLE_RANGES, TARGET
from .modelling import (split_by_vehicle, evaluate, fixed_rate_predict, fit_and_time,
                        grouped_permutation_importance, error_by_group,
                        TRIP_FEATURES, ELECTRICAL_FEATURES)

__all__ = ["load_data", "quality_report", "clean_data", "PLAUSIBLE_RANGES", "TARGET",
           "split_by_vehicle", "evaluate", "fixed_rate_predict", "fit_and_time",
           "grouped_permutation_importance", "error_by_group",
           "TRIP_FEATURES", "ELECTRICAL_FEATURES"]
