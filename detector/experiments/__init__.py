"""Experiments module for insect detection research."""

from .experiment_runner import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentResults,
    ModelConfig,
    ModelFamily,
    MODEL_CONFIGS,
    create_experiment_suite,
)

__all__ = [
    'ExperimentRunner',
    'ExperimentConfig',
    'ExperimentResults',
    'ModelConfig',
    'ModelFamily',
    'MODEL_CONFIGS',
    'create_experiment_suite',
]
