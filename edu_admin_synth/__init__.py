"""
edu_admin_synth
===============

Research-grade synthetic multimodal dataset generator for the study:

    "Multimodal Representation Learning for Educational Administrative
     Behavior Analysis: A Transformer-Based Fusion Framework"

The package simulates educational administrators over time using hidden
latent behavioural profiles, temporal drift, seasonal academic effects and
stochastic events. From those latent states it derives three correlated
modalities (text, administrative logs, teacher surveys), output labels,
validation reports and rich metadata.

Pipeline
--------
Config -> Profiles -> Temporal Simulator -> {Text, Logs, Survey} ->
Labels -> Fusion -> Missingness/Noise -> Validation -> Metadata -> Export

The public entry point is :func:`edu_admin_synth.pipeline.run_pipeline`.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
