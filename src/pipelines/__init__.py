"""Alpha evaluation pipelines, one per payoff shape.

cross_sectional (rank L/S), time_series (directional trend), convexity
(options/vol & trend-with-stops). They share contracts but stay separate by
design. Old top-level names (alpha_pipeline, ts_pipeline, convexity_pipeline)
remain importable via re-export shims. See docs/RESEARCH_PIPELINE_REORGANIZATION.md.
"""
