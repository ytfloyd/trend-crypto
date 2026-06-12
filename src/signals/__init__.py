"""Signal library: pure signal functions, organized by family.

A signal function takes market bars and returns target weights (or a directional
forecast); it does NO data loading or I/O. Registry entries point at these via a
dotted ``signal_fn`` path (e.g. ``signals.trend.ma_crossover``); the research
runner resolves and executes them. See registry/README.md and
docs/RESEARCH_PIPELINE_REORGANIZATION.md.
"""
