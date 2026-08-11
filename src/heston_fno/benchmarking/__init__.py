"""Benchmarking utilities for Heston FNO."""

from .adi import BenchmarkResult as ADIBenchmarkResult, benchmark_adi, time_adi_inference
from .fno import BenchmarkResult as FNOBenchmarkResult, benchmark_fno
from .speedup import benchmark_speedup, time_fno_inference
