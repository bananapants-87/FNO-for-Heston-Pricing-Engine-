"""Hundsdorfer-Verwer (HV) ADI solver and benchmark helpers for the Heston PDE.

The HV scheme performs two implicit correction passes per time step:

    Y0  = U^n + dt * F(U^n)                          # explicit predictor
    Y1  = Y0  + theta*dt * (F1(Y1) - F1(U^n))        # S-sweep  (pass 1)
    Y2  = Y1  + theta*dt * (F2(Y2) - F2(U^n))        # v-sweep  (pass 1)
    Yt0 = Y0  + 0.5*dt   * (F(Y2)  - F(U^n))         # corrector predictor
    Yt1 = Yt0 + theta*dt * (F1(Yt1) - F1(Y2))        # S-sweep  (pass 2)
    Yt2 = Yt1 + theta*dt * (F2(Yt2) - F2(Y2))        # v-sweep  (pass 2)
    U^{n+1} = Yt2

This gives second-order convergence in time (vs first-order for Douglas).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Benchmark helpers (unchanged public API)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkResult:
	"""Timing summary for a reference ADI solver."""

	average_time: float
	throughput: float

	def to_dict(self) -> dict[str, float]:
		return {
			"average_time": float(self.average_time),
			"throughput": float(self.throughput),
		}


def time_adi_inference(
	adi_fn: Callable[[torch.Tensor], torch.Tensor | float],
	batch: torch.Tensor,
	warmup: int = 5,
	repeats: int = 20,
) -> float:
	"""Measure average wall-clock runtime for an ADI reference callable."""

	batch = batch.detach().cpu()
	for _ in range(warmup):
		_ = adi_fn(batch)

	start = time.perf_counter()
	for _ in range(repeats):
		_ = adi_fn(batch)
	elapsed = time.perf_counter() - start
	return elapsed / max(repeats, 1)


def benchmark_adi(
	adi_fn: Callable[[torch.Tensor], torch.Tensor | float],
	batch: torch.Tensor,
	warmup: int = 5,
	repeats: int = 20,
) -> BenchmarkResult:
	"""Return timing and throughput information for an ADI baseline."""

	average_time = time_adi_inference(adi_fn, batch, warmup=warmup, repeats=repeats)
	batch_size = int(batch.shape[0]) if batch.ndim > 0 else 1
	throughput = float(batch_size / average_time) if average_time > 0 else float("inf")
	return BenchmarkResult(average_time=average_time, throughput=throughput)


# ---------------------------------------------------------------------------
# Internal: Thomas algorithm for batched tridiagonal solves
# ---------------------------------------------------------------------------

def _solve_tridiagonal(
	a: np.ndarray,
	b: np.ndarray,
	c: np.ndarray,
	d: np.ndarray,
) -> np.ndarray:
	"""Thomas algorithm for a batch of tridiagonal systems.

	Parameters
	----------
	a : sub-diagonal,   shape ``(m-1, k)``
	b : main diagonal,  shape ``(m,   k)``
	c : super-diagonal, shape ``(m-1, k)``
	d : right-hand side, shape ``(m,  k)``

	Returns
	-------
	np.ndarray of shape ``(m, k)`` — the solution vectors.

	Shapes may broadcast along *k* (e.g. ``(m, 1)`` coefficients with
	``(m, k)`` right-hand sides).
	"""

	m, k = d.shape
	cp = np.zeros((m - 1, k), dtype=d.dtype)
	dp = np.zeros((m, k), dtype=d.dtype)

	_safe = lambda x: np.where(np.abs(x) < 1e-14, np.copysign(1e-14, x + 1e-30), x)

	dp[0] = d[0] / _safe(b[0])
	cp[0] = c[0] / _safe(b[0])

	for i in range(1, m - 1):
		w = _safe(b[i] - a[i - 1] * cp[i - 1])
		cp[i] = c[i] / w
		dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / w

	w = _safe(b[m - 1] - a[m - 2] * cp[m - 2])
	dp[m - 1] = (d[m - 1] - a[m - 2] * dp[m - 2]) / w

	x = np.zeros_like(d)
	x[m - 1] = dp[m - 1]
	for i in range(m - 2, -1, -1):
		x[i] = dp[i] - cp[i] * x[i + 1]
	return x


# ---------------------------------------------------------------------------
# Internal: compute the three split PDE operators F0, F1, F2
# ---------------------------------------------------------------------------

def _compute_operators(
	u: np.ndarray,
	s_arr: np.ndarray,
	v_arr: np.ndarray,
	ds: float,
	dv: float,
	kappa: float,
	theta_h: float,
	sigma: float,
	rho: float,
	r: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return (F0, F1, F2) applied to grid function *u*.

	*  F0 — mixed-derivative (cross) term           (explicit only)
	*  F1 — S-direction diffusion + drift + discount (implicit in S-sweep)
	*  F2 — v-direction diffusion + drift + discount (implicit in v-sweep)
	"""

	s_int = s_arr[1:-1]
	v_int = v_arr[1:-1]

	# ---- finite-difference stencils ----
	d2u_ds2  = (u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :]) / (ds ** 2)
	du_ds    = (u[2:, :] - u[:-2, :]) / (2.0 * ds)

	d2u_dv2  = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / (dv ** 2)
	du_dv    = (u[:, 2:] - u[:, :-2]) / (2.0 * dv)

	d2u_dsdv = (u[2:, 2:] - u[2:, :-2] - u[:-2, 2:] + u[:-2, :-2]) / (4.0 * ds * dv)

	# F0: mixed derivative (defined on interior-S x interior-v)
	f0 = np.zeros_like(u)
	f0[1:-1, 1:-1] = rho * sigma * s_int[:, None] * v_int[None, :] * d2u_dsdv

	# F1: S-direction (defined on interior-S x all-v)
	f1 = np.zeros_like(u)
	f1[1:-1, :] = (
		0.5 * (s_int[:, None] ** 2) * v_arr[None, :] * d2u_ds2
		+ r * s_int[:, None] * du_ds
		- 0.5 * r * u[1:-1, :]
	)

	# F2: v-direction (defined on all-S x interior-v)
	f2 = np.zeros_like(u)
	f2[:, 1:-1] = (
		0.5 * (sigma ** 2) * v_int[None, :] * d2u_dv2
		+ kappa * (theta_h - v_int[None, :]) * du_dv
		- 0.5 * r * u[:, 1:-1]
	)

	return f0, f1, f2


# ---------------------------------------------------------------------------
# Internal: implicit sweeps
# ---------------------------------------------------------------------------

def _implicit_s_sweep(
	rhs: np.ndarray,
	bc_lo: np.ndarray,
	bc_hi: np.ndarray,
	s_int: np.ndarray,
	v_arr: np.ndarray,
	ds: float,
	r: float,
	theta_dt: float,
) -> np.ndarray:
	"""Solve ``(I - theta*dt*L1) Y = rhs`` along S for every v column."""

	m_s = len(s_int)

	# tridiag coefficients — shape (m_s, n_v) or broadcastable
	alpha = theta_dt * (0.5 * (s_int[:, None] ** 2) * v_arr[None, :] / (ds ** 2)
	                     - r * s_int[:, None] / (2.0 * ds))
	beta  = 1.0 + theta_dt * ((s_int[:, None] ** 2) * v_arr[None, :] / (ds ** 2)
	                           + 0.5 * r)
	gamma = theta_dt * (0.5 * (s_int[:, None] ** 2) * v_arr[None, :] / (ds ** 2)
	                     + r * s_int[:, None] / (2.0 * ds))

	d = rhs.copy()
	d[0, :]  += alpha[0, :]  * bc_lo
	d[-1, :] += gamma[-1, :] * bc_hi

	return _solve_tridiagonal(-alpha[1:, :], beta, -gamma[:-1, :], d)


def _implicit_v_sweep(
	rhs: np.ndarray,
	bc_lo: np.ndarray,
	bc_hi: np.ndarray,
	v_int: np.ndarray,
	dv: float,
	kappa: float,
	theta_h: float,
	sigma: float,
	r: float,
	theta_dt: float,
) -> np.ndarray:
	"""Solve ``(I - theta*dt*L2) Y = rhs`` along v for every interior-S row.

	*rhs* has shape ``(m_v, m_s)``  (transposed from the natural grid layout).
	"""

	alpha = theta_dt * (0.5 * (sigma ** 2) * v_int[:, None] / (dv ** 2)
	                     - kappa * (theta_h - v_int[:, None]) / (2.0 * dv))
	beta  = 1.0 + theta_dt * ((sigma ** 2) * v_int[:, None] / (dv ** 2)
	                           + 0.5 * r)
	gamma = theta_dt * (0.5 * (sigma ** 2) * v_int[:, None] / (dv ** 2)
	                     + kappa * (theta_h - v_int[:, None]) / (2.0 * dv))

	d = rhs.copy()
	d[0, :]  += alpha[0, :]  * bc_lo
	d[-1, :] += gamma[-1, :] * bc_hi

	return _solve_tridiagonal(-alpha[1:, :], beta, -gamma[:-1, :], d)


# ---------------------------------------------------------------------------
# Public: HV ADI solver
# ---------------------------------------------------------------------------

def solve_heston_adi(
	params: torch.Tensor | Sequence[float],
	s_grid: torch.Tensor | Sequence[float],
	v_grid: torch.Tensor | Sequence[float],
	strike: float = 100.0,
	r: float = 0.0,
	maturity: float = 1.0,
	n_steps: int = 100,
	theta_adi: float = 0.5,
) -> torch.Tensor:
	"""Solve the Heston PDE with the Hundsdorfer-Verwer (HV) ADI scheme.

	Parameters
	----------
	params : tensor or sequence
		``[kappa, theta, sigma, rho, v0]`` — Heston model parameters.
	s_grid, v_grid : tensor or sequence
		Uniform spatial grids for spot price *S* and variance *v*.
	strike : float
		Strike price *K* for the European call payoff.
	r : float
		Risk-free interest rate.
	maturity : float
		Time to maturity *T*.
	n_steps : int
		Number of time steps (higher → more accurate).
	theta_adi : float
		Implicitness parameter (default ``0.5``).

	Returns
	-------
	torch.Tensor
		Price surface of shape ``(n_s, n_v)``.
	"""

	# ---- convert inputs to numpy ----
	if isinstance(params, torch.Tensor):
		p = params.detach().cpu().reshape(-1).numpy().astype(float)
	else:
		p = np.asarray(params, dtype=float).reshape(-1)

	if isinstance(s_grid, torch.Tensor):
		s_arr = s_grid.detach().cpu().reshape(-1).numpy().astype(float)
	else:
		s_arr = np.asarray(s_grid, dtype=float).reshape(-1)

	if isinstance(v_grid, torch.Tensor):
		v_arr = v_grid.detach().cpu().reshape(-1).numpy().astype(float)
	else:
		v_arr = np.asarray(v_grid, dtype=float).reshape(-1)

	kappa, theta_h, sigma, rho, _v0 = p[:5]
	ds = s_arr[1] - s_arr[0]
	dv = v_arr[1] - v_arr[0]
	dt = maturity / max(n_steps, 1)
	theta_dt = theta_adi * dt

	s_int = s_arr[1:-1]
	v_int = v_arr[1:-1]

	# Boundary values
	bc_s_lo = 0.0
	bc_s_hi = s_arr[-1] - strike * np.exp(-r * maturity)

	# Initial condition: European call payoff
	u = np.maximum(s_arr[:, None] - strike, 0.0)

	def apply_bcs(w: np.ndarray) -> None:
		w[0, :]  = bc_s_lo
		w[-1, :] = bc_s_hi

	def ops(w: np.ndarray):
		return _compute_operators(w, s_arr, v_arr, ds, dv, kappa, theta_h, sigma, rho, r)

	def s_sweep(rhs_full, ref):
		rhs = rhs_full[1:-1, :].copy()
		return _implicit_s_sweep(rhs, ref[0, :], ref[-1, :], s_int, v_arr, ds, r, theta_dt)

	def v_sweep(rhs_full, ref):
		rhs = rhs_full[1:-1, 1:-1].T.copy()
		return _implicit_v_sweep(rhs, ref[1:-1, 0], ref[1:-1, -1],
		                         v_int, dv, kappa, theta_h, sigma, r, theta_dt).T

	# ---- HV time stepping ----
	for _ in range(n_steps):
		f0_n, f1_n, f2_n = ops(u)
		fn = f0_n + f1_n + f2_n

		# ---- Pass 1 ----
		# Y0 = U^n + dt * F(U^n)
		y0 = u + dt * fn
		apply_bcs(y0)

		# Y1: implicit S-correction
		y1 = y0.copy()
		y1[1:-1, :] = s_sweep(y0 - theta_dt * f1_n, y0)
		apply_bcs(y1)

		# Y2: implicit v-correction
		y2 = y1.copy()
		y2[1:-1, 1:-1] = v_sweep(y1 - theta_dt * f2_n, y1)
		apply_bcs(y2)

		# ---- Pass 2 (HV corrector) ----
		f0_2, f1_2, f2_2 = ops(y2)
		f_y2 = f0_2 + f1_2 + f2_2

		# Yt0 = Y0 + 0.5*dt*(F(Y2) - F(U^n))
		yt0 = y0 + 0.5 * dt * (f_y2 - fn)
		apply_bcs(yt0)

		# Yt1: implicit S-correction
		yt1 = yt0.copy()
		yt1[1:-1, :] = s_sweep(yt0 - theta_dt * f1_2, yt0)
		apply_bcs(yt1)

		# Yt2: implicit v-correction
		yt2 = yt1.copy()
		yt2[1:-1, 1:-1] = v_sweep(yt1 - theta_dt * f2_2, yt1)
		apply_bcs(yt2)

		u = yt2

	return torch.from_numpy(u).float()


# ---------------------------------------------------------------------------
# Public: convenience factory for benchmarking
# ---------------------------------------------------------------------------

def create_adi_benchmark_fn(
	s_grid: torch.Tensor,
	v_grid: torch.Tensor,
	strike: float = 100.0,
	r: float = 0.0,
	maturity: float = 1.0,
	n_steps: int = 100,
) -> Callable[[torch.Tensor], torch.Tensor]:
	"""Return a callable that runs the HV ADI solver on a batch of FNO inputs.

	The callable accepts a ``[B, C, H, W]`` tensor (the same format used by the
	FNO) and returns a ``[B, 1, H, W]`` tensor of reference price surfaces.
	"""

	def adi_fn(batch_x: torch.Tensor) -> torch.Tensor:
		batch_x = batch_x.detach().cpu()
		if batch_x.ndim == 3:
			batch_x = batch_x.unsqueeze(0)

		surfaces = []
		for b in range(batch_x.shape[0]):
			params = batch_x[b, :5, 0, 0]
			surf = solve_heston_adi(
				params=params,
				s_grid=s_grid,
				v_grid=v_grid,
				strike=strike,
				r=r,
				maturity=maturity,
				n_steps=n_steps,
			)
			surfaces.append(surf.unsqueeze(0))  # [1, H, W]

		return torch.stack(surfaces, dim=0)  # [B, 1, H, W]

	return adi_fn
