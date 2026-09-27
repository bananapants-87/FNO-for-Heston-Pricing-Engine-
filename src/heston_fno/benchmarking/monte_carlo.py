"""Monte Carlo pricing for the Heston model."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MonteCarloResult:
    """Monte Carlo price estimate and uncertainty."""

    price: torch.Tensor
    standard_error: torch.Tensor
    ci95_low: torch.Tensor
    ci95_high: torch.Tensor

    def to_dict(self) -> dict[str, float]:
        """Return scalar values for a single-point result."""
        return {
            "price": float(self.price.item()),
            "standard_error": float(self.standard_error.item()),
            "ci95_low": float(self.ci95_low.item()),
            "ci95_high": float(self.ci95_high.item()),
        }


def price_heston_call_mc_curve(
    params: torch.Tensor,
    s_grid: torch.Tensor,
    v0: float,
    strike: float = 100.0,
    r: float = 0.0,
    maturity: float = 1.0,
    n_paths: int = 5000,
    n_steps: int = 100,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> MonteCarloResult:
    """Price a European call for many initial S values at one initial variance.

    Uses full-truncation Euler discretisation for the Heston variance process
    and antithetic variates to reduce Monte Carlo variance.

    Parameters
    ----------
    params:
        Heston parameters [kappa, theta, sigma, rho, v0_parameter].
        The fifth value is not used here because ``v0`` is supplied explicitly
        from the surface's variance coordinate.
    s_grid:
        Initial spot prices at which to calculate option values.
    v0:
        Initial variance for this price curve.
    strike:
        European call strike.
    r:
        Risk-free interest rate.
    maturity:
        Time to maturity.
    n_paths:
        Total number of Monte Carlo paths. Must be even.
    n_steps:
        Number of Euler time steps.
    seed:
        Random seed.
    device:
        ``cpu`` or ``cuda``.

    Returns
    -------
    MonteCarloResult
        Price, standard error and approximate 95% confidence interval
        for every S value in ``s_grid``.
    """

    if n_paths < 2 or n_paths % 2 != 0:
        raise ValueError("n_paths must be an even integer >= 2")

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    device = torch.device(device)

    params = torch.as_tensor(params, dtype=torch.float32, device=device).reshape(-1)
    s_grid = torch.as_tensor(s_grid, dtype=torch.float32, device=device).reshape(-1)

    if params.numel() < 5:
        raise ValueError("params must contain [kappa, theta, sigma, rho, v0]")

    kappa, theta, sigma, rho, _ = params[:5]

    if not (-1.0 <= float(rho) <= 1.0):
        raise ValueError("rho must be between -1 and 1")

    dt = maturity / n_steps
    sqrt_dt = dt ** 0.5

    n_pairs = n_paths // 2

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # Two antithetic sets of paths.
    variance_plus = torch.full(
        (n_pairs,),
        float(v0),
        dtype=torch.float32,
        device=device,
    )
    variance_minus = variance_plus.clone()

    # Shape: [number_of_S_values, number_of_paths_in_pair]
    spot_plus = s_grid[:, None].expand(-1, n_pairs).clone()
    spot_minus = spot_plus.clone()

    correlation_scale = torch.sqrt(
        torch.clamp(1.0 - rho ** 2, min=0.0)
    )

    for _ in range(n_steps):
        z1 = torch.randn(
            n_pairs,
            generator=generator,
            device=device,
        )
        z2_independent = torch.randn(
            n_pairs,
            generator=generator,
            device=device,
        )

        # Correlated Brownian increment for the variance process.
        z2 = rho * z1 + correlation_scale * z2_independent

        # --------------------------------------------------------------
        # Antithetic path +
        # --------------------------------------------------------------
        variance_plus_pos = torch.clamp(variance_plus, min=0.0)

        spot_plus = spot_plus * torch.exp(
            (r - 0.5 * variance_plus_pos) * dt
            + torch.sqrt(variance_plus_pos) * sqrt_dt * z1
        )

        variance_plus = (
            variance_plus
            + kappa * (theta - variance_plus_pos) * dt
            + sigma * torch.sqrt(variance_plus_pos) * sqrt_dt * z2
        )

        variance_plus = torch.clamp(variance_plus, min=0.0)

        # --------------------------------------------------------------
        # Antithetic path -
        # --------------------------------------------------------------
        variance_minus_pos = torch.clamp(variance_minus, min=0.0)

        spot_minus = spot_minus * torch.exp(
            (r - 0.5 * variance_minus_pos) * dt
            + torch.sqrt(variance_minus_pos) * sqrt_dt * (-z1)
        )

        variance_minus = (
            variance_minus
            + kappa * (theta - variance_minus_pos) * dt
            + sigma * torch.sqrt(variance_minus_pos) * sqrt_dt * (-z2)
        )

        variance_minus = torch.clamp(variance_minus, min=0.0)

    # Discounted European call payoff.
    payoff_plus = torch.exp(
        torch.tensor(-r * maturity, device=device)
    ) * torch.clamp(spot_plus - strike, min=0.0)

    payoff_minus = torch.exp(
        torch.tensor(-r * maturity, device=device)
    ) * torch.clamp(spot_minus - strike, min=0.0)

    # One observation per antithetic pair.
    pair_payoffs = 0.5 * (payoff_plus + payoff_minus)

    price = pair_payoffs.mean(dim=1)

    # Standard error of the antithetic-pair estimator.
    if n_pairs > 1:
        standard_error = (
            pair_payoffs.std(dim=1, unbiased=True)
            / torch.sqrt(
                torch.tensor(
                    float(n_pairs),
                    dtype=torch.float32,
                    device=device,
                )
            )
        )
    else:
        standard_error = torch.zeros_like(price)

    ci95_half_width = 1.96 * standard_error

    return MonteCarloResult(
        price=price.detach().cpu(),
        standard_error=standard_error.detach().cpu(),
        ci95_low=(price - ci95_half_width).detach().cpu(),
        ci95_high=(price + ci95_half_width).detach().cpu(),
    )


def price_heston_call_mc(
    params: torch.Tensor,
    s0: float,
    v0: float,
    strike: float = 100.0,
    r: float = 0.0,
    maturity: float = 1.0,
    n_paths: int = 5000,
    n_steps: int = 100,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> MonteCarloResult:
    """Price one European call using Heston Monte Carlo."""

    result = price_heston_call_mc_curve(
        params=params,
        s_grid=torch.tensor([s0], dtype=torch.float32),
        v0=v0,
        strike=strike,
        r=r,
        maturity=maturity,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        device=device,
    )

    return MonteCarloResult(
        price=result.price[0],
        standard_error=result.standard_error[0],
        ci95_low=result.ci95_low[0],
        ci95_high=result.ci95_high[0],
    )