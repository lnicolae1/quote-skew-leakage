# fundamental_value.py: hidden fundamental value V_t (seen only by informed traders)
# - GBM, no drift: V_{t+dt} = V_t * exp(sigma * sqrt(dt_years) * Z), Z ~ N(0, 1)
# - sigma annualized (Phase 3: 0.3721); dt in seconds, dt_years = dt / SECONDS_PER_YEAR
# - same annualization as compute_volatility.py; test_fundamental_value.py checks the round trip
# - constant volatility: no clustering, thinner tails than real BTC

import math
import random

# same convention as compute_volatility.py
SECONDS_PER_YEAR = 365 * 24 * 3600


def generate_value_path(sigma, start_price, dt_seconds, n_steps, seed):
    """Geometric random walk [V_0, ..., V_n_steps]; sigma annualized, dt in seconds, private seeded RNG."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative, got %r" % (sigma,))
    if start_price <= 0:
        raise ValueError("start_price must be positive, got %r" % (start_price,))
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive, got %r" % (dt_seconds,))
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative, got %r" % (n_steps,))

    dt_years = dt_seconds / SECONDS_PER_YEAR
    sigma_per_step = sigma * math.sqrt(dt_years)

    rng = random.Random(seed)

    path = [start_price]
    v = start_price
    for _ in range(n_steps):
        z = rng.gauss(0.0, 1.0)
        v = v * math.exp(sigma_per_step * z)
        path.append(v)
    return path


if __name__ == "__main__":
    print("fundamental_value.py is a library module; run test_fundamental_value.py")
