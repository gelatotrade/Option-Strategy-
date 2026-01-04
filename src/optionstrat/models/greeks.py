"""
Greeks: Option Sensitivity Measures

This module provides a comprehensive Greeks class for tracking and
analyzing option sensitivities.

The Greeks:
- Delta (Δ): Price sensitivity to underlying movement
- Gamma (Γ): Delta sensitivity to underlying movement
- Theta (Θ): Price sensitivity to time decay
- Vega (ν): Price sensitivity to volatility changes
- Rho (ρ): Price sensitivity to interest rate changes

Second-order Greeks:
- Vanna: Delta sensitivity to volatility
- Charm: Delta sensitivity to time
- Vomma: Vega sensitivity to volatility
- Speed: Gamma sensitivity to underlying
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
from scipy.stats import norm


@dataclass
class Greeks:
    """
    Container for option Greeks values.

    Attributes:
        delta: Rate of change of option price vs underlying price
        gamma: Rate of change of delta vs underlying price
        theta: Rate of change of option price vs time (per day)
        vega: Rate of change of option price vs volatility (per 1%)
        rho: Rate of change of option price vs interest rate (per 1%)
        vanna: Rate of change of delta vs volatility
        charm: Rate of change of delta vs time
        vomma: Rate of change of vega vs volatility
        speed: Rate of change of gamma vs underlying price
    """
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    # Second-order Greeks (optional)
    vanna: Optional[float] = None
    charm: Optional[float] = None
    vomma: Optional[float] = None
    speed: Optional[float] = None

    def __add__(self, other: 'Greeks') -> 'Greeks':
        """Add two Greeks objects (for multi-leg strategies)."""
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho,
            vanna=(self.vanna or 0) + (other.vanna or 0) if self.vanna is not None or other.vanna is not None else None,
            charm=(self.charm or 0) + (other.charm or 0) if self.charm is not None or other.charm is not None else None,
            vomma=(self.vomma or 0) + (other.vomma or 0) if self.vomma is not None or other.vomma is not None else None,
            speed=(self.speed or 0) + (other.speed or 0) if self.speed is not None or other.speed is not None else None,
        )

    def __mul__(self, scalar: float) -> 'Greeks':
        """Multiply Greeks by a scalar (for position sizing)."""
        return Greeks(
            delta=self.delta * scalar,
            gamma=self.gamma * scalar,
            theta=self.theta * scalar,
            vega=self.vega * scalar,
            rho=self.rho * scalar,
            vanna=self.vanna * scalar if self.vanna is not None else None,
            charm=self.charm * scalar if self.charm is not None else None,
            vomma=self.vomma * scalar if self.vomma is not None else None,
            speed=self.speed * scalar if self.speed is not None else None,
        )

    def __rmul__(self, scalar: float) -> 'Greeks':
        """Right multiply for scalar * Greeks."""
        return self.__mul__(scalar)

    def __neg__(self) -> 'Greeks':
        """Negate Greeks (for short positions)."""
        return self * -1

    def to_dict(self) -> Dict[str, float]:
        """Convert Greeks to dictionary."""
        result = {
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho,
        }
        if self.vanna is not None:
            result['vanna'] = self.vanna
        if self.charm is not None:
            result['charm'] = self.charm
        if self.vomma is not None:
            result['vomma'] = self.vomma
        if self.speed is not None:
            result['speed'] = self.speed
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Greeks':
        """Create Greeks from dictionary."""
        return cls(
            delta=data.get('delta', 0.0),
            gamma=data.get('gamma', 0.0),
            theta=data.get('theta', 0.0),
            vega=data.get('vega', 0.0),
            rho=data.get('rho', 0.0),
            vanna=data.get('vanna'),
            charm=data.get('charm'),
            vomma=data.get('vomma'),
            speed=data.get('speed'),
        )

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "═══════════════════════════════",
            "          GREEKS SUMMARY        ",
            "═══════════════════════════════",
            f"  Delta (Δ):  {self.delta:>10.4f}",
            f"  Gamma (Γ):  {self.gamma:>10.4f}",
            f"  Theta (Θ):  {self.theta:>10.4f} /day",
            f"  Vega  (ν):  {self.vega:>10.4f} /1%",
            f"  Rho   (ρ):  {self.rho:>10.4f} /1%",
        ]
        if self.vanna is not None:
            lines.append(f"  Vanna:      {self.vanna:>10.4f}")
        if self.charm is not None:
            lines.append(f"  Charm:      {self.charm:>10.4f}")
        lines.append("═══════════════════════════════")
        return "\n".join(lines)

    def dollar_values(self, contracts: int = 1, multiplier: int = 100) -> Dict[str, float]:
        """
        Calculate dollar-denominated Greeks.

        Args:
            contracts: Number of contracts
            multiplier: Contract multiplier (100 for standard options)

        Returns:
            Dictionary with dollar values
        """
        total = contracts * multiplier
        return {
            'dollar_delta': self.delta * total,
            'dollar_gamma': self.gamma * total,
            'dollar_theta': self.theta * total,
            'dollar_vega': self.vega * total,
            'dollar_rho': self.rho * total,
        }


class GreeksCalculator:
    """
    Advanced Greeks calculator with second-order Greeks support.

    This class extends the basic Greeks calculations to include:
    - Second-order Greeks (Vanna, Charm, Vomma, Speed)
    - Greeks evolution over time
    - Greeks sensitivity analysis
    """

    @staticmethod
    def calculate_all(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        q: float = 0.0,
        include_second_order: bool = False
    ) -> Greeks:
        """
        Calculate all Greeks for an option.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            q: Dividend yield
            include_second_order: Whether to calculate second-order Greeks

        Returns:
            Greeks object with all values
        """
        # Avoid division by zero
        T = max(T, 1e-10)
        sigma = max(sigma, 1e-10)

        # Calculate d1 and d2
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Pre-calculate common terms
        discount = np.exp(-r * T)
        div_discount = np.exp(-q * T)
        n_d1 = norm.cdf(d1)
        n_d2 = norm.cdf(d2)
        n_prime_d1 = norm.pdf(d1)

        is_call = option_type.lower() == 'call'

        # Delta
        if is_call:
            delta = div_discount * n_d1
        else:
            delta = div_discount * (n_d1 - 1)

        # Gamma (same for calls and puts)
        gamma = div_discount * n_prime_d1 / (S * sigma * sqrt_T)

        # Theta
        common_theta = -(S * div_discount * n_prime_d1 * sigma) / (2 * sqrt_T)
        if is_call:
            theta = (common_theta + q * S * div_discount * n_d1 -
                    r * K * discount * n_d2) / 365
        else:
            theta = (common_theta - q * S * div_discount * norm.cdf(-d1) +
                    r * K * discount * norm.cdf(-d2)) / 365

        # Vega (same for calls and puts, per 1% change)
        vega = S * div_discount * n_prime_d1 * sqrt_T / 100

        # Rho (per 1% change)
        if is_call:
            rho = K * T * discount * n_d2 / 100
        else:
            rho = -K * T * discount * norm.cdf(-d2) / 100

        greeks = Greeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho
        )

        # Calculate second-order Greeks if requested
        if include_second_order:
            # Vanna: dDelta/dSigma = dVega/dS
            greeks.vanna = vega / S * (1 - d1 / (sigma * sqrt_T))

            # Charm (delta decay): dDelta/dT
            greeks.charm = -div_discount * n_prime_d1 * (
                2 * (r - q) * T - d2 * sigma * sqrt_T
            ) / (2 * T * sigma * sqrt_T)

            # Vomma: dVega/dSigma
            greeks.vomma = vega * d1 * d2 / sigma

            # Speed: dGamma/dS
            greeks.speed = -gamma / S * (1 + d1 / (sigma * sqrt_T))

        return greeks

    @staticmethod
    def greeks_over_time(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        q: float = 0.0,
        time_points: int = 30
    ) -> Dict[str, List[float]]:
        """
        Calculate how Greeks change over time until expiration.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            q: Dividend yield
            time_points: Number of time points to calculate

        Returns:
            Dictionary with time array and Greeks arrays
        """
        times = np.linspace(T, 0.001, time_points)
        result = {
            'time': times.tolist(),
            'delta': [],
            'gamma': [],
            'theta': [],
            'vega': [],
        }

        for t in times:
            greeks = GreeksCalculator.calculate_all(S, K, t, r, sigma, option_type, q)
            result['delta'].append(greeks.delta)
            result['gamma'].append(greeks.gamma)
            result['theta'].append(greeks.theta)
            result['vega'].append(greeks.vega)

        return result

    @staticmethod
    def greeks_over_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        q: float = 0.0,
        price_range: float = 0.3,
        price_points: int = 50
    ) -> Dict[str, List[float]]:
        """
        Calculate how Greeks change over different underlying prices.

        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            q: Dividend yield
            price_range: Range as fraction of S (e.g., 0.3 = ±30%)
            price_points: Number of price points

        Returns:
            Dictionary with price array and Greeks arrays
        """
        low = S * (1 - price_range)
        high = S * (1 + price_range)
        prices = np.linspace(low, high, price_points)

        result = {
            'price': prices.tolist(),
            'delta': [],
            'gamma': [],
            'theta': [],
            'vega': [],
        }

        for p in prices:
            greeks = GreeksCalculator.calculate_all(p, K, T, r, sigma, option_type, q)
            result['delta'].append(greeks.delta)
            result['gamma'].append(greeks.gamma)
            result['theta'].append(greeks.theta)
            result['vega'].append(greeks.vega)

        return result

    @staticmethod
    def greeks_over_volatility(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        q: float = 0.0,
        vol_range: tuple = (0.05, 1.0),
        vol_points: int = 50
    ) -> Dict[str, List[float]]:
        """
        Calculate how Greeks change over different volatility levels.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Current volatility (used as reference)
            option_type: 'call' or 'put'
            q: Dividend yield
            vol_range: Tuple of (min_vol, max_vol)
            vol_points: Number of volatility points

        Returns:
            Dictionary with volatility array and Greeks arrays
        """
        vols = np.linspace(vol_range[0], vol_range[1], vol_points)

        result = {
            'volatility': vols.tolist(),
            'delta': [],
            'gamma': [],
            'theta': [],
            'vega': [],
        }

        for v in vols:
            greeks = GreeksCalculator.calculate_all(S, K, T, r, v, option_type, q)
            result['delta'].append(greeks.delta)
            result['gamma'].append(greeks.gamma)
            result['theta'].append(greeks.theta)
            result['vega'].append(greeks.vega)

        return result


def aggregate_greeks(greeks_list: List[Greeks]) -> Greeks:
    """
    Aggregate Greeks from multiple options/legs.

    Args:
        greeks_list: List of Greeks objects

    Returns:
        Combined Greeks object
    """
    if not greeks_list:
        return Greeks()

    result = greeks_list[0]
    for g in greeks_list[1:]:
        result = result + g

    return result
