"""
Black-Scholes Option Pricing Model

This module implements the Black-Scholes-Merton model for pricing
European-style options and calculating the Greeks.

The Black-Scholes formula:
    C = S * N(d1) - K * e^(-rT) * N(d2)  (for calls)
    P = K * e^(-rT) * N(-d2) - S * N(-d1)  (for puts)

Where:
    d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
    d2 = d1 - σ√T

    S = Current stock price
    K = Strike price
    r = Risk-free interest rate
    T = Time to expiration (in years)
    σ = Volatility (annualized)
    N(x) = Cumulative standard normal distribution
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class BlackScholesResult:
    """Container for Black-Scholes calculation results."""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    d1: float
    d2: float


class BlackScholes:
    """
    Black-Scholes-Merton Option Pricing Model.

    This class provides methods for:
    - European option pricing (calls and puts)
    - Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
    - Implied volatility calculation
    - Probability calculations

    Example:
        bs = BlackScholes(
            S=100,      # Stock price
            K=100,      # Strike price
            T=30/365,   # 30 days to expiry
            r=0.05,     # 5% risk-free rate
            sigma=0.25  # 25% volatility
        )

        call_price = bs.call_price()
        put_price = bs.put_price()
        greeks = bs.calculate_all_greeks('call')
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ):
        """
        Initialize Black-Scholes model.

        Args:
            S: Current stock/underlying price
            K: Strike price
            T: Time to expiration in years (e.g., 30/365 for 30 days)
            r: Risk-free interest rate (annualized, e.g., 0.05 for 5%)
            sigma: Volatility (annualized, e.g., 0.25 for 25%)
            q: Continuous dividend yield (default 0)
        """
        self.S = S
        self.K = K
        self.T = max(T, 1e-10)  # Avoid division by zero
        self.r = r
        self.sigma = max(sigma, 1e-10)  # Avoid division by zero
        self.q = q

        # Pre-calculate d1 and d2
        self._d1, self._d2 = self._calculate_d1_d2()

    def _calculate_d1_d2(self) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes formula."""
        sqrt_T = np.sqrt(self.T)
        d1 = (
            np.log(self.S / self.K) +
            (self.r - self.q + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * sqrt_T)
        d2 = d1 - self.sigma * sqrt_T
        return d1, d2

    @property
    def d1(self) -> float:
        """Get d1 value."""
        return self._d1

    @property
    def d2(self) -> float:
        """Get d2 value."""
        return self._d2

    def call_price(self) -> float:
        """
        Calculate call option price using Black-Scholes formula.

        Returns:
            Call option price
        """
        discount = np.exp(-self.r * self.T)
        dividend_discount = np.exp(-self.q * self.T)

        price = (
            self.S * dividend_discount * norm.cdf(self._d1) -
            self.K * discount * norm.cdf(self._d2)
        )
        return max(price, 0.0)

    def put_price(self) -> float:
        """
        Calculate put option price using Black-Scholes formula.

        Returns:
            Put option price
        """
        discount = np.exp(-self.r * self.T)
        dividend_discount = np.exp(-self.q * self.T)

        price = (
            self.K * discount * norm.cdf(-self._d2) -
            self.S * dividend_discount * norm.cdf(-self._d1)
        )
        return max(price, 0.0)

    def price(self, option_type: str) -> float:
        """
        Calculate option price based on type.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        if option_type.lower() == 'call':
            return self.call_price()
        elif option_type.lower() == 'put':
            return self.put_price()
        else:
            raise ValueError(f"Invalid option type: {option_type}. Use 'call' or 'put'.")

    # ==================== GREEKS ====================

    def delta(self, option_type: str) -> float:
        """
        Calculate Delta - rate of change of option price with respect to underlying.

        Delta measures how much the option price changes for a $1 change in the underlying.
        - Call delta: 0 to 1 (positive, option gains when stock rises)
        - Put delta: -1 to 0 (negative, option gains when stock falls)

        Args:
            option_type: 'call' or 'put'

        Returns:
            Delta value
        """
        dividend_discount = np.exp(-self.q * self.T)

        if option_type.lower() == 'call':
            return dividend_discount * norm.cdf(self._d1)
        else:
            return dividend_discount * (norm.cdf(self._d1) - 1)

    def gamma(self) -> float:
        """
        Calculate Gamma - rate of change of Delta with respect to underlying.

        Gamma measures the acceleration of option price change. Higher gamma
        means delta changes more rapidly. Same for calls and puts.

        Returns:
            Gamma value
        """
        dividend_discount = np.exp(-self.q * self.T)
        sqrt_T = np.sqrt(self.T)

        return (
            dividend_discount * norm.pdf(self._d1) /
            (self.S * self.sigma * sqrt_T)
        )

    def theta(self, option_type: str) -> float:
        """
        Calculate Theta - rate of change of option price with respect to time.

        Theta measures time decay - how much value the option loses per day.
        Usually negative for long options (time decay hurts buyers).

        Args:
            option_type: 'call' or 'put'

        Returns:
            Theta value (per day, not per year)
        """
        sqrt_T = np.sqrt(self.T)
        discount = np.exp(-self.r * self.T)
        dividend_discount = np.exp(-self.q * self.T)

        # Common term for both calls and puts
        common = -(
            self.S * dividend_discount * norm.pdf(self._d1) * self.sigma /
            (2 * sqrt_T)
        )

        if option_type.lower() == 'call':
            theta = (
                common +
                self.q * self.S * dividend_discount * norm.cdf(self._d1) -
                self.r * self.K * discount * norm.cdf(self._d2)
            )
        else:
            theta = (
                common -
                self.q * self.S * dividend_discount * norm.cdf(-self._d1) +
                self.r * self.K * discount * norm.cdf(-self._d2)
            )

        # Convert to per-day theta (divide by 365)
        return theta / 365

    def vega(self) -> float:
        """
        Calculate Vega - rate of change of option price with respect to volatility.

        Vega measures how much the option price changes for a 1% change in IV.
        Same for calls and puts. Higher for ATM options.

        Returns:
            Vega value (per 1% change in volatility)
        """
        dividend_discount = np.exp(-self.q * self.T)
        sqrt_T = np.sqrt(self.T)

        # Vega per 1 point of volatility (0.01)
        return self.S * dividend_discount * norm.pdf(self._d1) * sqrt_T / 100

    def rho(self, option_type: str) -> float:
        """
        Calculate Rho - rate of change of option price with respect to interest rate.

        Rho measures how much the option price changes for a 1% change in
        risk-free interest rate.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Rho value (per 1% change in interest rate)
        """
        discount = np.exp(-self.r * self.T)

        if option_type.lower() == 'call':
            rho = self.K * self.T * discount * norm.cdf(self._d2) / 100
        else:
            rho = -self.K * self.T * discount * norm.cdf(-self._d2) / 100

        return rho

    def calculate_all_greeks(self, option_type: str) -> BlackScholesResult:
        """
        Calculate all Greeks and option price.

        Args:
            option_type: 'call' or 'put'

        Returns:
            BlackScholesResult with all values
        """
        return BlackScholesResult(
            price=self.price(option_type),
            delta=self.delta(option_type),
            gamma=self.gamma(),
            theta=self.theta(option_type),
            vega=self.vega(),
            rho=self.rho(option_type),
            d1=self._d1,
            d2=self._d2
        )

    # ==================== PROBABILITY CALCULATIONS ====================

    def probability_itm(self, option_type: str) -> float:
        """
        Calculate probability of option expiring in-the-money.

        This is the risk-neutral probability, not the real-world probability.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Probability (0 to 1)
        """
        if option_type.lower() == 'call':
            return norm.cdf(self._d2)
        else:
            return norm.cdf(-self._d2)

    def probability_touch(self, option_type: str) -> float:
        """
        Calculate probability of underlying touching the strike price.

        This is approximately twice the probability of expiring ITM.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Probability (0 to 1)
        """
        return min(2 * self.probability_itm(option_type), 1.0)

    def expected_value(self, option_type: str) -> float:
        """
        Calculate expected value at expiration.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Expected value
        """
        return self.price(option_type)

    # ==================== IMPLIED VOLATILITY ====================

    @classmethod
    def implied_volatility(
        cls,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str,
        q: float = 0.0,
        precision: float = 1e-5,
        max_iterations: int = 100
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method.

        Args:
            market_price: Observed market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free interest rate
            option_type: 'call' or 'put'
            q: Dividend yield
            precision: Desired precision for IV
            max_iterations: Maximum iterations for Newton-Raphson

        Returns:
            Implied volatility, or None if not found
        """
        # Initial guess using Brenner & Subrahmanyam approximation
        sigma = np.sqrt(2 * np.pi / T) * market_price / S
        sigma = max(min(sigma, 5.0), 0.01)  # Bound between 1% and 500%

        for _ in range(max_iterations):
            bs = cls(S, K, T, r, sigma, q)
            price = bs.price(option_type)
            vega = bs.vega() * 100  # Convert back to per 1.0 volatility

            if abs(vega) < 1e-10:
                break

            diff = market_price - price

            if abs(diff) < precision:
                return sigma

            sigma = sigma + diff / vega
            sigma = max(min(sigma, 5.0), 0.001)

        return sigma if abs(market_price - cls(S, K, T, r, sigma, q).price(option_type)) < precision * 10 else None

    # ==================== UTILITY METHODS ====================

    def intrinsic_value(self, option_type: str) -> float:
        """
        Calculate intrinsic value of the option.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Intrinsic value
        """
        if option_type.lower() == 'call':
            return max(self.S - self.K, 0)
        else:
            return max(self.K - self.S, 0)

    def extrinsic_value(self, option_type: str) -> float:
        """
        Calculate extrinsic (time) value of the option.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Extrinsic value
        """
        return self.price(option_type) - self.intrinsic_value(option_type)

    def moneyness(self) -> str:
        """
        Determine if option is ITM, ATM, or OTM.

        Returns:
            'ITM', 'ATM', or 'OTM'
        """
        ratio = self.S / self.K
        if abs(ratio - 1) < 0.02:  # Within 2% of strike
            return 'ATM'
        elif ratio > 1:
            return 'ITM'  # For calls (put would be OTM)
        else:
            return 'OTM'  # For calls (put would be ITM)

    def breakeven(self, option_type: str) -> float:
        """
        Calculate breakeven price at expiration.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Breakeven price
        """
        premium = self.price(option_type)
        if option_type.lower() == 'call':
            return self.K + premium
        else:
            return self.K - premium


def calculate_option_chain_greeks(
    S: float,
    strikes: list,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> dict:
    """
    Calculate Greeks for an entire option chain.

    Args:
        S: Current stock price
        strikes: List of strike prices
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility
        q: Dividend yield

    Returns:
        Dictionary with call and put data for each strike
    """
    chain = {'calls': [], 'puts': []}

    for K in strikes:
        bs = BlackScholes(S, K, T, r, sigma, q)

        chain['calls'].append({
            'strike': K,
            **vars(bs.calculate_all_greeks('call'))
        })

        chain['puts'].append({
            'strike': K,
            **vars(bs.calculate_all_greeks('put'))
        })

    return chain
