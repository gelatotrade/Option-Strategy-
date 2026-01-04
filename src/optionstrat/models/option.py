"""
Option and OptionLeg Classes

This module provides the core Option and OptionLeg classes for
representing and manipulating options positions.

An Option represents a single option contract with all its properties:
- Type (call/put)
- Strike price
- Expiration
- Position (long/short)
- Greeks

An OptionLeg wraps an Option with quantity information for use
in multi-leg strategies.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Union, Tuple
from enum import Enum
import numpy as np

from optionstrat.models.pricing import BlackScholes
from optionstrat.models.greeks import Greeks, GreeksCalculator


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class PositionType(Enum):
    """Position type enumeration."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Option:
    """
    Represents a single option contract.

    This class encapsulates all the properties of an option and provides
    methods for pricing, Greeks calculation, and payoff analysis.

    Attributes:
        option_type: 'call' or 'put'
        strike: Strike price
        expiry_days: Days to expiration
        underlying_price: Current price of underlying
        volatility: Implied volatility (annualized, e.g., 0.25 for 25%)
        risk_free_rate: Risk-free interest rate (annualized)
        dividend_yield: Continuous dividend yield
        premium: Option premium (if known, otherwise calculated)

    Example:
        call = Option(
            option_type='call',
            strike=100,
            expiry_days=30,
            underlying_price=100,
            volatility=0.25
        )
        print(f"Price: ${call.price:.2f}")
        print(f"Delta: {call.delta:.4f}")
    """

    option_type: str
    strike: float
    expiry_days: int
    underlying_price: float
    volatility: float
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    premium: Optional[float] = None
    symbol: Optional[str] = None

    def __post_init__(self):
        """Validate and initialize option."""
        self.option_type = self.option_type.lower()
        if self.option_type not in ['call', 'put']:
            raise ValueError(f"Invalid option type: {self.option_type}")

        if self.strike <= 0:
            raise ValueError("Strike price must be positive")

        if self.expiry_days <= 0:
            raise ValueError("Days to expiry must be positive")

        # Calculate price if not provided
        if self.premium is None:
            self.premium = self._calculate_price()

    @property
    def time_to_expiry(self) -> float:
        """Get time to expiry in years."""
        return self.expiry_days / 365.0

    @property
    def expiry_date(self) -> date:
        """Get expiration date."""
        return date.today() + timedelta(days=self.expiry_days)

    def _get_bs_model(self) -> BlackScholes:
        """Get Black-Scholes model instance."""
        return BlackScholes(
            S=self.underlying_price,
            K=self.strike,
            T=self.time_to_expiry,
            r=self.risk_free_rate,
            sigma=self.volatility,
            q=self.dividend_yield
        )

    def _calculate_price(self) -> float:
        """Calculate option price using Black-Scholes."""
        bs = self._get_bs_model()
        return bs.price(self.option_type)

    @property
    def price(self) -> float:
        """Get option price."""
        return self.premium if self.premium is not None else self._calculate_price()

    # ==================== GREEKS ====================

    @property
    def delta(self) -> float:
        """Get option delta."""
        return self._get_bs_model().delta(self.option_type)

    @property
    def gamma(self) -> float:
        """Get option gamma."""
        return self._get_bs_model().gamma()

    @property
    def theta(self) -> float:
        """Get option theta (per day)."""
        return self._get_bs_model().theta(self.option_type)

    @property
    def vega(self) -> float:
        """Get option vega (per 1% IV change)."""
        return self._get_bs_model().vega()

    @property
    def rho(self) -> float:
        """Get option rho (per 1% rate change)."""
        return self._get_bs_model().rho(self.option_type)

    @property
    def greeks(self) -> Greeks:
        """Get all Greeks as a Greeks object."""
        return GreeksCalculator.calculate_all(
            S=self.underlying_price,
            K=self.strike,
            T=self.time_to_expiry,
            r=self.risk_free_rate,
            sigma=self.volatility,
            option_type=self.option_type,
            q=self.dividend_yield,
            include_second_order=True
        )

    # ==================== PAYOFF ====================

    def payoff_at_expiry(self, price: float) -> float:
        """
        Calculate payoff at expiration for a given underlying price.

        Args:
            price: Underlying price at expiration

        Returns:
            Payoff value (not including premium)
        """
        if self.option_type == 'call':
            return max(price - self.strike, 0)
        else:
            return max(self.strike - price, 0)

    def profit_at_expiry(self, price: float) -> float:
        """
        Calculate profit/loss at expiration for a given underlying price.

        Args:
            price: Underlying price at expiration

        Returns:
            Profit/loss value (including premium paid)
        """
        return self.payoff_at_expiry(price) - self.price

    def payoff_table(
        self,
        price_range: Optional[Tuple[float, float]] = None,
        price_points: int = 50
    ) -> Dict[str, List[float]]:
        """
        Generate payoff table for range of prices.

        Args:
            price_range: Tuple of (min_price, max_price), defaults to ±30% of strike
            price_points: Number of price points

        Returns:
            Dictionary with 'prices', 'payoffs', and 'profits' lists
        """
        if price_range is None:
            low = self.strike * 0.7
            high = self.strike * 1.3
        else:
            low, high = price_range

        prices = np.linspace(low, high, price_points)
        payoffs = [self.payoff_at_expiry(p) for p in prices]
        profits = [self.profit_at_expiry(p) for p in prices]

        return {
            'prices': prices.tolist(),
            'payoffs': payoffs,
            'profits': profits
        }

    # ==================== PROBABILITY ====================

    @property
    def probability_itm(self) -> float:
        """Get probability of expiring in-the-money."""
        return self._get_bs_model().probability_itm(self.option_type)

    @property
    def probability_profit(self) -> float:
        """Get probability of profit (price > breakeven)."""
        be = self.breakeven
        bs = self._get_bs_model()
        if self.option_type == 'call':
            # P(S > breakeven)
            d2_be = (np.log(self.underlying_price / be) +
                    (self.risk_free_rate - self.dividend_yield -
                     0.5 * self.volatility**2) * self.time_to_expiry) / \
                   (self.volatility * np.sqrt(self.time_to_expiry))
            from scipy.stats import norm
            return norm.cdf(d2_be)
        else:
            # P(S < breakeven)
            d2_be = (np.log(self.underlying_price / be) +
                    (self.risk_free_rate - self.dividend_yield -
                     0.5 * self.volatility**2) * self.time_to_expiry) / \
                   (self.volatility * np.sqrt(self.time_to_expiry))
            from scipy.stats import norm
            return norm.cdf(-d2_be)

    # ==================== VALUE ANALYSIS ====================

    @property
    def intrinsic_value(self) -> float:
        """Get intrinsic value."""
        return self._get_bs_model().intrinsic_value(self.option_type)

    @property
    def extrinsic_value(self) -> float:
        """Get extrinsic (time) value."""
        return self.price - self.intrinsic_value

    @property
    def moneyness(self) -> str:
        """Get moneyness (ITM, ATM, OTM)."""
        ratio = self.underlying_price / self.strike
        if abs(ratio - 1) < 0.02:
            return 'ATM'
        if self.option_type == 'call':
            return 'ITM' if ratio > 1 else 'OTM'
        else:
            return 'ITM' if ratio < 1 else 'OTM'

    @property
    def breakeven(self) -> float:
        """Get breakeven price at expiration."""
        if self.option_type == 'call':
            return self.strike + self.price
        else:
            return self.strike - self.price

    @property
    def max_profit(self) -> float:
        """Get maximum profit (unlimited for calls)."""
        if self.option_type == 'call':
            return float('inf')
        else:
            return self.strike - self.price

    @property
    def max_loss(self) -> float:
        """Get maximum loss (premium paid)."""
        return self.price

    # ==================== UTILITY ====================

    def update_underlying(self, new_price: float) -> 'Option':
        """
        Create a new Option with updated underlying price.

        Args:
            new_price: New underlying price

        Returns:
            New Option instance
        """
        return Option(
            option_type=self.option_type,
            strike=self.strike,
            expiry_days=self.expiry_days,
            underlying_price=new_price,
            volatility=self.volatility,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            symbol=self.symbol
        )

    def with_new_expiry(self, new_days: int) -> 'Option':
        """
        Create a new Option with different expiry.

        Args:
            new_days: New days to expiration

        Returns:
            New Option instance
        """
        return Option(
            option_type=self.option_type,
            strike=self.strike,
            expiry_days=new_days,
            underlying_price=self.underlying_price,
            volatility=self.volatility,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            symbol=self.symbol
        )

    def summary(self) -> str:
        """Get formatted summary string."""
        return f"""
╔══════════════════════════════════════════════════╗
║              OPTION SUMMARY                       ║
╠══════════════════════════════════════════════════╣
║  Type:           {self.option_type.upper():>6}                        ║
║  Strike:         ${self.strike:>10.2f}                   ║
║  Expiry:         {self.expiry_days:>6} days ({self.expiry_date})     ║
║  Underlying:     ${self.underlying_price:>10.2f}                   ║
║  IV:             {self.volatility*100:>10.1f}%                   ║
╠══════════════════════════════════════════════════╣
║  Price:          ${self.price:>10.2f}                   ║
║  Intrinsic:      ${self.intrinsic_value:>10.2f}                   ║
║  Extrinsic:      ${self.extrinsic_value:>10.2f}                   ║
║  Moneyness:      {self.moneyness:>10}                   ║
╠══════════════════════════════════════════════════╣
║  Delta:          {self.delta:>10.4f}                   ║
║  Gamma:          {self.gamma:>10.4f}                   ║
║  Theta:          {self.theta:>10.4f} /day              ║
║  Vega:           {self.vega:>10.4f} /1%               ║
║  Rho:            {self.rho:>10.4f} /1%               ║
╠══════════════════════════════════════════════════╣
║  Breakeven:      ${self.breakeven:>10.2f}                   ║
║  P(ITM):         {self.probability_itm*100:>10.1f}%                   ║
║  Max Loss:       ${self.max_loss:>10.2f}                   ║
╚══════════════════════════════════════════════════╝
"""

    def to_dict(self) -> dict:
        """Convert option to dictionary."""
        return {
            'option_type': self.option_type,
            'strike': self.strike,
            'expiry_days': self.expiry_days,
            'underlying_price': self.underlying_price,
            'volatility': self.volatility,
            'risk_free_rate': self.risk_free_rate,
            'dividend_yield': self.dividend_yield,
            'premium': self.premium,
            'symbol': self.symbol,
            'price': self.price,
            'greeks': self.greeks.to_dict(),
            'breakeven': self.breakeven,
            'probability_itm': self.probability_itm,
            'moneyness': self.moneyness,
        }


@dataclass
class OptionLeg:
    """
    Represents a leg in a multi-leg options strategy.

    An OptionLeg wraps an Option with position information:
    - Quantity (positive for long, negative for short)
    - Entry price (premium paid/received)

    Attributes:
        option: The underlying Option
        quantity: Number of contracts (positive=long, negative=short)
        entry_price: Price at entry (optional, defaults to current price)

    Example:
        # Long 2 calls
        long_call = OptionLeg(call_option, quantity=2)

        # Short 1 put
        short_put = OptionLeg(put_option, quantity=-1)
    """

    option: Option
    quantity: int = 1
    entry_price: Optional[float] = None

    def __post_init__(self):
        """Initialize entry price if not provided."""
        if self.entry_price is None:
            self.entry_price = self.option.price

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def position_type(self) -> str:
        """Get position type string."""
        return "long" if self.is_long else "short"

    @property
    def cost(self) -> float:
        """
        Get cost of the position.

        Returns:
            Positive for debit (long), negative for credit (short)
        """
        return self.entry_price * abs(self.quantity) * 100 * (1 if self.is_long else -1)

    @property
    def current_value(self) -> float:
        """Get current market value of the position."""
        return self.option.price * abs(self.quantity) * 100 * (1 if self.is_long else -1)

    @property
    def pnl(self) -> float:
        """Get unrealized P&L."""
        return self.current_value - self.cost

    @property
    def greeks(self) -> Greeks:
        """Get position-adjusted Greeks."""
        base_greeks = self.option.greeks
        return base_greeks * self.quantity

    def payoff_at_expiry(self, price: float) -> float:
        """
        Calculate payoff at expiration.

        Args:
            price: Underlying price at expiration

        Returns:
            Total payoff for the leg
        """
        per_contract = self.option.payoff_at_expiry(price)
        if self.is_long:
            return per_contract * abs(self.quantity) * 100
        else:
            return -per_contract * abs(self.quantity) * 100

    def profit_at_expiry(self, price: float) -> float:
        """
        Calculate profit/loss at expiration.

        Args:
            price: Underlying price at expiration

        Returns:
            Total profit/loss for the leg
        """
        payoff = self.payoff_at_expiry(price)
        return payoff - self.cost

    def summary(self) -> str:
        """Get formatted summary string."""
        direction = "LONG" if self.is_long else "SHORT"
        return f"{direction} {abs(self.quantity)}x {self.option.option_type.upper()} @ ${self.option.strike:.2f}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'option': self.option.to_dict(),
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'position_type': self.position_type,
            'cost': self.cost,
            'greeks': self.greeks.to_dict(),
        }


def create_option_from_market_data(
    symbol: str,
    option_type: str,
    strike: float,
    expiry_date: Union[str, date],
    market_price: float,
    underlying_price: float,
    risk_free_rate: float = 0.05
) -> Option:
    """
    Create an Option from market data, calculating implied volatility.

    Args:
        symbol: Underlying symbol
        option_type: 'call' or 'put'
        strike: Strike price
        expiry_date: Expiration date (string 'YYYY-MM-DD' or date object)
        market_price: Current market price of the option
        underlying_price: Current underlying price
        risk_free_rate: Risk-free rate

    Returns:
        Option with implied volatility calculated from market price
    """
    if isinstance(expiry_date, str):
        expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d').date()

    days_to_expiry = (expiry_date - date.today()).days

    # Calculate implied volatility
    iv = BlackScholes.implied_volatility(
        market_price=market_price,
        S=underlying_price,
        K=strike,
        T=days_to_expiry / 365.0,
        r=risk_free_rate,
        option_type=option_type
    )

    return Option(
        option_type=option_type,
        strike=strike,
        expiry_days=days_to_expiry,
        underlying_price=underlying_price,
        volatility=iv or 0.25,  # Default to 25% if IV calculation fails
        risk_free_rate=risk_free_rate,
        premium=market_price,
        symbol=symbol
    )
