"""
Strategy Class for Multi-Leg Options Strategies

This module provides the Strategy class for building and analyzing
complex multi-leg options strategies like iron condors, butterflies,
straddles, and custom combinations.

A Strategy combines multiple OptionLegs and provides:
- Combined P&L calculation
- Aggregated Greeks
- Risk metrics (max profit, max loss, breakevens)
- Probability of profit
- Visualization support
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from optionstrat.models.option import Option, OptionLeg
from optionstrat.models.greeks import Greeks, aggregate_greeks


@dataclass
class StrategyMetrics:
    """Container for strategy risk/reward metrics."""
    max_profit: float
    max_loss: float
    breakevens: List[float]
    risk_reward_ratio: float
    probability_of_profit: float
    net_debit_credit: float  # Positive = debit, Negative = credit
    return_on_risk: float  # max_profit / max_loss


@dataclass
class Strategy:
    """
    Represents a multi-leg options strategy.

    A Strategy is a collection of OptionLegs that together form a
    complete trading strategy. Common strategies include:
    - Vertical spreads (bull call, bear put, etc.)
    - Iron condors and iron butterflies
    - Straddles and strangles
    - Butterflies and condors
    - Calendar and diagonal spreads

    Attributes:
        name: Strategy name
        legs: List of OptionLeg objects
        underlying_price: Current underlying price
        symbol: Underlying symbol (optional)

    Example:
        # Create an iron condor
        strategy = Strategy(
            name="Iron Condor",
            legs=[
                OptionLeg(put_buy, quantity=1),   # Long put (wing)
                OptionLeg(put_sell, quantity=-1),  # Short put
                OptionLeg(call_sell, quantity=-1), # Short call
                OptionLeg(call_buy, quantity=1),   # Long call (wing)
            ],
            underlying_price=100
        )

        print(strategy.summary())
        strategy.plot_payoff()
    """

    name: str
    legs: List[OptionLeg]
    underlying_price: float
    symbol: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        """Validate strategy."""
        if not self.legs:
            raise ValueError("Strategy must have at least one leg")

    @property
    def num_legs(self) -> int:
        """Get number of legs in strategy."""
        return len(self.legs)

    @property
    def total_contracts(self) -> int:
        """Get total number of contracts across all legs."""
        return sum(abs(leg.quantity) for leg in self.legs)

    # ==================== COST & VALUE ====================

    @property
    def net_premium(self) -> float:
        """
        Get net premium paid or received.

        Returns:
            Positive value = net debit (you pay)
            Negative value = net credit (you receive)
        """
        return sum(leg.cost for leg in self.legs)

    @property
    def is_debit(self) -> bool:
        """Check if strategy is a net debit."""
        return self.net_premium > 0

    @property
    def is_credit(self) -> bool:
        """Check if strategy is a net credit."""
        return self.net_premium < 0

    @property
    def current_value(self) -> float:
        """Get current market value of the strategy."""
        return sum(leg.current_value for leg in self.legs)

    @property
    def unrealized_pnl(self) -> float:
        """Get unrealized P&L."""
        return self.current_value - self.net_premium

    # ==================== GREEKS ====================

    @property
    def greeks(self) -> Greeks:
        """Get aggregated Greeks for the strategy."""
        return aggregate_greeks([leg.greeks for leg in self.legs])

    @property
    def delta(self) -> float:
        """Get net delta."""
        return self.greeks.delta

    @property
    def gamma(self) -> float:
        """Get net gamma."""
        return self.greeks.gamma

    @property
    def theta(self) -> float:
        """Get net theta."""
        return self.greeks.theta

    @property
    def vega(self) -> float:
        """Get net vega."""
        return self.greeks.vega

    # ==================== PAYOFF ANALYSIS ====================

    def payoff_at_expiry(self, price: float) -> float:
        """
        Calculate total payoff at expiration.

        Args:
            price: Underlying price at expiration

        Returns:
            Total payoff in dollars
        """
        return sum(leg.payoff_at_expiry(price) for leg in self.legs)

    def profit_at_expiry(self, price: float) -> float:
        """
        Calculate total profit/loss at expiration.

        Args:
            price: Underlying price at expiration

        Returns:
            Total P&L in dollars
        """
        return sum(leg.profit_at_expiry(price) for leg in self.legs)

    def profit_at_price_and_time(
        self,
        price: float,
        days_remaining: int
    ) -> float:
        """
        Estimate profit/loss at a given price and time before expiry.

        Uses Black-Scholes to estimate option values at future date.

        Args:
            price: Underlying price
            days_remaining: Days remaining until expiration

        Returns:
            Estimated P&L
        """
        total_value = 0

        for leg in self.legs:
            # Create option with new parameters
            new_option = Option(
                option_type=leg.option.option_type,
                strike=leg.option.strike,
                expiry_days=days_remaining,
                underlying_price=price,
                volatility=leg.option.volatility,
                risk_free_rate=leg.option.risk_free_rate,
                dividend_yield=leg.option.dividend_yield
            )

            # Calculate value
            value = new_option.price * abs(leg.quantity) * 100
            if leg.is_short:
                value = -value

            total_value += value

        return total_value - self.net_premium

    def payoff_table(
        self,
        price_range: Optional[Tuple[float, float]] = None,
        price_points: int = 100
    ) -> Dict[str, List[float]]:
        """
        Generate payoff table for range of prices.

        Args:
            price_range: (min_price, max_price), defaults to ±30% of current price
            price_points: Number of price points

        Returns:
            Dictionary with 'prices' and 'profits' lists
        """
        if price_range is None:
            # Use strikes to determine reasonable range
            strikes = [leg.option.strike for leg in self.legs]
            min_strike = min(strikes)
            max_strike = max(strikes)
            margin = (max_strike - min_strike) * 0.5 + self.underlying_price * 0.2
            low = min_strike - margin
            high = max_strike + margin
        else:
            low, high = price_range

        prices = np.linspace(low, high, price_points)
        profits = [self.profit_at_expiry(p) for p in prices]

        return {
            'prices': prices.tolist(),
            'profits': profits
        }

    def pnl_matrix(
        self,
        price_range: Optional[Tuple[float, float]] = None,
        price_points: int = 50,
        time_points: int = 10
    ) -> Dict:
        """
        Generate P&L matrix over price and time.

        Args:
            price_range: (min_price, max_price)
            price_points: Number of price points
            time_points: Number of time points

        Returns:
            Dictionary with 'prices', 'days', and 'pnl' matrix
        """
        if price_range is None:
            strikes = [leg.option.strike for leg in self.legs]
            margin = (max(strikes) - min(strikes)) * 0.5 + self.underlying_price * 0.2
            low = min(strikes) - margin
            high = max(strikes) + margin
        else:
            low, high = price_range

        prices = np.linspace(low, high, price_points)

        # Get days to expiry from first leg
        max_days = self.legs[0].option.expiry_days
        days = np.linspace(max_days, 1, time_points).astype(int)

        pnl_matrix = []
        for d in days:
            row = [self.profit_at_price_and_time(p, d) for p in prices]
            pnl_matrix.append(row)

        return {
            'prices': prices.tolist(),
            'days': days.tolist(),
            'pnl': pnl_matrix
        }

    # ==================== RISK METRICS ====================

    def calculate_max_profit(self, price_points: int = 1000) -> float:
        """Calculate maximum possible profit."""
        table = self.payoff_table(price_points=price_points)
        return max(table['profits'])

    def calculate_max_loss(self, price_points: int = 1000) -> float:
        """Calculate maximum possible loss."""
        table = self.payoff_table(price_points=price_points)
        return min(table['profits'])

    def find_breakevens(self, price_points: int = 1000) -> List[float]:
        """
        Find breakeven prices at expiration.

        Returns:
            List of breakeven prices
        """
        table = self.payoff_table(price_points=price_points)
        prices = table['prices']
        profits = table['profits']

        breakevens = []
        for i in range(1, len(profits)):
            # Check for sign change
            if profits[i-1] * profits[i] < 0:
                # Linear interpolation
                p1, p2 = prices[i-1], prices[i]
                pnl1, pnl2 = profits[i-1], profits[i]
                breakeven = p1 - pnl1 * (p2 - p1) / (pnl2 - pnl1)
                breakevens.append(breakeven)

        return breakevens

    @property
    def max_profit(self) -> float:
        """Get maximum profit."""
        return self.calculate_max_profit()

    @property
    def max_loss(self) -> float:
        """Get maximum loss (as a positive number representing the loss)."""
        return abs(self.calculate_max_loss())

    @property
    def breakevens(self) -> List[float]:
        """Get breakeven prices."""
        return self.find_breakevens()

    @property
    def risk_reward_ratio(self) -> float:
        """Get risk/reward ratio (max_loss / max_profit)."""
        mp = self.max_profit
        if mp <= 0:
            return float('inf')
        return self.max_loss / mp

    @property
    def return_on_risk(self) -> float:
        """Get return on risk (max_profit / max_loss)."""
        ml = self.max_loss
        if ml <= 0:
            return float('inf')
        return self.max_profit / ml

    def probability_of_profit(self, num_simulations: int = 10000) -> float:
        """
        Estimate probability of profit using Monte Carlo simulation.

        Uses geometric Brownian motion to simulate price paths.

        Args:
            num_simulations: Number of Monte Carlo simulations

        Returns:
            Probability of profit (0 to 1)
        """
        # Use first leg's parameters
        leg = self.legs[0]
        S = self.underlying_price
        T = leg.option.time_to_expiry
        r = leg.option.risk_free_rate
        sigma = leg.option.volatility

        # Generate random price paths
        np.random.seed(42)  # For reproducibility
        z = np.random.standard_normal(num_simulations)
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

        # Calculate profit for each simulation
        profits = np.array([self.profit_at_expiry(s) for s in ST])

        return np.mean(profits > 0)

    def get_metrics(self) -> StrategyMetrics:
        """Get all strategy metrics."""
        max_profit = self.max_profit
        max_loss = self.max_loss
        return StrategyMetrics(
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=self.breakevens,
            risk_reward_ratio=self.risk_reward_ratio,
            probability_of_profit=self.probability_of_profit(),
            net_debit_credit=self.net_premium,
            return_on_risk=max_profit / max_loss if max_loss > 0 else float('inf')
        )

    # ==================== STRATEGY INFO ====================

    def get_strategy_type(self) -> str:
        """
        Attempt to identify the strategy type based on legs.

        Returns:
            Strategy type name or 'Custom'
        """
        if self.name:
            return self.name

        num_legs = len(self.legs)

        if num_legs == 1:
            leg = self.legs[0]
            if leg.is_long:
                return f"Long {leg.option.option_type.title()}"
            else:
                return f"Short {leg.option.option_type.title()}"

        if num_legs == 2:
            types = [l.option.option_type for l in self.legs]
            positions = [l.position_type for l in self.legs]

            if types[0] == types[1]:
                # Vertical spread
                if 'long' in positions and 'short' in positions:
                    if types[0] == 'call':
                        return "Call Spread"
                    else:
                        return "Put Spread"
            else:
                # Straddle or strangle
                strikes = [l.option.strike for l in self.legs]
                if strikes[0] == strikes[1]:
                    return "Straddle"
                else:
                    return "Strangle"

        if num_legs == 4:
            types = [l.option.option_type for l in self.legs]
            if 'call' in types and 'put' in types:
                return "Iron Condor/Butterfly"

        return "Custom Strategy"

    def summary(self) -> str:
        """Get formatted summary string."""
        metrics = self.get_metrics()

        legs_str = "\n".join([f"  • {leg.summary()}" for leg in self.legs])

        be_str = ", ".join([f"${be:.2f}" for be in metrics.breakevens]) if metrics.breakevens else "N/A"

        return f"""
╔══════════════════════════════════════════════════════════════════╗
║  STRATEGY: {self.name.upper():^53} ║
╠══════════════════════════════════════════════════════════════════╣
║  Underlying: {self.symbol or 'N/A':>10} @ ${self.underlying_price:>10.2f}                    ║
╠══════════════════════════════════════════════════════════════════╣
║  LEGS:                                                           ║
{legs_str}
╠══════════════════════════════════════════════════════════════════╣
║  RISK/REWARD:                                                    ║
║    Net Premium:    {'DEBIT' if self.is_debit else 'CREDIT':>8} ${abs(self.net_premium):>10.2f}                ║
║    Max Profit:              ${metrics.max_profit:>10.2f}                ║
║    Max Loss:                ${metrics.max_loss:>10.2f}                ║
║    Breakevens:              {be_str:>20}                ║
║    P(Profit):               {metrics.probability_of_profit*100:>10.1f}%                ║
║    Return on Risk:          {metrics.return_on_risk*100:>10.1f}%                ║
╠══════════════════════════════════════════════════════════════════╣
║  GREEKS:                                                         ║
║    Delta:  {self.delta:>8.4f}    Gamma: {self.gamma:>8.4f}                     ║
║    Theta:  {self.theta:>8.4f}    Vega:  {self.vega:>8.4f}                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

    def to_dict(self) -> dict:
        """Convert strategy to dictionary."""
        metrics = self.get_metrics()
        return {
            'name': self.name,
            'symbol': self.symbol,
            'underlying_price': self.underlying_price,
            'legs': [leg.to_dict() for leg in self.legs],
            'net_premium': self.net_premium,
            'is_debit': self.is_debit,
            'max_profit': metrics.max_profit,
            'max_loss': metrics.max_loss,
            'breakevens': metrics.breakevens,
            'probability_of_profit': metrics.probability_of_profit,
            'greeks': self.greeks.to_dict(),
        }

    # ==================== MODIFICATION ====================

    def add_leg(self, leg: OptionLeg) -> 'Strategy':
        """
        Add a leg to the strategy.

        Args:
            leg: OptionLeg to add

        Returns:
            New Strategy with added leg
        """
        return Strategy(
            name=self.name,
            legs=self.legs + [leg],
            underlying_price=self.underlying_price,
            symbol=self.symbol,
            description=self.description
        )

    def remove_leg(self, index: int) -> 'Strategy':
        """
        Remove a leg from the strategy.

        Args:
            index: Index of leg to remove

        Returns:
            New Strategy with leg removed
        """
        new_legs = self.legs[:index] + self.legs[index+1:]
        return Strategy(
            name=self.name,
            legs=new_legs,
            underlying_price=self.underlying_price,
            symbol=self.symbol,
            description=self.description
        )

    def with_updated_price(self, new_price: float) -> 'Strategy':
        """
        Create new strategy with updated underlying price.

        Args:
            new_price: New underlying price

        Returns:
            New Strategy with updated options
        """
        new_legs = []
        for leg in self.legs:
            new_option = leg.option.update_underlying(new_price)
            new_legs.append(OptionLeg(new_option, leg.quantity, leg.entry_price))

        return Strategy(
            name=self.name,
            legs=new_legs,
            underlying_price=new_price,
            symbol=self.symbol,
            description=self.description
        )
