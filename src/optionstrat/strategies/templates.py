"""
Strategy Templates - Pre-defined Strategy Configurations

This module provides pre-configured strategy templates that can be
easily applied to any underlying. Templates include optimal strike
selection based on current price and desired risk profile.

Templates are organized by:
- Market outlook (Bullish, Bearish, Neutral, Volatile)
- Risk profile (Conservative, Moderate, Aggressive)
- Time horizon (Short-term, Medium-term, Long-term)
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from optionstrat.strategies.builder import StrategyBuilder
from optionstrat.models.strategy import Strategy


class MarketOutlook(Enum):
    """Market direction expectation."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"


class RiskProfile(Enum):
    """Risk tolerance level."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class TimeHorizon(Enum):
    """Trading time horizon."""
    SHORT = "short"       # < 14 days
    MEDIUM = "medium"     # 14-45 days
    LONG = "long"         # > 45 days


@dataclass
class StrategyTemplate:
    """
    Template for creating a strategy.

    Attributes:
        name: Template name
        description: Detailed description
        outlook: Market outlook
        risk_profile: Risk tolerance
        time_horizon: Recommended time frame
        builder_method: Name of StrategyBuilder method
        strike_calculator: Function to calculate strikes from current price
        default_expiry_days: Default days to expiration
    """
    name: str
    description: str
    outlook: MarketOutlook
    risk_profile: RiskProfile
    time_horizon: TimeHorizon
    builder_method: str
    strike_calculator: Callable[[float], Dict[str, float]]
    default_expiry_days: int
    max_profit: str  # Description of max profit
    max_loss: str    # Description of max loss
    ideal_conditions: str


class StrategyTemplates:
    """
    Collection of pre-defined strategy templates.

    Provides easy-to-use templates for common trading scenarios.
    Each template includes optimal strike selection logic.

    Example:
        templates = StrategyTemplates("AAPL", 150, volatility=0.30)

        # Get all bullish strategies
        bullish = templates.get_by_outlook(MarketOutlook.BULLISH)

        # Create a specific strategy
        strategy = templates.create("aggressive_bull_call")

        # List all available templates
        for name, info in templates.list_all().items():
            print(f"{name}: {info['description']}")
    """

    def __init__(
        self,
        symbol: str,
        underlying_price: float,
        volatility: float = 0.25,
        risk_free_rate: float = 0.05
    ):
        """
        Initialize templates for a specific underlying.

        Args:
            symbol: Underlying symbol
            underlying_price: Current price
            volatility: Implied volatility
            risk_free_rate: Risk-free rate
        """
        self.symbol = symbol
        self.price = underlying_price
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.builder = StrategyBuilder(
            symbol, underlying_price, volatility, risk_free_rate
        )

        self._templates = self._define_templates()

    def _round_strike(self, price: float, increment: float = 1.0) -> float:
        """Round to nearest strike increment."""
        return round(price / increment) * increment

    def _define_templates(self) -> Dict[str, StrategyTemplate]:
        """Define all available templates."""
        p = self.price
        rs = self._round_strike

        return {
            # ==================== BULLISH ====================

            "long_call_atm": StrategyTemplate(
                name="Long Call (ATM)",
                description="Buy at-the-money call for moderate bullish outlook",
                outlook=MarketOutlook.BULLISH,
                risk_profile=RiskProfile.MODERATE,
                time_horizon=TimeHorizon.MEDIUM,
                builder_method="long_call",
                strike_calculator=lambda p: {"strike": rs(p)},
                default_expiry_days=30,
                max_profit="Unlimited",
                max_loss="Premium paid",
                ideal_conditions="Expecting moderate to large upward move"
            ),

            "long_call_otm": StrategyTemplate(
                name="Long Call (OTM)",
                description="Buy out-of-the-money call for aggressive bullish bet",
                outlook=MarketOutlook.BULLISH,
                risk_profile=RiskProfile.AGGRESSIVE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="long_call",
                strike_calculator=lambda p: {"strike": rs(p * 1.05)},
                default_expiry_days=21,
                max_profit="Unlimited",
                max_loss="Premium paid (lower than ATM)",
                ideal_conditions="Expecting large upward move, lower probability"
            ),

            "bull_call_spread_moderate": StrategyTemplate(
                name="Bull Call Spread (Moderate)",
                description="Bullish spread with defined risk and reward",
                outlook=MarketOutlook.BULLISH,
                risk_profile=RiskProfile.MODERATE,
                time_horizon=TimeHorizon.MEDIUM,
                builder_method="bull_call_spread",
                strike_calculator=lambda p: {
                    "buy_strike": rs(p),
                    "sell_strike": rs(p * 1.05)
                },
                default_expiry_days=30,
                max_profit="Strike width - debit paid",
                max_loss="Debit paid",
                ideal_conditions="Moderately bullish, want to reduce cost"
            ),

            "bull_call_spread_aggressive": StrategyTemplate(
                name="Bull Call Spread (Aggressive)",
                description="Wider spread for larger potential profit",
                outlook=MarketOutlook.BULLISH,
                risk_profile=RiskProfile.AGGRESSIVE,
                time_horizon=TimeHorizon.MEDIUM,
                builder_method="bull_call_spread",
                strike_calculator=lambda p: {
                    "buy_strike": rs(p * 0.98),
                    "sell_strike": rs(p * 1.10)
                },
                default_expiry_days=45,
                max_profit="Strike width - debit (larger)",
                max_loss="Higher debit",
                ideal_conditions="Strong bullish conviction"
            ),

            "bull_put_spread_income": StrategyTemplate(
                name="Bull Put Spread (Income)",
                description="Credit spread for bullish/neutral outlook",
                outlook=MarketOutlook.BULLISH,
                risk_profile=RiskProfile.CONSERVATIVE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="bull_put_spread",
                strike_calculator=lambda p: {
                    "sell_strike": rs(p * 0.95),
                    "buy_strike": rs(p * 0.90)
                },
                default_expiry_days=21,
                max_profit="Credit received",
                max_loss="Strike width - credit",
                ideal_conditions="Neutral to bullish, want income"
            ),

            "risk_reversal_bullish": StrategyTemplate(
                name="Risk Reversal (Bullish)",
                description="Synthetic long with OTM options",
                outlook=MarketOutlook.BULLISH,
                risk_profile=RiskProfile.AGGRESSIVE,
                time_horizon=TimeHorizon.MEDIUM,
                builder_method="risk_reversal",
                strike_calculator=lambda p: {
                    "put_strike": rs(p * 0.95),
                    "call_strike": rs(p * 1.05)
                },
                default_expiry_days=45,
                max_profit="Unlimited",
                max_loss="Significant if stock drops",
                ideal_conditions="Strong bullish conviction, want leverage"
            ),

            "pmcc_conservative": StrategyTemplate(
                name="Poor Man's Covered Call",
                description="Synthetic covered call with less capital",
                outlook=MarketOutlook.BULLISH,
                risk_profile=RiskProfile.MODERATE,
                time_horizon=TimeHorizon.LONG,
                builder_method="poor_mans_covered_call",
                strike_calculator=lambda p: {
                    "long_strike": rs(p * 0.80),
                    "short_strike": rs(p * 1.05),
                    "long_expiry_days": 180,
                    "short_expiry_days": 30
                },
                default_expiry_days=30,
                max_profit="Short strike - long strike - debit",
                max_loss="Debit paid",
                ideal_conditions="Moderately bullish, want income"
            ),

            "call_backspread": StrategyTemplate(
                name="Call Backspread",
                description="Unlimited upside with limited risk",
                outlook=MarketOutlook.BULLISH,
                risk_profile=RiskProfile.AGGRESSIVE,
                time_horizon=TimeHorizon.MEDIUM,
                builder_method="call_backspread",
                strike_calculator=lambda p: {
                    "sell_strike": rs(p),
                    "buy_strike": rs(p * 1.05),
                    "sell_quantity": 1,
                    "buy_quantity": 2
                },
                default_expiry_days=45,
                max_profit="Unlimited",
                max_loss="Limited to zone between strikes",
                ideal_conditions="Expecting large move up, volatile market"
            ),

            # ==================== BEARISH ====================

            "long_put_atm": StrategyTemplate(
                name="Long Put (ATM)",
                description="Buy at-the-money put for moderate bearish outlook",
                outlook=MarketOutlook.BEARISH,
                risk_profile=RiskProfile.MODERATE,
                time_horizon=TimeHorizon.MEDIUM,
                builder_method="long_put",
                strike_calculator=lambda p: {"strike": rs(p)},
                default_expiry_days=30,
                max_profit="Strike - Premium (if stock goes to $0)",
                max_loss="Premium paid",
                ideal_conditions="Expecting moderate to large downward move"
            ),

            "long_put_otm": StrategyTemplate(
                name="Long Put (OTM)",
                description="Cheap put for aggressive bearish bet or hedge",
                outlook=MarketOutlook.BEARISH,
                risk_profile=RiskProfile.AGGRESSIVE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="long_put",
                strike_calculator=lambda p: {"strike": rs(p * 0.95)},
                default_expiry_days=21,
                max_profit="Strike - Premium",
                max_loss="Lower premium",
                ideal_conditions="Expecting large drop or hedging portfolio"
            ),

            "bear_put_spread": StrategyTemplate(
                name="Bear Put Spread",
                description="Bearish spread with defined risk",
                outlook=MarketOutlook.BEARISH,
                risk_profile=RiskProfile.MODERATE,
                time_horizon=TimeHorizon.MEDIUM,
                builder_method="bear_put_spread",
                strike_calculator=lambda p: {
                    "buy_strike": rs(p),
                    "sell_strike": rs(p * 0.95)
                },
                default_expiry_days=30,
                max_profit="Strike width - debit",
                max_loss="Debit paid",
                ideal_conditions="Moderately bearish with risk control"
            ),

            "bear_call_spread_income": StrategyTemplate(
                name="Bear Call Spread (Income)",
                description="Credit spread for bearish/neutral outlook",
                outlook=MarketOutlook.BEARISH,
                risk_profile=RiskProfile.CONSERVATIVE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="bear_call_spread",
                strike_calculator=lambda p: {
                    "sell_strike": rs(p * 1.05),
                    "buy_strike": rs(p * 1.10)
                },
                default_expiry_days=21,
                max_profit="Credit received",
                max_loss="Strike width - credit",
                ideal_conditions="Neutral to bearish, want income"
            ),

            "put_backspread": StrategyTemplate(
                name="Put Backspread",
                description="Large profit potential on crash",
                outlook=MarketOutlook.BEARISH,
                risk_profile=RiskProfile.AGGRESSIVE,
                time_horizon=TimeHorizon.MEDIUM,
                builder_method="put_backspread",
                strike_calculator=lambda p: {
                    "sell_strike": rs(p),
                    "buy_strike": rs(p * 0.95),
                    "sell_quantity": 1,
                    "buy_quantity": 2
                },
                default_expiry_days=45,
                max_profit="Large on significant drop",
                max_loss="Limited to zone between strikes",
                ideal_conditions="Expecting crash or high volatility down"
            ),

            # ==================== NEUTRAL ====================

            "iron_condor_standard": StrategyTemplate(
                name="Iron Condor (Standard)",
                description="Classic neutral income strategy",
                outlook=MarketOutlook.NEUTRAL,
                risk_profile=RiskProfile.MODERATE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="iron_condor",
                strike_calculator=lambda p: {
                    "put_buy": rs(p * 0.90),
                    "put_sell": rs(p * 0.95),
                    "call_sell": rs(p * 1.05),
                    "call_buy": rs(p * 1.10)
                },
                default_expiry_days=30,
                max_profit="Credit received",
                max_loss="Wider spread width - credit",
                ideal_conditions="Low volatility, range-bound market"
            ),

            "iron_condor_wide": StrategyTemplate(
                name="Iron Condor (Wide)",
                description="Higher probability, lower reward",
                outlook=MarketOutlook.NEUTRAL,
                risk_profile=RiskProfile.CONSERVATIVE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="iron_condor",
                strike_calculator=lambda p: {
                    "put_buy": rs(p * 0.85),
                    "put_sell": rs(p * 0.90),
                    "call_sell": rs(p * 1.10),
                    "call_buy": rs(p * 1.15)
                },
                default_expiry_days=45,
                max_profit="Lower credit",
                max_loss="Spread width - credit",
                ideal_conditions="Want high probability, accept lower return"
            ),

            "iron_condor_narrow": StrategyTemplate(
                name="Iron Condor (Narrow)",
                description="Higher reward, lower probability",
                outlook=MarketOutlook.NEUTRAL,
                risk_profile=RiskProfile.AGGRESSIVE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="iron_condor",
                strike_calculator=lambda p: {
                    "put_buy": rs(p * 0.93),
                    "put_sell": rs(p * 0.97),
                    "call_sell": rs(p * 1.03),
                    "call_buy": rs(p * 1.07)
                },
                default_expiry_days=21,
                max_profit="Higher credit",
                max_loss="Spread width - credit",
                ideal_conditions="Confident market won't move much"
            ),

            "iron_butterfly": StrategyTemplate(
                name="Iron Butterfly",
                description="Max profit at exact price",
                outlook=MarketOutlook.NEUTRAL,
                risk_profile=RiskProfile.MODERATE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="iron_butterfly",
                strike_calculator=lambda p: {
                    "wing_put": rs(p * 0.90),
                    "body_strike": rs(p),
                    "wing_call": rs(p * 1.10)
                },
                default_expiry_days=30,
                max_profit="Higher than iron condor",
                max_loss="Wing width - credit",
                ideal_conditions="Expecting price to stay at current level"
            ),

            "short_strangle": StrategyTemplate(
                name="Short Strangle",
                description="Sell volatility premium",
                outlook=MarketOutlook.NEUTRAL,
                risk_profile=RiskProfile.AGGRESSIVE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="short_strangle",
                strike_calculator=lambda p: {
                    "put_strike": rs(p * 0.95),
                    "call_strike": rs(p * 1.05)
                },
                default_expiry_days=30,
                max_profit="Credit received",
                max_loss="Unlimited - WARNING!",
                ideal_conditions="High IV, expect contraction"
            ),

            "call_butterfly": StrategyTemplate(
                name="Long Call Butterfly",
                description="Low cost neutral bet",
                outlook=MarketOutlook.NEUTRAL,
                risk_profile=RiskProfile.CONSERVATIVE,
                time_horizon=TimeHorizon.MEDIUM,
                builder_method="long_call_butterfly",
                strike_calculator=lambda p: {
                    "lower_strike": rs(p * 0.95),
                    "middle_strike": rs(p),
                    "upper_strike": rs(p * 1.05)
                },
                default_expiry_days=30,
                max_profit="Middle - lower - debit",
                max_loss="Debit paid (small)",
                ideal_conditions="Expecting price to stay near current"
            ),

            "calendar_spread": StrategyTemplate(
                name="Calendar Spread",
                description="Profit from time decay",
                outlook=MarketOutlook.NEUTRAL,
                risk_profile=RiskProfile.MODERATE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="call_calendar_spread",
                strike_calculator=lambda p: {
                    "strike": rs(p),
                    "front_expiry_days": 14,
                    "back_expiry_days": 45
                },
                default_expiry_days=14,
                max_profit="Depends on volatility",
                max_loss="Debit paid",
                ideal_conditions="Expecting stock to stay near strike"
            ),

            "double_calendar": StrategyTemplate(
                name="Double Calendar",
                description="Wider profit zone calendar",
                outlook=MarketOutlook.NEUTRAL,
                risk_profile=RiskProfile.MODERATE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="double_calendar",
                strike_calculator=lambda p: {
                    "put_strike": rs(p * 0.95),
                    "call_strike": rs(p * 1.05),
                    "front_expiry_days": 14,
                    "back_expiry_days": 45
                },
                default_expiry_days=14,
                max_profit="At short strikes",
                max_loss="Debit paid",
                ideal_conditions="Range-bound with flexibility"
            ),

            # ==================== VOLATILE ====================

            "long_straddle": StrategyTemplate(
                name="Long Straddle",
                description="Profit from big move either way",
                outlook=MarketOutlook.VOLATILE,
                risk_profile=RiskProfile.MODERATE,
                time_horizon=TimeHorizon.MEDIUM,
                builder_method="long_straddle",
                strike_calculator=lambda p: {"strike": rs(p)},
                default_expiry_days=30,
                max_profit="Unlimited",
                max_loss="Premium paid",
                ideal_conditions="Expecting large move, unsure of direction"
            ),

            "long_strangle": StrategyTemplate(
                name="Long Strangle",
                description="Cheaper volatility bet",
                outlook=MarketOutlook.VOLATILE,
                risk_profile=RiskProfile.MODERATE,
                time_horizon=TimeHorizon.MEDIUM,
                builder_method="long_strangle",
                strike_calculator=lambda p: {
                    "put_strike": rs(p * 0.95),
                    "call_strike": rs(p * 1.05)
                },
                default_expiry_days=30,
                max_profit="Unlimited",
                max_loss="Premium paid (less than straddle)",
                ideal_conditions="Expecting big move, budget conscious"
            ),

            "reverse_iron_condor": StrategyTemplate(
                name="Reverse Iron Condor",
                description="Defined risk volatility play",
                outlook=MarketOutlook.VOLATILE,
                risk_profile=RiskProfile.MODERATE,
                time_horizon=TimeHorizon.MEDIUM,
                builder_method="reverse_iron_condor",
                strike_calculator=lambda p: {
                    "put_sell": rs(p * 0.90),
                    "put_buy": rs(p * 0.95),
                    "call_buy": rs(p * 1.05),
                    "call_sell": rs(p * 1.10)
                },
                default_expiry_days=30,
                max_profit="Spread width - debit",
                max_loss="Debit paid",
                ideal_conditions="Expecting breakout, want limited risk"
            ),

            "short_butterfly": StrategyTemplate(
                name="Short Call Butterfly",
                description="Credit for expecting move",
                outlook=MarketOutlook.VOLATILE,
                risk_profile=RiskProfile.CONSERVATIVE,
                time_horizon=TimeHorizon.MEDIUM,
                builder_method="short_call_butterfly",
                strike_calculator=lambda p: {
                    "lower_strike": rs(p * 0.95),
                    "middle_strike": rs(p),
                    "upper_strike": rs(p * 1.05)
                },
                default_expiry_days=30,
                max_profit="Credit received",
                max_loss="Middle - lower - credit",
                ideal_conditions="Expecting move away from middle"
            ),

            "long_guts": StrategyTemplate(
                name="Long Guts",
                description="ITM straddle variation",
                outlook=MarketOutlook.VOLATILE,
                risk_profile=RiskProfile.AGGRESSIVE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="long_guts",
                strike_calculator=lambda p: {
                    "call_strike": rs(p * 0.95),
                    "put_strike": rs(p * 1.05)
                },
                default_expiry_days=21,
                max_profit="Unlimited",
                max_loss="Premium paid (high)",
                ideal_conditions="Expecting extreme move"
            ),

            # ==================== INCOME ====================

            "covered_call_atm": StrategyTemplate(
                name="Covered Call (ATM)",
                description="Max premium, may be called away",
                outlook=MarketOutlook.NEUTRAL,
                risk_profile=RiskProfile.MODERATE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="covered_call",
                strike_calculator=lambda p: {"call_strike": rs(p)},
                default_expiry_days=30,
                max_profit="Premium received",
                max_loss="Stock price drop - premium",
                ideal_conditions="Willing to sell at current price"
            ),

            "covered_call_otm": StrategyTemplate(
                name="Covered Call (OTM)",
                description="Lower premium, more upside",
                outlook=MarketOutlook.BULLISH,
                risk_profile=RiskProfile.CONSERVATIVE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="covered_call",
                strike_calculator=lambda p: {"call_strike": rs(p * 1.05)},
                default_expiry_days=30,
                max_profit="Premium + upside to strike",
                max_loss="Stock price drop - premium",
                ideal_conditions="Want income plus some upside"
            ),

            "cash_secured_put": StrategyTemplate(
                name="Cash-Secured Put",
                description="Get paid to buy at lower price",
                outlook=MarketOutlook.BULLISH,
                risk_profile=RiskProfile.CONSERVATIVE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="short_put",
                strike_calculator=lambda p: {"strike": rs(p * 0.95)},
                default_expiry_days=30,
                max_profit="Premium received",
                max_loss="Strike - premium (if to $0)",
                ideal_conditions="Want to buy stock cheaper"
            ),

            "collar": StrategyTemplate(
                name="Collar",
                description="Protect stock, reduce cost",
                outlook=MarketOutlook.NEUTRAL,
                risk_profile=RiskProfile.CONSERVATIVE,
                time_horizon=TimeHorizon.MEDIUM,
                builder_method="collar",
                strike_calculator=lambda p: {
                    "put_strike": rs(p * 0.95),
                    "call_strike": rs(p * 1.05)
                },
                default_expiry_days=45,
                max_profit="Call strike - current price",
                max_loss="Current price - put strike",
                ideal_conditions="Own stock, want protection"
            ),

            "jade_lizard": StrategyTemplate(
                name="Jade Lizard",
                description="Short put + call credit spread",
                outlook=MarketOutlook.BULLISH,
                risk_profile=RiskProfile.MODERATE,
                time_horizon=TimeHorizon.SHORT,
                builder_method="jade_lizard",
                strike_calculator=lambda p: {
                    "put_strike": rs(p * 0.95),
                    "call_sell_strike": rs(p * 1.05),
                    "call_buy_strike": rs(p * 1.10)
                },
                default_expiry_days=30,
                max_profit="Credit received",
                max_loss="Put strike - credit (no upside risk if credit > spread)",
                ideal_conditions="Bullish, want no upside risk"
            ),
        }

    def list_all(self) -> Dict[str, dict]:
        """
        List all available templates with their info.

        Returns:
            Dictionary with template names as keys
        """
        result = {}
        for name, template in self._templates.items():
            result[name] = {
                "name": template.name,
                "description": template.description,
                "outlook": template.outlook.value,
                "risk_profile": template.risk_profile.value,
                "time_horizon": template.time_horizon.value,
                "max_profit": template.max_profit,
                "max_loss": template.max_loss,
                "ideal_conditions": template.ideal_conditions,
            }
        return result

    def get_by_outlook(self, outlook: MarketOutlook) -> Dict[str, StrategyTemplate]:
        """Get all templates for a given market outlook."""
        return {
            name: t for name, t in self._templates.items()
            if t.outlook == outlook
        }

    def get_by_risk(self, risk: RiskProfile) -> Dict[str, StrategyTemplate]:
        """Get all templates for a given risk profile."""
        return {
            name: t for name, t in self._templates.items()
            if t.risk_profile == risk
        }

    def get_by_time_horizon(self, horizon: TimeHorizon) -> Dict[str, StrategyTemplate]:
        """Get all templates for a given time horizon."""
        return {
            name: t for name, t in self._templates.items()
            if t.time_horizon == horizon
        }

    def create(
        self,
        template_name: str,
        expiry_days: Optional[int] = None,
        quantity: int = 1
    ) -> Strategy:
        """
        Create a strategy from a template.

        Args:
            template_name: Name of the template
            expiry_days: Override default expiry days
            quantity: Number of contracts

        Returns:
            Strategy object

        Example:
            strategy = templates.create("iron_condor_standard", expiry_days=45)
        """
        if template_name not in self._templates:
            raise ValueError(f"Unknown template: {template_name}")

        template = self._templates[template_name]

        # Calculate strikes based on current price
        kwargs = template.strike_calculator(self.price)
        kwargs["quantity"] = quantity

        # Use provided or default expiry
        if expiry_days:
            kwargs["expiry_days"] = expiry_days
        elif "expiry_days" not in kwargs:
            kwargs["expiry_days"] = template.default_expiry_days

        # Get the builder method and create strategy
        method = getattr(self.builder, template.builder_method)
        return method(**kwargs)

    def suggest(
        self,
        outlook: Optional[MarketOutlook] = None,
        risk: Optional[RiskProfile] = None,
        max_loss: Optional[float] = None
    ) -> List[str]:
        """
        Suggest templates based on criteria.

        Args:
            outlook: Desired market outlook
            risk: Desired risk profile
            max_loss: Maximum acceptable loss in dollars

        Returns:
            List of suggested template names
        """
        suggestions = []

        for name, template in self._templates.items():
            if outlook and template.outlook != outlook:
                continue
            if risk and template.risk_profile != risk:
                continue

            # Check max loss if specified
            if max_loss is not None:
                try:
                    strategy = self.create(name)
                    if strategy.max_loss > max_loss:
                        continue
                except Exception:
                    continue

            suggestions.append(name)

        return suggestions

    def compare(self, template_names: List[str], expiry_days: int = 30) -> Dict:
        """
        Compare multiple templates side by side.

        Args:
            template_names: List of template names to compare
            expiry_days: Common expiry for comparison

        Returns:
            Comparison dictionary with metrics for each strategy
        """
        comparison = {}

        for name in template_names:
            if name not in self._templates:
                continue

            try:
                strategy = self.create(name, expiry_days=expiry_days)
                metrics = strategy.get_metrics()

                comparison[name] = {
                    "net_premium": strategy.net_premium,
                    "max_profit": metrics.max_profit,
                    "max_loss": metrics.max_loss,
                    "breakevens": metrics.breakevens,
                    "probability_of_profit": metrics.probability_of_profit,
                    "return_on_risk": metrics.return_on_risk,
                    "delta": strategy.delta,
                    "theta": strategy.theta,
                    "vega": strategy.vega,
                }
            except Exception as e:
                comparison[name] = {"error": str(e)}

        return comparison
