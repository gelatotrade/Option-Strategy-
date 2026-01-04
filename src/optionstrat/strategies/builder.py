"""
Strategy Builder - Create Complex Options Strategies

This module provides the StrategyBuilder class for easily creating
common options strategies with a fluent interface.

Supports 50+ pre-defined strategies including:
- Basic: Long/Short Call/Put
- Spreads: Bull Call, Bear Put, etc.
- Income: Covered Call, Cash-Secured Put
- Volatility: Straddle, Strangle
- Complex: Iron Condor, Butterfly, Calendar Spreads

Example:
    builder = StrategyBuilder("AAPL", underlying_price=150, volatility=0.25)

    # Create an iron condor
    iron_condor = builder.iron_condor(
        put_buy=140,
        put_sell=145,
        call_sell=155,
        call_buy=160,
        expiry_days=30
    )

    print(iron_condor.summary())
"""

from typing import Optional, List
from optionstrat.models.option import Option, OptionLeg
from optionstrat.models.strategy import Strategy


class StrategyBuilder:
    """
    Fluent builder for creating options strategies.

    This class provides methods for building all common options strategies
    with sensible defaults and clear parameter names.

    Attributes:
        symbol: Underlying symbol
        underlying_price: Current underlying price
        volatility: Implied volatility (default 0.25)
        risk_free_rate: Risk-free rate (default 0.05)
        dividend_yield: Dividend yield (default 0.0)
    """

    def __init__(
        self,
        symbol: str,
        underlying_price: float,
        volatility: float = 0.25,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0
    ):
        """
        Initialize the strategy builder.

        Args:
            symbol: Underlying symbol (e.g., 'AAPL')
            underlying_price: Current price of the underlying
            volatility: Implied volatility (annualized)
            risk_free_rate: Risk-free interest rate
            dividend_yield: Continuous dividend yield
        """
        self.symbol = symbol
        self.underlying_price = underlying_price
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

    def _create_option(
        self,
        option_type: str,
        strike: float,
        expiry_days: int,
        volatility: Optional[float] = None
    ) -> Option:
        """Create an option with the builder's defaults."""
        return Option(
            option_type=option_type,
            strike=strike,
            expiry_days=expiry_days,
            underlying_price=self.underlying_price,
            volatility=volatility or self.volatility,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            symbol=self.symbol
        )

    def _create_strategy(self, name: str, legs: List[OptionLeg], description: str = "") -> Strategy:
        """Create a strategy with the given legs."""
        return Strategy(
            name=name,
            legs=legs,
            underlying_price=self.underlying_price,
            symbol=self.symbol,
            description=description
        )

    # ==================== BASIC STRATEGIES ====================

    def long_call(
        self,
        strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Long Call - Bullish bet with unlimited upside.

        Buy a call option. Profit when stock rises above strike + premium.
        Max profit: Unlimited
        Max loss: Premium paid

        Args:
            strike: Strike price
            expiry_days: Days to expiration
            quantity: Number of contracts
        """
        option = self._create_option('call', strike, expiry_days)
        leg = OptionLeg(option, quantity=quantity)
        return self._create_strategy(
            "Long Call",
            [leg],
            "Bullish strategy with unlimited upside potential"
        )

    def short_call(
        self,
        strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Short Call (Naked Call) - Bearish/neutral bet.

        Sell a call option. Profit when stock stays below strike.
        Max profit: Premium received
        Max loss: Unlimited

        WARNING: High risk strategy!
        """
        option = self._create_option('call', strike, expiry_days)
        leg = OptionLeg(option, quantity=-quantity)
        return self._create_strategy(
            "Short Call",
            [leg],
            "Bearish/neutral strategy - WARNING: Unlimited risk!"
        )

    def long_put(
        self,
        strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Long Put - Bearish bet or portfolio hedge.

        Buy a put option. Profit when stock falls below strike - premium.
        Max profit: Strike - Premium (if stock goes to $0)
        Max loss: Premium paid
        """
        option = self._create_option('put', strike, expiry_days)
        leg = OptionLeg(option, quantity=quantity)
        return self._create_strategy(
            "Long Put",
            [leg],
            "Bearish strategy or portfolio hedge"
        )

    def short_put(
        self,
        strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Short Put (Cash-Secured Put) - Bullish/neutral income strategy.

        Sell a put option. Profit when stock stays above strike.
        Max profit: Premium received
        Max loss: Strike - Premium (if stock goes to $0)
        """
        option = self._create_option('put', strike, expiry_days)
        leg = OptionLeg(option, quantity=-quantity)
        return self._create_strategy(
            "Short Put",
            [leg],
            "Bullish/neutral income strategy"
        )

    # ==================== VERTICAL SPREADS ====================

    def bull_call_spread(
        self,
        buy_strike: float,
        sell_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Bull Call Spread - Bullish with limited risk and reward.

        Buy lower strike call, sell higher strike call.
        Max profit: Difference in strikes - Net debit
        Max loss: Net debit paid
        """
        buy_call = self._create_option('call', buy_strike, expiry_days)
        sell_call = self._create_option('call', sell_strike, expiry_days)

        return self._create_strategy(
            "Bull Call Spread",
            [
                OptionLeg(buy_call, quantity=quantity),
                OptionLeg(sell_call, quantity=-quantity)
            ],
            "Bullish vertical spread with defined risk"
        )

    def bear_call_spread(
        self,
        sell_strike: float,
        buy_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Bear Call Spread (Call Credit Spread) - Bearish income strategy.

        Sell lower strike call, buy higher strike call.
        Max profit: Net credit received
        Max loss: Difference in strikes - Net credit
        """
        sell_call = self._create_option('call', sell_strike, expiry_days)
        buy_call = self._create_option('call', buy_strike, expiry_days)

        return self._create_strategy(
            "Bear Call Spread",
            [
                OptionLeg(sell_call, quantity=-quantity),
                OptionLeg(buy_call, quantity=quantity)
            ],
            "Bearish credit spread"
        )

    def bull_put_spread(
        self,
        sell_strike: float,
        buy_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Bull Put Spread (Put Credit Spread) - Bullish income strategy.

        Sell higher strike put, buy lower strike put.
        Max profit: Net credit received
        Max loss: Difference in strikes - Net credit
        """
        sell_put = self._create_option('put', sell_strike, expiry_days)
        buy_put = self._create_option('put', buy_strike, expiry_days)

        return self._create_strategy(
            "Bull Put Spread",
            [
                OptionLeg(sell_put, quantity=-quantity),
                OptionLeg(buy_put, quantity=quantity)
            ],
            "Bullish credit spread"
        )

    def bear_put_spread(
        self,
        buy_strike: float,
        sell_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Bear Put Spread - Bearish with limited risk and reward.

        Buy higher strike put, sell lower strike put.
        Max profit: Difference in strikes - Net debit
        Max loss: Net debit paid
        """
        buy_put = self._create_option('put', buy_strike, expiry_days)
        sell_put = self._create_option('put', sell_strike, expiry_days)

        return self._create_strategy(
            "Bear Put Spread",
            [
                OptionLeg(buy_put, quantity=quantity),
                OptionLeg(sell_put, quantity=-quantity)
            ],
            "Bearish vertical spread with defined risk"
        )

    # ==================== INCOME STRATEGIES ====================

    def covered_call(
        self,
        call_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Covered Call - Income strategy on existing stock.

        Sell call against 100 shares per contract.
        Max profit: (Strike - Current Price) + Premium
        Max loss: Current Price - Premium (if stock goes to $0)

        Note: This assumes you own the underlying shares.
        """
        call = self._create_option('call', call_strike, expiry_days)

        return self._create_strategy(
            "Covered Call",
            [OptionLeg(call, quantity=-quantity)],
            "Income strategy - sell calls against owned shares"
        )

    def protective_put(
        self,
        put_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Protective Put - Hedge existing stock position.

        Buy put to protect 100 shares per contract.
        Acts as insurance against downside.

        Note: This assumes you own the underlying shares.
        """
        put = self._create_option('put', put_strike, expiry_days)

        return self._create_strategy(
            "Protective Put",
            [OptionLeg(put, quantity=quantity)],
            "Hedge strategy - buy puts to protect owned shares"
        )

    def collar(
        self,
        put_strike: float,
        call_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Collar - Protect stock while reducing cost.

        Buy protective put, sell covered call.
        Limits both upside and downside.

        Note: This assumes you own the underlying shares.
        """
        put = self._create_option('put', put_strike, expiry_days)
        call = self._create_option('call', call_strike, expiry_days)

        return self._create_strategy(
            "Collar",
            [
                OptionLeg(put, quantity=quantity),
                OptionLeg(call, quantity=-quantity)
            ],
            "Protection strategy with limited upside"
        )

    # ==================== VOLATILITY STRATEGIES ====================

    def long_straddle(
        self,
        strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Long Straddle - Bet on high volatility in either direction.

        Buy call and put at same strike.
        Profit when stock moves significantly in either direction.
        Max profit: Unlimited
        Max loss: Total premium paid
        """
        call = self._create_option('call', strike, expiry_days)
        put = self._create_option('put', strike, expiry_days)

        return self._create_strategy(
            "Long Straddle",
            [
                OptionLeg(call, quantity=quantity),
                OptionLeg(put, quantity=quantity)
            ],
            "Volatility play - profit from large move in either direction"
        )

    def short_straddle(
        self,
        strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Short Straddle - Bet on low volatility.

        Sell call and put at same strike.
        Profit when stock stays near strike.
        Max profit: Total premium received
        Max loss: Unlimited

        WARNING: High risk strategy!
        """
        call = self._create_option('call', strike, expiry_days)
        put = self._create_option('put', strike, expiry_days)

        return self._create_strategy(
            "Short Straddle",
            [
                OptionLeg(call, quantity=-quantity),
                OptionLeg(put, quantity=-quantity)
            ],
            "Income from low volatility - WARNING: Unlimited risk!"
        )

    def long_strangle(
        self,
        put_strike: float,
        call_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Long Strangle - Cheaper volatility bet than straddle.

        Buy OTM call and OTM put at different strikes.
        Profit when stock moves significantly.
        Cheaper than straddle but needs bigger move.
        """
        call = self._create_option('call', call_strike, expiry_days)
        put = self._create_option('put', put_strike, expiry_days)

        return self._create_strategy(
            "Long Strangle",
            [
                OptionLeg(call, quantity=quantity),
                OptionLeg(put, quantity=quantity)
            ],
            "Volatility play - cheaper than straddle"
        )

    def short_strangle(
        self,
        put_strike: float,
        call_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Short Strangle - Income from low volatility.

        Sell OTM call and OTM put at different strikes.
        Wider profit zone than short straddle.

        WARNING: High risk strategy!
        """
        call = self._create_option('call', call_strike, expiry_days)
        put = self._create_option('put', put_strike, expiry_days)

        return self._create_strategy(
            "Short Strangle",
            [
                OptionLeg(call, quantity=-quantity),
                OptionLeg(put, quantity=-quantity)
            ],
            "Income from low volatility - WARNING: Unlimited risk!"
        )

    # ==================== IRON STRATEGIES ====================

    def iron_condor(
        self,
        put_buy: float,
        put_sell: float,
        call_sell: float,
        call_buy: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Iron Condor - Neutral strategy with defined risk.

        Combination of bull put spread and bear call spread.
        Profit when stock stays between the short strikes.
        Max profit: Net credit received
        Max loss: Width of wider spread - Net credit

        Args:
            put_buy: Long put strike (lowest)
            put_sell: Short put strike
            call_sell: Short call strike
            call_buy: Long call strike (highest)
        """
        long_put = self._create_option('put', put_buy, expiry_days)
        short_put = self._create_option('put', put_sell, expiry_days)
        short_call = self._create_option('call', call_sell, expiry_days)
        long_call = self._create_option('call', call_buy, expiry_days)

        return self._create_strategy(
            "Iron Condor",
            [
                OptionLeg(long_put, quantity=quantity),
                OptionLeg(short_put, quantity=-quantity),
                OptionLeg(short_call, quantity=-quantity),
                OptionLeg(long_call, quantity=quantity)
            ],
            "Neutral income strategy with defined risk"
        )

    def iron_butterfly(
        self,
        wing_put: float,
        body_strike: float,
        wing_call: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Iron Butterfly - Neutral strategy with tighter profit zone.

        Like iron condor but short strikes are at same price.
        Higher max profit but narrower profit zone.

        Args:
            wing_put: Long put strike (lowest)
            body_strike: Short put and call strike (ATM)
            wing_call: Long call strike (highest)
        """
        long_put = self._create_option('put', wing_put, expiry_days)
        short_put = self._create_option('put', body_strike, expiry_days)
        short_call = self._create_option('call', body_strike, expiry_days)
        long_call = self._create_option('call', wing_call, expiry_days)

        return self._create_strategy(
            "Iron Butterfly",
            [
                OptionLeg(long_put, quantity=quantity),
                OptionLeg(short_put, quantity=-quantity),
                OptionLeg(short_call, quantity=-quantity),
                OptionLeg(long_call, quantity=quantity)
            ],
            "Neutral strategy with higher profit potential than iron condor"
        )

    def reverse_iron_condor(
        self,
        put_sell: float,
        put_buy: float,
        call_buy: float,
        call_sell: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Reverse Iron Condor - Volatility bet with defined risk.

        Opposite of iron condor - profit from big moves.
        Max profit: Width of spread - Net debit
        Max loss: Net debit paid
        """
        short_put = self._create_option('put', put_sell, expiry_days)
        long_put = self._create_option('put', put_buy, expiry_days)
        long_call = self._create_option('call', call_buy, expiry_days)
        short_call = self._create_option('call', call_sell, expiry_days)

        return self._create_strategy(
            "Reverse Iron Condor",
            [
                OptionLeg(short_put, quantity=-quantity),
                OptionLeg(long_put, quantity=quantity),
                OptionLeg(long_call, quantity=quantity),
                OptionLeg(short_call, quantity=-quantity)
            ],
            "Volatility strategy - profit from large moves"
        )

    # ==================== BUTTERFLY STRATEGIES ====================

    def long_call_butterfly(
        self,
        lower_strike: float,
        middle_strike: float,
        upper_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Long Call Butterfly - Neutral with limited risk.

        Buy 1 lower call, sell 2 middle calls, buy 1 upper call.
        Max profit at middle strike at expiration.
        """
        lower = self._create_option('call', lower_strike, expiry_days)
        middle = self._create_option('call', middle_strike, expiry_days)
        upper = self._create_option('call', upper_strike, expiry_days)

        return self._create_strategy(
            "Long Call Butterfly",
            [
                OptionLeg(lower, quantity=quantity),
                OptionLeg(middle, quantity=-2*quantity),
                OptionLeg(upper, quantity=quantity)
            ],
            "Neutral strategy - max profit at middle strike"
        )

    def long_put_butterfly(
        self,
        lower_strike: float,
        middle_strike: float,
        upper_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Long Put Butterfly - Neutral with limited risk.

        Buy 1 upper put, sell 2 middle puts, buy 1 lower put.
        Max profit at middle strike at expiration.
        """
        lower = self._create_option('put', lower_strike, expiry_days)
        middle = self._create_option('put', middle_strike, expiry_days)
        upper = self._create_option('put', upper_strike, expiry_days)

        return self._create_strategy(
            "Long Put Butterfly",
            [
                OptionLeg(lower, quantity=quantity),
                OptionLeg(middle, quantity=-2*quantity),
                OptionLeg(upper, quantity=quantity)
            ],
            "Neutral strategy - max profit at middle strike"
        )

    def short_call_butterfly(
        self,
        lower_strike: float,
        middle_strike: float,
        upper_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Short Call Butterfly - Volatility bet.

        Sell 1 lower call, buy 2 middle calls, sell 1 upper call.
        Profit from large move in either direction.
        """
        lower = self._create_option('call', lower_strike, expiry_days)
        middle = self._create_option('call', middle_strike, expiry_days)
        upper = self._create_option('call', upper_strike, expiry_days)

        return self._create_strategy(
            "Short Call Butterfly",
            [
                OptionLeg(lower, quantity=-quantity),
                OptionLeg(middle, quantity=2*quantity),
                OptionLeg(upper, quantity=-quantity)
            ],
            "Volatility strategy - profit from large moves"
        )

    def broken_wing_butterfly_call(
        self,
        lower_strike: float,
        middle_strike: float,
        upper_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Broken Wing Call Butterfly - Directional butterfly.

        Asymmetric butterfly with directional bias.
        Skip strikes on one side for credit entry.
        """
        lower = self._create_option('call', lower_strike, expiry_days)
        middle = self._create_option('call', middle_strike, expiry_days)
        upper = self._create_option('call', upper_strike, expiry_days)

        return self._create_strategy(
            "Broken Wing Call Butterfly",
            [
                OptionLeg(lower, quantity=quantity),
                OptionLeg(middle, quantity=-2*quantity),
                OptionLeg(upper, quantity=quantity)
            ],
            "Directional butterfly with credit potential"
        )

    # ==================== CONDOR STRATEGIES ====================

    def long_call_condor(
        self,
        strike1: float,
        strike2: float,
        strike3: float,
        strike4: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Long Call Condor - Neutral with wider profit zone.

        Buy 1 lowest call, sell 1 lower-middle call,
        sell 1 upper-middle call, buy 1 highest call.
        Wider profit zone than butterfly.
        """
        c1 = self._create_option('call', strike1, expiry_days)
        c2 = self._create_option('call', strike2, expiry_days)
        c3 = self._create_option('call', strike3, expiry_days)
        c4 = self._create_option('call', strike4, expiry_days)

        return self._create_strategy(
            "Long Call Condor",
            [
                OptionLeg(c1, quantity=quantity),
                OptionLeg(c2, quantity=-quantity),
                OptionLeg(c3, quantity=-quantity),
                OptionLeg(c4, quantity=quantity)
            ],
            "Neutral strategy with wider profit zone than butterfly"
        )

    def long_put_condor(
        self,
        strike1: float,
        strike2: float,
        strike3: float,
        strike4: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Long Put Condor - Neutral with wider profit zone.

        Similar to call condor but using puts.
        """
        p1 = self._create_option('put', strike1, expiry_days)
        p2 = self._create_option('put', strike2, expiry_days)
        p3 = self._create_option('put', strike3, expiry_days)
        p4 = self._create_option('put', strike4, expiry_days)

        return self._create_strategy(
            "Long Put Condor",
            [
                OptionLeg(p1, quantity=quantity),
                OptionLeg(p2, quantity=-quantity),
                OptionLeg(p3, quantity=-quantity),
                OptionLeg(p4, quantity=quantity)
            ],
            "Neutral strategy with wider profit zone"
        )

    # ==================== CALENDAR STRATEGIES ====================

    def call_calendar_spread(
        self,
        strike: float,
        front_expiry_days: int,
        back_expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Call Calendar Spread (Horizontal Spread) - Time decay play.

        Sell near-term call, buy longer-term call at same strike.
        Profit from faster decay of front-month option.
        """
        front = self._create_option('call', strike, front_expiry_days)
        back = self._create_option('call', strike, back_expiry_days)

        return self._create_strategy(
            "Call Calendar Spread",
            [
                OptionLeg(front, quantity=-quantity),
                OptionLeg(back, quantity=quantity)
            ],
            "Time decay strategy - profit from faster near-term decay"
        )

    def put_calendar_spread(
        self,
        strike: float,
        front_expiry_days: int,
        back_expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Put Calendar Spread - Time decay play with puts.

        Sell near-term put, buy longer-term put at same strike.
        """
        front = self._create_option('put', strike, front_expiry_days)
        back = self._create_option('put', strike, back_expiry_days)

        return self._create_strategy(
            "Put Calendar Spread",
            [
                OptionLeg(front, quantity=-quantity),
                OptionLeg(back, quantity=quantity)
            ],
            "Time decay strategy using puts"
        )

    def double_calendar(
        self,
        put_strike: float,
        call_strike: float,
        front_expiry_days: int,
        back_expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Double Calendar - Neutral calendar strategy.

        Combine put and call calendar spreads.
        Profit from time decay while maintaining directional neutrality.
        """
        front_put = self._create_option('put', put_strike, front_expiry_days)
        back_put = self._create_option('put', put_strike, back_expiry_days)
        front_call = self._create_option('call', call_strike, front_expiry_days)
        back_call = self._create_option('call', call_strike, back_expiry_days)

        return self._create_strategy(
            "Double Calendar",
            [
                OptionLeg(front_put, quantity=-quantity),
                OptionLeg(back_put, quantity=quantity),
                OptionLeg(front_call, quantity=-quantity),
                OptionLeg(back_call, quantity=quantity)
            ],
            "Neutral time decay strategy"
        )

    # ==================== DIAGONAL STRATEGIES ====================

    def call_diagonal_spread(
        self,
        front_strike: float,
        back_strike: float,
        front_expiry_days: int,
        back_expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Call Diagonal Spread - Directional calendar.

        Sell near-term call, buy longer-term call at different strike.
        Combines time decay with directional view.
        """
        front = self._create_option('call', front_strike, front_expiry_days)
        back = self._create_option('call', back_strike, back_expiry_days)

        return self._create_strategy(
            "Call Diagonal Spread",
            [
                OptionLeg(front, quantity=-quantity),
                OptionLeg(back, quantity=quantity)
            ],
            "Directional calendar spread"
        )

    def put_diagonal_spread(
        self,
        front_strike: float,
        back_strike: float,
        front_expiry_days: int,
        back_expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Put Diagonal Spread - Bearish diagonal.

        Sell near-term put, buy longer-term put at different strike.
        """
        front = self._create_option('put', front_strike, front_expiry_days)
        back = self._create_option('put', back_strike, back_expiry_days)

        return self._create_strategy(
            "Put Diagonal Spread",
            [
                OptionLeg(front, quantity=-quantity),
                OptionLeg(back, quantity=quantity)
            ],
            "Bearish diagonal spread"
        )

    def poor_mans_covered_call(
        self,
        long_strike: float,
        short_strike: float,
        long_expiry_days: int,
        short_expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Poor Man's Covered Call (PMCC) - Synthetic covered call.

        Buy deep ITM LEAP call, sell near-term OTM call.
        Mimics covered call with less capital.
        """
        long_call = self._create_option('call', long_strike, long_expiry_days)
        short_call = self._create_option('call', short_strike, short_expiry_days)

        return self._create_strategy(
            "Poor Man's Covered Call",
            [
                OptionLeg(long_call, quantity=quantity),
                OptionLeg(short_call, quantity=-quantity)
            ],
            "Synthetic covered call with less capital requirement"
        )

    # ==================== RATIO STRATEGIES ====================

    def call_ratio_spread(
        self,
        buy_strike: float,
        sell_strike: float,
        expiry_days: int,
        buy_quantity: int = 1,
        sell_quantity: int = 2
    ) -> Strategy:
        """
        Call Ratio Spread - Buy 1, sell 2 (or other ratio).

        Buy lower strike calls, sell more higher strike calls.
        Can be entered for credit. Risk on upside.
        """
        buy_call = self._create_option('call', buy_strike, expiry_days)
        sell_call = self._create_option('call', sell_strike, expiry_days)

        return self._create_strategy(
            f"Call Ratio Spread ({buy_quantity}:{sell_quantity})",
            [
                OptionLeg(buy_call, quantity=buy_quantity),
                OptionLeg(sell_call, quantity=-sell_quantity)
            ],
            f"Ratio spread - buy {buy_quantity}, sell {sell_quantity}"
        )

    def put_ratio_spread(
        self,
        buy_strike: float,
        sell_strike: float,
        expiry_days: int,
        buy_quantity: int = 1,
        sell_quantity: int = 2
    ) -> Strategy:
        """
        Put Ratio Spread - Buy 1 higher, sell 2 lower.

        Buy higher strike puts, sell more lower strike puts.
        Risk on downside.
        """
        buy_put = self._create_option('put', buy_strike, expiry_days)
        sell_put = self._create_option('put', sell_strike, expiry_days)

        return self._create_strategy(
            f"Put Ratio Spread ({buy_quantity}:{sell_quantity})",
            [
                OptionLeg(buy_put, quantity=buy_quantity),
                OptionLeg(sell_put, quantity=-sell_quantity)
            ],
            f"Put ratio spread - buy {buy_quantity}, sell {sell_quantity}"
        )

    def call_backspread(
        self,
        sell_strike: float,
        buy_strike: float,
        expiry_days: int,
        sell_quantity: int = 1,
        buy_quantity: int = 2
    ) -> Strategy:
        """
        Call Backspread - Sell 1, buy 2 (reverse ratio).

        Bullish strategy with unlimited upside.
        Sell lower strike, buy more higher strike.
        """
        sell_call = self._create_option('call', sell_strike, expiry_days)
        buy_call = self._create_option('call', buy_strike, expiry_days)

        return self._create_strategy(
            f"Call Backspread ({sell_quantity}:{buy_quantity})",
            [
                OptionLeg(sell_call, quantity=-sell_quantity),
                OptionLeg(buy_call, quantity=buy_quantity)
            ],
            "Bullish strategy with unlimited upside potential"
        )

    def put_backspread(
        self,
        sell_strike: float,
        buy_strike: float,
        expiry_days: int,
        sell_quantity: int = 1,
        buy_quantity: int = 2
    ) -> Strategy:
        """
        Put Backspread - Sell 1 higher, buy 2 lower.

        Bearish strategy with large downside profit potential.
        """
        sell_put = self._create_option('put', sell_strike, expiry_days)
        buy_put = self._create_option('put', buy_strike, expiry_days)

        return self._create_strategy(
            f"Put Backspread ({sell_quantity}:{buy_quantity})",
            [
                OptionLeg(sell_put, quantity=-sell_quantity),
                OptionLeg(buy_put, quantity=buy_quantity)
            ],
            "Bearish strategy with large profit potential"
        )

    # ==================== SYNTHETIC STRATEGIES ====================

    def synthetic_long_stock(
        self,
        strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Synthetic Long Stock - Replicate stock ownership.

        Buy call, sell put at same strike.
        Behaves like owning 100 shares per contract.
        """
        call = self._create_option('call', strike, expiry_days)
        put = self._create_option('put', strike, expiry_days)

        return self._create_strategy(
            "Synthetic Long Stock",
            [
                OptionLeg(call, quantity=quantity),
                OptionLeg(put, quantity=-quantity)
            ],
            "Synthetic stock position using options"
        )

    def synthetic_short_stock(
        self,
        strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Synthetic Short Stock - Replicate short selling.

        Sell call, buy put at same strike.
        Behaves like shorting 100 shares per contract.
        """
        call = self._create_option('call', strike, expiry_days)
        put = self._create_option('put', strike, expiry_days)

        return self._create_strategy(
            "Synthetic Short Stock",
            [
                OptionLeg(call, quantity=-quantity),
                OptionLeg(put, quantity=quantity)
            ],
            "Synthetic short stock position"
        )

    def risk_reversal(
        self,
        put_strike: float,
        call_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Risk Reversal (Combo) - Directional play.

        Sell OTM put, buy OTM call.
        Bullish bet that can be entered for credit.
        """
        put = self._create_option('put', put_strike, expiry_days)
        call = self._create_option('call', call_strike, expiry_days)

        return self._create_strategy(
            "Risk Reversal",
            [
                OptionLeg(put, quantity=-quantity),
                OptionLeg(call, quantity=quantity)
            ],
            "Bullish directional play"
        )

    # ==================== GUTS STRATEGIES ====================

    def long_guts(
        self,
        call_strike: float,
        put_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Long Guts - ITM straddle variation.

        Buy ITM call and ITM put.
        More expensive than straddle but always has intrinsic value.

        Args:
            call_strike: ITM call strike (below current price)
            put_strike: ITM put strike (above current price)
        """
        call = self._create_option('call', call_strike, expiry_days)
        put = self._create_option('put', put_strike, expiry_days)

        return self._create_strategy(
            "Long Guts",
            [
                OptionLeg(call, quantity=quantity),
                OptionLeg(put, quantity=quantity)
            ],
            "ITM straddle - always has intrinsic value"
        )

    def short_guts(
        self,
        call_strike: float,
        put_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Short Guts - Sell ITM call and put.

        High premium collection but requires large margin.

        WARNING: High risk strategy!
        """
        call = self._create_option('call', call_strike, expiry_days)
        put = self._create_option('put', put_strike, expiry_days)

        return self._create_strategy(
            "Short Guts",
            [
                OptionLeg(call, quantity=-quantity),
                OptionLeg(put, quantity=-quantity)
            ],
            "High premium income - WARNING: High risk!"
        )

    # ==================== JADE LIZARD & FRIENDS ====================

    def jade_lizard(
        self,
        put_strike: float,
        call_sell_strike: float,
        call_buy_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Jade Lizard - Short put + call spread.

        Sell put, sell call spread.
        No upside risk if credit > call spread width.
        """
        put = self._create_option('put', put_strike, expiry_days)
        sell_call = self._create_option('call', call_sell_strike, expiry_days)
        buy_call = self._create_option('call', call_buy_strike, expiry_days)

        return self._create_strategy(
            "Jade Lizard",
            [
                OptionLeg(put, quantity=-quantity),
                OptionLeg(sell_call, quantity=-quantity),
                OptionLeg(buy_call, quantity=quantity)
            ],
            "Short put + call credit spread"
        )

    def twisted_sister(
        self,
        put_buy_strike: float,
        put_sell_strike: float,
        call_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Twisted Sister - Put spread + short call.

        Opposite of jade lizard.
        No downside risk if credit > put spread width.
        """
        buy_put = self._create_option('put', put_buy_strike, expiry_days)
        sell_put = self._create_option('put', put_sell_strike, expiry_days)
        call = self._create_option('call', call_strike, expiry_days)

        return self._create_strategy(
            "Twisted Sister",
            [
                OptionLeg(buy_put, quantity=quantity),
                OptionLeg(sell_put, quantity=-quantity),
                OptionLeg(call, quantity=-quantity)
            ],
            "Put debit spread + short call"
        )

    # ==================== CHRISTMAS TREE ====================

    def christmas_tree_call(
        self,
        buy_strike: float,
        sell_strike1: float,
        sell_strike2: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Christmas Tree with Calls - Ladder-like structure.

        Buy 1 call, sell 1 higher call, sell 1 even higher call.
        Bullish with limited upside.
        """
        buy = self._create_option('call', buy_strike, expiry_days)
        sell1 = self._create_option('call', sell_strike1, expiry_days)
        sell2 = self._create_option('call', sell_strike2, expiry_days)

        return self._create_strategy(
            "Christmas Tree (Calls)",
            [
                OptionLeg(buy, quantity=quantity),
                OptionLeg(sell1, quantity=-quantity),
                OptionLeg(sell2, quantity=-quantity)
            ],
            "Bullish ladder with limited upside"
        )

    def christmas_tree_put(
        self,
        buy_strike: float,
        sell_strike1: float,
        sell_strike2: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Christmas Tree with Puts - Bearish ladder.

        Buy 1 put, sell 1 lower put, sell 1 even lower put.
        Bearish with limited downside profit.
        """
        buy = self._create_option('put', buy_strike, expiry_days)
        sell1 = self._create_option('put', sell_strike1, expiry_days)
        sell2 = self._create_option('put', sell_strike2, expiry_days)

        return self._create_strategy(
            "Christmas Tree (Puts)",
            [
                OptionLeg(buy, quantity=quantity),
                OptionLeg(sell1, quantity=-quantity),
                OptionLeg(sell2, quantity=-quantity)
            ],
            "Bearish ladder with limited downside profit"
        )

    # ==================== BOX SPREAD ====================

    def box_spread(
        self,
        lower_strike: float,
        upper_strike: float,
        expiry_days: int,
        quantity: int = 1
    ) -> Strategy:
        """
        Box Spread - Arbitrage strategy.

        Combination of bull call spread and bear put spread.
        Theoretically risk-free profit from mispricing.
        Value at expiration = upper_strike - lower_strike.
        """
        long_call = self._create_option('call', lower_strike, expiry_days)
        short_call = self._create_option('call', upper_strike, expiry_days)
        long_put = self._create_option('put', upper_strike, expiry_days)
        short_put = self._create_option('put', lower_strike, expiry_days)

        return self._create_strategy(
            "Box Spread",
            [
                OptionLeg(long_call, quantity=quantity),
                OptionLeg(short_call, quantity=-quantity),
                OptionLeg(long_put, quantity=quantity),
                OptionLeg(short_put, quantity=-quantity)
            ],
            "Arbitrage strategy - profit from mispricing"
        )

    # ==================== CUSTOM STRATEGY ====================

    def custom(
        self,
        name: str,
        legs: List[dict],
        description: str = ""
    ) -> Strategy:
        """
        Create a custom strategy from leg specifications.

        Args:
            name: Strategy name
            legs: List of leg dictionaries with keys:
                - option_type: 'call' or 'put'
                - strike: Strike price
                - expiry_days: Days to expiration
                - quantity: Positive for long, negative for short
            description: Strategy description

        Example:
            custom_strategy = builder.custom(
                name="My Strategy",
                legs=[
                    {'option_type': 'call', 'strike': 100, 'expiry_days': 30, 'quantity': 1},
                    {'option_type': 'put', 'strike': 95, 'expiry_days': 30, 'quantity': -1}
                ]
            )
        """
        option_legs = []
        for leg in legs:
            option = self._create_option(
                leg['option_type'],
                leg['strike'],
                leg['expiry_days'],
                leg.get('volatility')
            )
            option_legs.append(OptionLeg(option, quantity=leg['quantity']))

        return self._create_strategy(name, option_legs, description)
