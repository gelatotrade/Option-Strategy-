"""
Options Flow Scanner

This module provides tools for tracking and analyzing unusual options
activity, detecting large trades, and identifying potential smart money
movements.

Features:
- Real-time flow monitoring (simulated or with data provider)
- Unusual activity detection
- Multi-leg trade detection
- Trade filtering and alerting
- Historical flow analysis
- Flow aggregation and statistics

Note: This module provides the framework and simulation capabilities.
For real data, integrate with a data provider like OPRA, Unusual Whales,
or similar services.
"""

import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
from enum import Enum
import numpy as np


class TradeSide(Enum):
    """Trade direction."""
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class TradeType(Enum):
    """Type of options trade."""
    CALL = "call"
    PUT = "put"
    SPREAD = "spread"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    CUSTOM = "custom"


class Sentiment(Enum):
    """Inferred sentiment from trade."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass
class FlowTrade:
    """
    Represents a single options flow trade.

    Attributes:
        timestamp: Trade timestamp
        symbol: Underlying symbol
        trade_type: Type of trade (call, put, spread, etc.)
        strike: Strike price(s)
        expiry: Expiration date
        premium: Total premium paid/received
        size: Number of contracts
        side: Buy or sell
        spot_price: Underlying price at time of trade
        implied_volatility: IV at time of trade
        open_interest: Open interest before trade
        volume: Daily volume at trade time
        sentiment: Inferred sentiment
        is_unusual: Whether trade is flagged as unusual
        is_sweep: Whether trade is a sweep order
        is_block: Whether trade is a block trade
        exchange: Exchange where traded
        trade_id: Unique trade identifier
    """
    timestamp: datetime
    symbol: str
    trade_type: TradeType
    strike: float
    expiry: datetime
    premium: float
    size: int
    side: TradeSide
    spot_price: float
    implied_volatility: float = 0.0
    open_interest: int = 0
    volume: int = 0
    sentiment: Sentiment = Sentiment.UNKNOWN
    is_unusual: bool = False
    is_sweep: bool = False
    is_block: bool = False
    exchange: str = ""
    trade_id: str = ""

    # For multi-leg trades
    legs: List[Dict] = field(default_factory=list)

    @property
    def days_to_expiry(self) -> int:
        """Days until expiration."""
        return max((self.expiry - datetime.now()).days, 0)

    @property
    def is_weekly(self) -> bool:
        """Check if option is a weekly."""
        return self.days_to_expiry <= 7

    @property
    def moneyness(self) -> str:
        """Get moneyness status."""
        ratio = self.spot_price / self.strike
        if abs(ratio - 1) < 0.02:
            return "ATM"
        if self.trade_type == TradeType.CALL:
            return "ITM" if ratio > 1 else "OTM"
        else:
            return "ITM" if ratio < 1 else "OTM"

    @property
    def size_category(self) -> str:
        """Categorize trade size."""
        if self.premium >= 1_000_000:
            return "WHALE"
        elif self.premium >= 250_000:
            return "LARGE"
        elif self.premium >= 50_000:
            return "MEDIUM"
        else:
            return "SMALL"

    @property
    def volume_oi_ratio(self) -> float:
        """Volume to open interest ratio."""
        if self.open_interest == 0:
            return float('inf')
        return self.volume / self.open_interest

    def infer_sentiment(self) -> Sentiment:
        """
        Infer sentiment from trade characteristics.

        Returns:
            Sentiment enum
        """
        if self.trade_type == TradeType.CALL:
            if self.side == TradeSide.BUY:
                return Sentiment.BULLISH
            else:
                return Sentiment.BEARISH
        elif self.trade_type == TradeType.PUT:
            if self.side == TradeSide.BUY:
                return Sentiment.BEARISH
            else:
                return Sentiment.BULLISH
        else:
            return Sentiment.NEUTRAL

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'trade_type': self.trade_type.value,
            'strike': self.strike,
            'expiry': self.expiry.isoformat(),
            'premium': self.premium,
            'size': self.size,
            'side': self.side.value,
            'spot_price': self.spot_price,
            'implied_volatility': self.implied_volatility,
            'open_interest': self.open_interest,
            'volume': self.volume,
            'sentiment': self.sentiment.value,
            'is_unusual': self.is_unusual,
            'is_sweep': self.is_sweep,
            'is_block': self.is_block,
            'days_to_expiry': self.days_to_expiry,
            'moneyness': self.moneyness,
            'size_category': self.size_category,
        }

    def summary(self) -> str:
        """Get formatted summary string."""
        flags = []
        if self.is_unusual:
            flags.append("UNUSUAL")
        if self.is_sweep:
            flags.append("SWEEP")
        if self.is_block:
            flags.append("BLOCK")

        flags_str = " ".join(f"[{f}]" for f in flags)

        return f"""
{self.timestamp.strftime('%H:%M:%S')} | {self.symbol:>6} | {self.side.value.upper():>4} {self.size:>5}x {self.trade_type.value.upper():>10} @ ${self.strike:.2f}
           | Exp: {self.expiry.strftime('%Y-%m-%d')} ({self.days_to_expiry}d) | Premium: ${self.premium:,.0f} | IV: {self.implied_volatility*100:.1f}%
           | Spot: ${self.spot_price:.2f} | {self.moneyness} | {self.sentiment.value.upper()} {flags_str}
""".strip()


@dataclass
class FlowFilter:
    """
    Filter criteria for flow scanning.

    Attributes:
        symbols: List of symbols to include (None = all)
        min_premium: Minimum premium threshold
        max_premium: Maximum premium threshold
        trade_types: Types of trades to include
        sides: Trade sides to include
        min_size: Minimum contract size
        max_dte: Maximum days to expiration
        min_dte: Minimum days to expiration
        sentiments: Sentiments to include
        only_unusual: Only show unusual activity
        only_sweeps: Only show sweep orders
        only_blocks: Only show block trades
        min_iv: Minimum implied volatility
        max_iv: Maximum implied volatility
        moneyness: List of moneyness types to include
    """
    symbols: Optional[List[str]] = None
    min_premium: float = 0
    max_premium: float = float('inf')
    trade_types: Optional[List[TradeType]] = None
    sides: Optional[List[TradeSide]] = None
    min_size: int = 0
    max_dte: int = 365
    min_dte: int = 0
    sentiments: Optional[List[Sentiment]] = None
    only_unusual: bool = False
    only_sweeps: bool = False
    only_blocks: bool = False
    min_iv: float = 0
    max_iv: float = float('inf')
    moneyness: Optional[List[str]] = None

    def matches(self, trade: FlowTrade) -> bool:
        """
        Check if a trade matches this filter.

        Args:
            trade: FlowTrade to check

        Returns:
            True if trade matches all criteria
        """
        if self.symbols and trade.symbol not in self.symbols:
            return False

        if not (self.min_premium <= trade.premium <= self.max_premium):
            return False

        if self.trade_types and trade.trade_type not in self.trade_types:
            return False

        if self.sides and trade.side not in self.sides:
            return False

        if trade.size < self.min_size:
            return False

        if not (self.min_dte <= trade.days_to_expiry <= self.max_dte):
            return False

        if self.sentiments and trade.sentiment not in self.sentiments:
            return False

        if self.only_unusual and not trade.is_unusual:
            return False

        if self.only_sweeps and not trade.is_sweep:
            return False

        if self.only_blocks and not trade.is_block:
            return False

        if not (self.min_iv <= trade.implied_volatility <= self.max_iv):
            return False

        if self.moneyness and trade.moneyness not in self.moneyness:
            return False

        return True


class FlowScanner:
    """
    Options flow scanner and analyzer.

    Provides tools for:
    - Scanning for unusual options activity
    - Filtering trades by various criteria
    - Aggregating flow by symbol/sentiment
    - Detecting patterns in flow data
    - Setting up alerts for specific conditions

    Example:
        scanner = FlowScanner()

        # Set up filter
        filter = FlowFilter(
            symbols=['AAPL', 'TSLA'],
            min_premium=100000,
            only_unusual=True
        )

        # Get simulated flow
        trades = scanner.get_flow(filter, limit=20)

        for trade in trades:
            print(trade.summary())

        # Get aggregated stats
        stats = scanner.aggregate_by_symbol(trades)
    """

    def __init__(self, data_provider: Optional[Callable] = None):
        """
        Initialize the flow scanner.

        Args:
            data_provider: Optional callable that returns flow data.
                          If None, simulated data is used.
        """
        self.data_provider = data_provider
        self._alerts: List[Dict] = []
        self._trade_history: List[FlowTrade] = []

        # Popular symbols for simulation
        self._popular_symbols = [
            'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'SPY', 'QQQ', 'IWM', 'DIA', 'NFLX', 'COIN', 'GME', 'AMC',
            'BA', 'JPM', 'GS', 'V', 'MA', 'CRM', 'SHOP', 'SQ'
        ]

    def get_flow(
        self,
        filter: Optional[FlowFilter] = None,
        limit: int = 50,
        simulated: bool = True
    ) -> List[FlowTrade]:
        """
        Get options flow data.

        Args:
            filter: Optional filter criteria
            limit: Maximum number of trades to return
            simulated: Use simulated data (True) or data provider (False)

        Returns:
            List of FlowTrade objects
        """
        if simulated or self.data_provider is None:
            trades = self._generate_simulated_flow(limit * 2)
        else:
            trades = self.data_provider()

        # Apply filter
        if filter:
            trades = [t for t in trades if filter.matches(t)]

        # Store in history
        self._trade_history.extend(trades[:limit])

        return trades[:limit]

    def _generate_simulated_flow(self, count: int) -> List[FlowTrade]:
        """Generate realistic simulated flow data."""
        trades = []
        base_time = datetime.now()

        for i in range(count):
            symbol = random.choice(self._popular_symbols)

            # Generate realistic price
            base_price = random.uniform(50, 500)

            # Generate strike near the money
            strike_offset = random.choice([-3, -2, -1, 0, 1, 2, 3])
            strike = round(base_price + strike_offset * 5, 0)

            # Trade type - mostly single leg, some multi-leg
            if random.random() < 0.15:
                trade_type = random.choice([
                    TradeType.SPREAD,
                    TradeType.STRADDLE,
                    TradeType.STRANGLE,
                    TradeType.IRON_CONDOR
                ])
            else:
                trade_type = random.choice([TradeType.CALL, TradeType.PUT])

            # Expiry - skewed towards near-term
            dte_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            dte_options = [7, 14, 30, 45, 90]
            dte = random.choices(dte_options, weights=dte_weights)[0]
            expiry = base_time + timedelta(days=dte + random.randint(0, 5))

            # Premium - log-normal distribution (most small, few large)
            base_premium = np.random.lognormal(mean=10, sigma=1.5)
            premium = min(base_premium * 1000, 5_000_000)

            # Size
            size = int(premium / (random.uniform(1, 5) * 100))
            size = max(1, min(size, 10000))

            # Side
            side = random.choice([TradeSide.BUY, TradeSide.SELL])

            # IV
            iv = random.uniform(0.15, 0.80)

            # Open interest and volume
            oi = random.randint(100, 50000)
            volume = random.randint(50, oi * 2)

            # Unusual detection
            is_unusual = (
                premium > 100000 or
                volume > oi * 1.5 or
                size > 500
            )

            # Sweep/block detection
            is_sweep = premium > 50000 and random.random() < 0.3
            is_block = size > 200 and random.random() < 0.2

            trade = FlowTrade(
                timestamp=base_time - timedelta(seconds=i * random.randint(1, 30)),
                symbol=symbol,
                trade_type=trade_type,
                strike=strike,
                expiry=expiry,
                premium=premium,
                size=size,
                side=side,
                spot_price=base_price,
                implied_volatility=iv,
                open_interest=oi,
                volume=volume,
                is_unusual=is_unusual,
                is_sweep=is_sweep,
                is_block=is_block,
                exchange=random.choice(['CBOE', 'PHLX', 'ISE', 'AMEX', 'BOX']),
                trade_id=f"TRD{i:08d}"
            )

            trade.sentiment = trade.infer_sentiment()
            trades.append(trade)

        # Sort by timestamp (newest first)
        trades.sort(key=lambda x: x.timestamp, reverse=True)

        return trades

    def aggregate_by_symbol(self, trades: List[FlowTrade]) -> Dict[str, Dict]:
        """
        Aggregate flow statistics by symbol.

        Args:
            trades: List of trades to aggregate

        Returns:
            Dictionary with symbol stats
        """
        stats = {}

        for trade in trades:
            if trade.symbol not in stats:
                stats[trade.symbol] = {
                    'total_premium': 0,
                    'total_contracts': 0,
                    'trade_count': 0,
                    'call_premium': 0,
                    'put_premium': 0,
                    'bullish_premium': 0,
                    'bearish_premium': 0,
                    'avg_iv': [],
                    'unusual_count': 0,
                }

            s = stats[trade.symbol]
            s['total_premium'] += trade.premium
            s['total_contracts'] += trade.size
            s['trade_count'] += 1
            s['avg_iv'].append(trade.implied_volatility)

            if trade.trade_type == TradeType.CALL:
                s['call_premium'] += trade.premium
            elif trade.trade_type == TradeType.PUT:
                s['put_premium'] += trade.premium

            if trade.sentiment == Sentiment.BULLISH:
                s['bullish_premium'] += trade.premium
            elif trade.sentiment == Sentiment.BEARISH:
                s['bearish_premium'] += trade.premium

            if trade.is_unusual:
                s['unusual_count'] += 1

        # Calculate final metrics
        for symbol, s in stats.items():
            s['avg_iv'] = np.mean(s['avg_iv']) if s['avg_iv'] else 0
            s['put_call_ratio'] = (
                s['put_premium'] / s['call_premium']
                if s['call_premium'] > 0 else float('inf')
            )
            s['sentiment_score'] = (
                (s['bullish_premium'] - s['bearish_premium']) /
                (s['bullish_premium'] + s['bearish_premium'])
                if (s['bullish_premium'] + s['bearish_premium']) > 0 else 0
            )

        return stats

    def aggregate_by_sentiment(self, trades: List[FlowTrade]) -> Dict[str, Dict]:
        """
        Aggregate flow statistics by sentiment.

        Args:
            trades: List of trades to aggregate

        Returns:
            Dictionary with sentiment stats
        """
        stats = {
            'bullish': {'premium': 0, 'count': 0, 'contracts': 0},
            'bearish': {'premium': 0, 'count': 0, 'contracts': 0},
            'neutral': {'premium': 0, 'count': 0, 'contracts': 0},
        }

        for trade in trades:
            key = trade.sentiment.value
            if key in stats:
                stats[key]['premium'] += trade.premium
                stats[key]['count'] += 1
                stats[key]['contracts'] += trade.size

        total_premium = sum(s['premium'] for s in stats.values())
        for key in stats:
            stats[key]['percentage'] = (
                stats[key]['premium'] / total_premium * 100
                if total_premium > 0 else 0
            )

        return stats

    def get_top_trades(
        self,
        trades: List[FlowTrade],
        by: str = 'premium',
        limit: int = 10
    ) -> List[FlowTrade]:
        """
        Get top trades by specified metric.

        Args:
            trades: List of trades
            by: Metric to sort by ('premium', 'size', 'iv')
            limit: Number of trades to return

        Returns:
            Top trades sorted by metric
        """
        if by == 'premium':
            sorted_trades = sorted(trades, key=lambda x: x.premium, reverse=True)
        elif by == 'size':
            sorted_trades = sorted(trades, key=lambda x: x.size, reverse=True)
        elif by == 'iv':
            sorted_trades = sorted(trades, key=lambda x: x.implied_volatility, reverse=True)
        else:
            sorted_trades = trades

        return sorted_trades[:limit]

    def add_alert(
        self,
        name: str,
        filter: FlowFilter,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Add an alert for specific flow conditions.

        Args:
            name: Alert name
            filter: Filter defining alert conditions
            callback: Optional callback function when triggered

        Returns:
            Alert ID
        """
        alert_id = f"ALERT_{len(self._alerts):04d}"
        self._alerts.append({
            'id': alert_id,
            'name': name,
            'filter': filter,
            'callback': callback,
            'triggered_count': 0,
            'last_triggered': None,
        })
        return alert_id

    def check_alerts(self, trades: List[FlowTrade]) -> List[Dict]:
        """
        Check trades against alerts.

        Args:
            trades: List of trades to check

        Returns:
            List of triggered alerts
        """
        triggered = []

        for trade in trades:
            for alert in self._alerts:
                if alert['filter'].matches(trade):
                    alert['triggered_count'] += 1
                    alert['last_triggered'] = datetime.now()

                    triggered.append({
                        'alert': alert,
                        'trade': trade
                    })

                    if alert['callback']:
                        alert['callback'](trade)

        return triggered

    def flow_summary(self, trades: List[FlowTrade]) -> str:
        """
        Generate flow summary report.

        Args:
            trades: List of trades

        Returns:
            Formatted summary string
        """
        if not trades:
            return "No trades to summarize."

        total_premium = sum(t.premium for t in trades)
        total_contracts = sum(t.size for t in trades)
        unusual_count = sum(1 for t in trades if t.is_unusual)
        sweep_count = sum(1 for t in trades if t.is_sweep)

        sentiment_stats = self.aggregate_by_sentiment(trades)
        symbol_stats = self.aggregate_by_symbol(trades)

        # Top symbols by premium
        top_symbols = sorted(
            symbol_stats.items(),
            key=lambda x: x[1]['total_premium'],
            reverse=True
        )[:5]

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    OPTIONS FLOW SUMMARY                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Total Trades:      {len(trades):>10,}                               ║
║  Total Premium:     ${total_premium:>12,.0f}                          ║
║  Total Contracts:   {total_contracts:>10,}                               ║
║  Unusual Trades:    {unusual_count:>10,}                               ║
║  Sweep Orders:      {sweep_count:>10,}                               ║
╠══════════════════════════════════════════════════════════════════╣
║                      SENTIMENT BREAKDOWN                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Bullish: ${sentiment_stats['bullish']['premium']:>12,.0f} ({sentiment_stats['bullish']['percentage']:.1f}%)                     ║
║  Bearish: ${sentiment_stats['bearish']['premium']:>12,.0f} ({sentiment_stats['bearish']['percentage']:.1f}%)                     ║
║  Neutral: ${sentiment_stats['neutral']['premium']:>12,.0f} ({sentiment_stats['neutral']['percentage']:.1f}%)                     ║
╠══════════════════════════════════════════════════════════════════╣
║                      TOP SYMBOLS BY PREMIUM                      ║
╠══════════════════════════════════════════════════════════════════╣
"""
        for symbol, stats in top_symbols:
            sentiment = "BULL" if stats['sentiment_score'] > 0.1 else ("BEAR" if stats['sentiment_score'] < -0.1 else "NEUT")
            report += f"║  {symbol:>6}: ${stats['total_premium']:>12,.0f} | P/C: {stats['put_call_ratio']:.2f} | {sentiment:>4}  ║\n"

        report += "╚══════════════════════════════════════════════════════════════════╝"

        return report

    def detect_unusual_activity(
        self,
        trades: List[FlowTrade],
        premium_threshold: float = 100000,
        volume_oi_threshold: float = 2.0
    ) -> List[FlowTrade]:
        """
        Detect unusual options activity.

        Args:
            trades: List of trades to analyze
            premium_threshold: Minimum premium for unusual
            volume_oi_threshold: Minimum volume/OI ratio for unusual

        Returns:
            List of unusual trades
        """
        unusual = []

        for trade in trades:
            is_unusual = (
                trade.premium >= premium_threshold or
                trade.volume_oi_ratio >= volume_oi_threshold or
                trade.is_sweep or
                trade.is_block
            )

            if is_unusual:
                trade.is_unusual = True
                unusual.append(trade)

        return unusual
