"""
Strategy Optimizer

This module provides tools for finding optimal options strategies
based on target price, date, and risk preferences.

Features:
- Strategy optimization by expected return
- Optimization by probability of profit
- Multi-objective optimization
- Risk-adjusted return optimization
- Strike selection optimization
- Expiry selection optimization
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

from optionstrat.strategies.builder import StrategyBuilder
from optionstrat.models.strategy import Strategy


class OptimizationGoal(Enum):
    """Optimization objective."""
    MAX_RETURN = "max_return"           # Maximize expected return
    MAX_PROBABILITY = "max_probability" # Maximize probability of profit
    MIN_RISK = "min_risk"               # Minimize maximum loss
    RISK_ADJUSTED = "risk_adjusted"     # Maximize Sharpe-like ratio
    BALANCED = "balanced"               # Balance return and probability


@dataclass
class OptimizationResult:
    """
    Result of strategy optimization.

    Attributes:
        strategy: Optimal strategy found
        score: Optimization score
        expected_return: Expected return at target
        probability_of_profit: Probability of profit
        max_profit: Maximum possible profit
        max_loss: Maximum possible loss
        risk_reward_ratio: Risk/reward ratio
        target_price: Target price used
        target_days: Target days used
    """
    strategy: Strategy
    score: float
    expected_return: float
    probability_of_profit: float
    max_profit: float
    max_loss: float
    risk_reward_ratio: float
    target_price: float
    target_days: int
    rank: int = 0

    def summary(self) -> str:
        """Get formatted summary."""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║  OPTIMIZATION RESULT #{self.rank}                                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Strategy:     {self.strategy.name:<40}       ║
║  Score:        {self.score:>10.2f}                                    ║
╠══════════════════════════════════════════════════════════════════╣
║  Target:       ${self.target_price:.2f} in {self.target_days} days                          ║
║  Exp. Return:  ${self.expected_return:>10.2f}                                ║
║  P(Profit):    {self.probability_of_profit*100:>10.1f}%                               ║
╠══════════════════════════════════════════════════════════════════╣
║  Max Profit:   ${self.max_profit:>10.2f}                                ║
║  Max Loss:     ${self.max_loss:>10.2f}                                ║
║  Risk/Reward:  {self.risk_reward_ratio:>10.2f}                                ║
╚══════════════════════════════════════════════════════════════════╝
"""


class StrategyOptimizer:
    """
    Optimizer for finding best options strategies.

    Scans through possible strategy configurations to find the
    optimal trade for given market expectations and risk preferences.

    Example:
        optimizer = StrategyOptimizer(
            symbol='AAPL',
            underlying_price=150,
            volatility=0.30
        )

        # Find best strategy for bullish outlook
        results = optimizer.optimize(
            target_price=160,
            target_days=30,
            goal=OptimizationGoal.MAX_RETURN,
            max_results=10
        )

        for result in results:
            print(result.summary())
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
        Initialize the optimizer.

        Args:
            symbol: Underlying symbol
            underlying_price: Current price
            volatility: Implied volatility
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
        """
        self.symbol = symbol
        self.underlying_price = underlying_price
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

        self.builder = StrategyBuilder(
            symbol=symbol,
            underlying_price=underlying_price,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield
        )

    def _generate_strikes(
        self,
        center: float,
        width: float = 0.2,
        count: int = 10
    ) -> List[float]:
        """Generate strike prices around a center."""
        low = center * (1 - width)
        high = center * (1 + width)
        step = (high - low) / count
        return [round(low + i * step, 0) for i in range(count + 1)]

    def _calculate_score(
        self,
        strategy: Strategy,
        target_price: float,
        target_days: int,
        goal: OptimizationGoal
    ) -> Tuple[float, float, float]:
        """
        Calculate optimization score for a strategy.

        Returns:
            Tuple of (score, expected_return, probability_of_profit)
        """
        # Calculate expected return at target
        expected_return = strategy.profit_at_price_and_time(target_price, target_days)

        # Calculate probability of profit
        prob_profit = strategy.probability_of_profit(num_simulations=1000)

        # Calculate metrics
        max_profit = strategy.max_profit
        max_loss = strategy.max_loss

        # Avoid division by zero
        risk_reward = max_profit / max_loss if max_loss > 0 else float('inf')

        # Calculate score based on goal
        if goal == OptimizationGoal.MAX_RETURN:
            score = expected_return

        elif goal == OptimizationGoal.MAX_PROBABILITY:
            score = prob_profit * 100

        elif goal == OptimizationGoal.MIN_RISK:
            score = -max_loss  # Negative because we want to minimize

        elif goal == OptimizationGoal.RISK_ADJUSTED:
            # Sharpe-like ratio
            if max_loss > 0:
                score = expected_return / max_loss
            else:
                score = expected_return

        elif goal == OptimizationGoal.BALANCED:
            # Combine return and probability
            return_score = expected_return / max(abs(expected_return), 1)
            prob_score = prob_profit
            score = (return_score + prob_score) / 2 * 100

        else:
            score = expected_return

        return score, expected_return, prob_profit

    def optimize(
        self,
        target_price: float,
        target_days: int,
        goal: OptimizationGoal = OptimizationGoal.MAX_RETURN,
        strategies: Optional[List[str]] = None,
        max_results: int = 10,
        max_loss: Optional[float] = None,
        min_probability: Optional[float] = None,
        expiry_range: Tuple[int, int] = None,
        parallel: bool = True
    ) -> List[OptimizationResult]:
        """
        Find optimal strategies for given target.

        Args:
            target_price: Expected price at target date
            target_days: Days until target date
            goal: Optimization objective
            strategies: List of strategy types to consider (None = all)
            max_results: Maximum number of results to return
            max_loss: Maximum acceptable loss (filter)
            min_probability: Minimum probability of profit (filter)
            expiry_range: Range of expiry days to consider
            parallel: Use parallel processing

        Returns:
            List of OptimizationResult sorted by score
        """
        if expiry_range is None:
            expiry_range = (max(7, target_days - 7), target_days + 14)

        if strategies is None:
            strategies = self._get_default_strategies()

        # Generate candidate strategies
        candidates = self._generate_candidates(
            strategies=strategies,
            target_price=target_price,
            expiry_range=expiry_range
        )

        # Evaluate candidates
        results = []

        if parallel:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_candidate,
                        candidate,
                        target_price,
                        target_days,
                        goal
                    ): candidate
                    for candidate in candidates
                }

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception:
                        pass
        else:
            for candidate in candidates:
                try:
                    result = self._evaluate_candidate(
                        candidate, target_price, target_days, goal
                    )
                    if result:
                        results.append(result)
                except Exception:
                    pass

        # Filter results
        if max_loss is not None:
            results = [r for r in results if r.max_loss <= max_loss]

        if min_probability is not None:
            results = [r for r in results if r.probability_of_profit >= min_probability]

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)

        # Assign ranks
        for i, result in enumerate(results[:max_results]):
            result.rank = i + 1

        return results[:max_results]

    def _evaluate_candidate(
        self,
        strategy: Strategy,
        target_price: float,
        target_days: int,
        goal: OptimizationGoal
    ) -> Optional[OptimizationResult]:
        """Evaluate a single candidate strategy."""
        try:
            score, expected_return, prob_profit = self._calculate_score(
                strategy, target_price, target_days, goal
            )

            max_profit = strategy.max_profit
            max_loss = strategy.max_loss

            return OptimizationResult(
                strategy=strategy,
                score=score,
                expected_return=expected_return,
                probability_of_profit=prob_profit,
                max_profit=max_profit,
                max_loss=max_loss,
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else float('inf'),
                target_price=target_price,
                target_days=target_days
            )
        except Exception:
            return None

    def _get_default_strategies(self) -> List[str]:
        """Get default list of strategies to consider."""
        return [
            'long_call', 'long_put',
            'bull_call_spread', 'bear_put_spread',
            'bull_put_spread', 'bear_call_spread',
            'long_straddle', 'long_strangle',
            'iron_condor', 'iron_butterfly',
            'long_call_butterfly', 'long_put_butterfly',
        ]

    def _generate_candidates(
        self,
        strategies: List[str],
        target_price: float,
        expiry_range: Tuple[int, int]
    ) -> List[Strategy]:
        """Generate candidate strategies to evaluate."""
        candidates = []
        strikes = self._generate_strikes(self.underlying_price, width=0.15, count=8)
        expiries = list(range(expiry_range[0], expiry_range[1] + 1, 7))

        for strategy_type in strategies:
            for expiry in expiries:
                try:
                    candidates.extend(
                        self._create_strategy_variants(strategy_type, strikes, expiry)
                    )
                except Exception:
                    pass

        return candidates

    def _create_strategy_variants(
        self,
        strategy_type: str,
        strikes: List[float],
        expiry_days: int
    ) -> List[Strategy]:
        """Create variants of a strategy type with different strikes."""
        variants = []

        if strategy_type == 'long_call':
            for strike in strikes:
                variants.append(self.builder.long_call(strike, expiry_days))

        elif strategy_type == 'long_put':
            for strike in strikes:
                variants.append(self.builder.long_put(strike, expiry_days))

        elif strategy_type == 'bull_call_spread':
            for i, buy in enumerate(strikes[:-1]):
                for sell in strikes[i+1:i+4]:
                    variants.append(self.builder.bull_call_spread(buy, sell, expiry_days))

        elif strategy_type == 'bear_put_spread':
            for i, buy in enumerate(strikes[1:], 1):
                for sell in strikes[max(0, i-3):i]:
                    variants.append(self.builder.bear_put_spread(buy, sell, expiry_days))

        elif strategy_type == 'bull_put_spread':
            for i, sell in enumerate(strikes[1:], 1):
                for buy in strikes[max(0, i-3):i]:
                    variants.append(self.builder.bull_put_spread(sell, buy, expiry_days))

        elif strategy_type == 'bear_call_spread':
            for i, sell in enumerate(strikes[:-1]):
                for buy in strikes[i+1:i+4]:
                    variants.append(self.builder.bear_call_spread(sell, buy, expiry_days))

        elif strategy_type == 'long_straddle':
            for strike in strikes[2:-2]:
                variants.append(self.builder.long_straddle(strike, expiry_days))

        elif strategy_type == 'long_strangle':
            for i, put in enumerate(strikes[:-3]):
                for call in strikes[i+2:i+5]:
                    variants.append(self.builder.long_strangle(put, call, expiry_days))

        elif strategy_type == 'iron_condor':
            # Generate iron condors with different widths
            for center_idx in range(2, len(strikes) - 2):
                for width in range(1, min(3, center_idx, len(strikes) - center_idx - 1)):
                    put_buy = strikes[center_idx - width - 1]
                    put_sell = strikes[center_idx - width]
                    call_sell = strikes[center_idx + width]
                    call_buy = strikes[center_idx + width + 1]
                    variants.append(self.builder.iron_condor(
                        put_buy, put_sell, call_sell, call_buy, expiry_days
                    ))

        elif strategy_type == 'iron_butterfly':
            for center in strikes[2:-2]:
                for wing_width in [5, 10, 15]:
                    variants.append(self.builder.iron_butterfly(
                        center - wing_width, center, center + wing_width, expiry_days
                    ))

        elif strategy_type == 'long_call_butterfly':
            for i in range(1, len(strikes) - 1):
                lower = strikes[max(0, i-1)]
                middle = strikes[i]
                upper = strikes[min(len(strikes)-1, i+1)]
                if lower < middle < upper:
                    variants.append(self.builder.long_call_butterfly(
                        lower, middle, upper, expiry_days
                    ))

        elif strategy_type == 'long_put_butterfly':
            for i in range(1, len(strikes) - 1):
                lower = strikes[max(0, i-1)]
                middle = strikes[i]
                upper = strikes[min(len(strikes)-1, i+1)]
                if lower < middle < upper:
                    variants.append(self.builder.long_put_butterfly(
                        lower, middle, upper, expiry_days
                    ))

        return variants

    def optimize_strikes(
        self,
        strategy_type: str,
        target_price: float,
        target_days: int,
        expiry_days: int,
        goal: OptimizationGoal = OptimizationGoal.MAX_RETURN
    ) -> OptimizationResult:
        """
        Find optimal strikes for a specific strategy type.

        Args:
            strategy_type: Type of strategy ('iron_condor', 'bull_call_spread', etc.)
            target_price: Expected price at target
            target_days: Days until target
            expiry_days: Days to expiration for strategy
            goal: Optimization objective

        Returns:
            Best OptimizationResult for the strategy type
        """
        strikes = self._generate_strikes(self.underlying_price, width=0.20, count=15)
        candidates = self._create_strategy_variants(strategy_type, strikes, expiry_days)

        best_result = None
        best_score = float('-inf')

        for candidate in candidates:
            result = self._evaluate_candidate(candidate, target_price, target_days, goal)
            if result and result.score > best_score:
                best_score = result.score
                best_result = result

        if best_result:
            best_result.rank = 1

        return best_result

    def compare_strategies(
        self,
        target_price: float,
        target_days: int,
        strategies: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Compare different strategy types for a given target.

        Args:
            target_price: Expected price at target
            target_days: Days until target
            strategies: List of strategies to compare

        Returns:
            Comparison dictionary with metrics for each strategy type
        """
        if strategies is None:
            strategies = self._get_default_strategies()

        comparison = {}
        expiry_days = target_days + 7

        for strategy_type in strategies:
            try:
                result = self.optimize_strikes(
                    strategy_type=strategy_type,
                    target_price=target_price,
                    target_days=target_days,
                    expiry_days=expiry_days
                )

                if result:
                    comparison[strategy_type] = {
                        'strategy_name': result.strategy.name,
                        'expected_return': result.expected_return,
                        'probability_of_profit': result.probability_of_profit,
                        'max_profit': result.max_profit,
                        'max_loss': result.max_loss,
                        'risk_reward_ratio': result.risk_reward_ratio,
                        'net_premium': result.strategy.net_premium,
                        'score': result.score
                    }
            except Exception:
                pass

        return comparison

    def find_best_expiry(
        self,
        strategy_type: str,
        target_price: float,
        target_days: int,
        expiry_min: int = 7,
        expiry_max: int = 90,
        goal: OptimizationGoal = OptimizationGoal.MAX_RETURN
    ) -> List[OptimizationResult]:
        """
        Find the optimal expiry date for a strategy.

        Args:
            strategy_type: Type of strategy
            target_price: Expected price at target
            target_days: Days until target
            expiry_min: Minimum expiry to consider
            expiry_max: Maximum expiry to consider
            goal: Optimization objective

        Returns:
            List of results for different expiries, sorted by score
        """
        results = []
        expiries = list(range(expiry_min, expiry_max + 1, 7))

        for expiry in expiries:
            result = self.optimize_strikes(
                strategy_type=strategy_type,
                target_price=target_price,
                target_days=target_days,
                expiry_days=expiry,
                goal=goal
            )

            if result:
                results.append(result)

        results.sort(key=lambda x: x.score, reverse=True)

        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def summary_report(
        self,
        target_price: float,
        target_days: int,
        top_n: int = 5
    ) -> str:
        """
        Generate a comprehensive optimization summary report.

        Args:
            target_price: Expected price at target
            target_days: Days until target
            top_n: Number of top strategies to show

        Returns:
            Formatted report string
        """
        # Get results for different goals
        max_return_results = self.optimize(
            target_price=target_price,
            target_days=target_days,
            goal=OptimizationGoal.MAX_RETURN,
            max_results=top_n
        )

        max_prob_results = self.optimize(
            target_price=target_price,
            target_days=target_days,
            goal=OptimizationGoal.MAX_PROBABILITY,
            max_results=top_n
        )

        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         STRATEGY OPTIMIZATION REPORT                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Symbol:        {self.symbol:>10}                                               ║
║  Current Price: ${self.underlying_price:>10.2f}                                          ║
║  Target Price:  ${target_price:>10.2f}                                          ║
║  Target Days:   {target_days:>10}                                               ║
║  Volatility:    {self.volatility*100:>10.1f}%                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                      TOP STRATEGIES BY EXPECTED RETURN                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
"""

        for result in max_return_results[:3]:
            report += f"║  #{result.rank} {result.strategy.name:<25} | Return: ${result.expected_return:>8.2f} | P(Win): {result.probability_of_profit*100:>5.1f}% ║\n"

        report += f"""╠══════════════════════════════════════════════════════════════════════════════╣
║                    TOP STRATEGIES BY PROBABILITY OF PROFIT                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
"""

        for result in max_prob_results[:3]:
            report += f"║  #{result.rank} {result.strategy.name:<25} | P(Win): {result.probability_of_profit*100:>5.1f}% | Return: ${result.expected_return:>8.2f} ║\n"

        report += "╚══════════════════════════════════════════════════════════════════════════════╝"

        return report
