"""
Option-Strat: A Comprehensive Options Trading Toolkit

This package provides tools for:
- Options pricing using Black-Scholes model
- Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Multi-leg strategy building
- P&L visualization
- Options flow analysis
- Strategy optimization

Example usage:
    from optionstrat import Option, Strategy, StrategyBuilder

    # Create a simple call option
    call = Option(
        option_type='call',
        strike=100,
        expiry_days=30,
        underlying_price=100,
        volatility=0.25,
        risk_free_rate=0.05
    )

    # Calculate Greeks
    print(f"Delta: {call.delta:.4f}")
    print(f"Gamma: {call.gamma:.4f}")
    print(f"Theta: {call.theta:.4f}")

    # Build a strategy
    builder = StrategyBuilder("AAPL", underlying_price=150)
    iron_condor = builder.iron_condor(
        put_sell=140,
        put_buy=135,
        call_sell=160,
        call_buy=165,
        expiry_days=30
    )

    # Visualize P&L
    iron_condor.plot_payoff()
"""

__version__ = "1.0.0"
__author__ = "Option-Strat Team"

from optionstrat.models.option import Option, OptionLeg
from optionstrat.models.strategy import Strategy
from optionstrat.models.greeks import Greeks
from optionstrat.strategies.builder import StrategyBuilder
from optionstrat.strategies.templates import StrategyTemplates
from optionstrat.visualization.payoff import PayoffVisualizer
from optionstrat.visualization.greeks_chart import GreeksVisualizer
from optionstrat.flow.scanner import FlowScanner
from optionstrat.optimizer.optimizer import StrategyOptimizer

__all__ = [
    "Option",
    "OptionLeg",
    "Strategy",
    "Greeks",
    "StrategyBuilder",
    "StrategyTemplates",
    "PayoffVisualizer",
    "GreeksVisualizer",
    "FlowScanner",
    "StrategyOptimizer",
]
