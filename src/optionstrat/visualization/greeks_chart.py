"""
Greeks Visualization Module

This module provides visualization tools for options Greeks:
- Greeks over price range
- Greeks over time
- Greeks surface plots
- Greeks comparison charts
- Real-time Greeks monitoring display
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class GreeksStyle:
    """Styling for Greeks charts."""
    delta_color: str = "#2196F3"    # Blue
    gamma_color: str = "#FF9800"    # Orange
    theta_color: str = "#E91E63"    # Pink
    vega_color: str = "#4CAF50"     # Green
    rho_color: str = "#9C27B0"      # Purple
    background_color: str = "#1a1a2e"
    text_color: str = "#FFFFFF"
    grid_color: str = "#424242"
    line_width: float = 2.0


class GreeksVisualizer:
    """
    Visualization tools for options Greeks.

    Provides multiple chart types:
    - Greeks over price range
    - Greeks over time decay
    - Greeks over volatility
    - 3D Greeks surface
    - Greeks dashboard

    Example:
        from optionstrat import StrategyBuilder, GreeksVisualizer

        builder = StrategyBuilder("AAPL", 150)
        straddle = builder.long_straddle(150, 30)

        viz = GreeksVisualizer(straddle)
        viz.plot_greeks_vs_price()
        viz.plot_greeks_over_time()
    """

    def __init__(
        self,
        strategy: Any,
        style: Optional[GreeksStyle] = None
    ):
        """
        Initialize visualizer with a strategy.

        Args:
            strategy: Strategy object to visualize
            style: Optional custom chart style
        """
        self.strategy = strategy
        self.style = style or GreeksStyle()

    def _get_price_range(self, margin_pct: float = 0.3) -> Tuple[float, float]:
        """Calculate appropriate price range."""
        strikes = [leg.option.strike for leg in self.strategy.legs]
        min_strike = min(strikes)
        max_strike = max(strikes)
        spread = max_strike - min_strike
        margin = max(spread * 0.5, self.strategy.underlying_price * margin_pct)
        return (min_strike - margin, max_strike + margin)

    def plot_greeks_vs_price(
        self,
        price_range: Optional[Tuple[float, float]] = None,
        price_points: int = 100,
        greeks: List[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ) -> Optional[Figure]:
        """
        Plot Greeks as a function of underlying price.

        Args:
            price_range: (min, max) price range
            price_points: Number of price points
            greeks: List of greeks to plot ['delta', 'gamma', 'theta', 'vega']
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure if available
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available")
            return None

        if greeks is None:
            greeks = ['delta', 'gamma', 'theta', 'vega']

        if price_range is None:
            price_range = self._get_price_range()

        prices = np.linspace(price_range[0], price_range[1], price_points)

        # Calculate Greeks at each price
        greeks_data = {g: [] for g in greeks}

        for price in prices:
            updated_strategy = self.strategy.with_updated_price(price)
            strategy_greeks = updated_strategy.greeks

            if 'delta' in greeks:
                greeks_data['delta'].append(strategy_greeks.delta)
            if 'gamma' in greeks:
                greeks_data['gamma'].append(strategy_greeks.gamma)
            if 'theta' in greeks:
                greeks_data['theta'].append(strategy_greeks.theta)
            if 'vega' in greeks:
                greeks_data['vega'].append(strategy_greeks.vega)

        # Create subplots
        n_greeks = len(greeks)
        fig, axes = plt.subplots(n_greeks, 1, figsize=figsize, facecolor=self.style.background_color)

        if n_greeks == 1:
            axes = [axes]

        greek_colors = {
            'delta': self.style.delta_color,
            'gamma': self.style.gamma_color,
            'theta': self.style.theta_color,
            'vega': self.style.vega_color,
        }

        greek_labels = {
            'delta': 'Delta (Δ)',
            'gamma': 'Gamma (Γ)',
            'theta': 'Theta (Θ) per day',
            'vega': 'Vega (ν) per 1%',
        }

        for ax, greek in zip(axes, greeks):
            ax.set_facecolor(self.style.background_color)

            color = greek_colors.get(greek, '#FFFFFF')
            values = greeks_data[greek]

            ax.plot(prices, values, color=color, linewidth=self.style.line_width)
            ax.fill_between(prices, values, 0, color=color, alpha=0.2)

            # Zero line
            ax.axhline(y=0, color='white', linewidth=0.5, linestyle='-', alpha=0.3)

            # Current price
            ax.axvline(x=self.strategy.underlying_price, color='#FFC107',
                      linewidth=1.5, linestyle='--', alpha=0.8)

            # Mark strikes
            for leg in self.strategy.legs:
                ax.axvline(x=leg.option.strike, color='#9C27B0',
                          linewidth=1, linestyle=':', alpha=0.5)

            ax.set_ylabel(greek_labels.get(greek, greek.title()),
                         fontsize=11, color=self.style.text_color)
            ax.tick_params(colors=self.style.text_color)
            ax.grid(True, alpha=0.3, color=self.style.grid_color)

            # Show current value
            current_val = greeks_data[greek][len(prices)//2]
            ax.annotate(f'{greek.title()}: {current_val:.4f}',
                       xy=(0.02, 0.85), xycoords='axes fraction',
                       fontsize=10, color=color, fontweight='bold')

        axes[-1].set_xlabel('Stock Price ($)', fontsize=12, color=self.style.text_color)
        axes[0].set_title(f'{self.strategy.name} - Greeks vs Price',
                         fontsize=14, color=self.style.text_color, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, facecolor=self.style.background_color)

        return fig

    def plot_greeks_over_time(
        self,
        time_points: int = 50,
        greeks: List[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ) -> Optional[Figure]:
        """
        Plot Greeks as a function of time to expiration.

        Shows how Greeks evolve as expiration approaches.

        Args:
            time_points: Number of time points
            greeks: List of greeks to plot
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure if available
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available")
            return None

        if greeks is None:
            greeks = ['delta', 'gamma', 'theta', 'vega']

        # Get max days from strategy
        max_days = max(leg.option.expiry_days for leg in self.strategy.legs)
        days = np.linspace(max_days, 1, time_points)

        # Calculate Greeks at each time point
        greeks_data = {g: [] for g in greeks}

        from optionstrat.models.option import Option, OptionLeg
        from optionstrat.models.strategy import Strategy

        for d in days:
            # Recreate strategy with new expiry
            new_legs = []
            for leg in self.strategy.legs:
                new_option = Option(
                    option_type=leg.option.option_type,
                    strike=leg.option.strike,
                    expiry_days=int(d),
                    underlying_price=self.strategy.underlying_price,
                    volatility=leg.option.volatility,
                    risk_free_rate=leg.option.risk_free_rate
                )
                new_legs.append(OptionLeg(new_option, leg.quantity, leg.entry_price))

            temp_strategy = Strategy(
                name=self.strategy.name,
                legs=new_legs,
                underlying_price=self.strategy.underlying_price
            )

            strategy_greeks = temp_strategy.greeks

            if 'delta' in greeks:
                greeks_data['delta'].append(strategy_greeks.delta)
            if 'gamma' in greeks:
                greeks_data['gamma'].append(strategy_greeks.gamma)
            if 'theta' in greeks:
                greeks_data['theta'].append(strategy_greeks.theta)
            if 'vega' in greeks:
                greeks_data['vega'].append(strategy_greeks.vega)

        # Create subplots
        n_greeks = len(greeks)
        fig, axes = plt.subplots(n_greeks, 1, figsize=figsize, facecolor=self.style.background_color)

        if n_greeks == 1:
            axes = [axes]

        greek_colors = {
            'delta': self.style.delta_color,
            'gamma': self.style.gamma_color,
            'theta': self.style.theta_color,
            'vega': self.style.vega_color,
        }

        greek_labels = {
            'delta': 'Delta (Δ)',
            'gamma': 'Gamma (Γ)',
            'theta': 'Theta (Θ) per day',
            'vega': 'Vega (ν) per 1%',
        }

        for ax, greek in zip(axes, greeks):
            ax.set_facecolor(self.style.background_color)

            color = greek_colors.get(greek, '#FFFFFF')
            values = greeks_data[greek]

            ax.plot(days, values, color=color, linewidth=self.style.line_width)
            ax.fill_between(days, values, 0, color=color, alpha=0.2)

            # Zero line
            ax.axhline(y=0, color='white', linewidth=0.5, linestyle='-', alpha=0.3)

            # Mark current time
            ax.axvline(x=max_days, color='#FFC107', linewidth=1.5, linestyle='--', alpha=0.8)

            ax.set_ylabel(greek_labels.get(greek, greek.title()),
                         fontsize=11, color=self.style.text_color)
            ax.tick_params(colors=self.style.text_color)
            ax.grid(True, alpha=0.3, color=self.style.grid_color)

            # Invert x-axis (time decreases towards expiry)
            ax.invert_xaxis()

        axes[-1].set_xlabel('Days to Expiration', fontsize=12, color=self.style.text_color)
        axes[0].set_title(f'{self.strategy.name} - Greeks Over Time',
                         fontsize=14, color=self.style.text_color, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, facecolor=self.style.background_color)

        return fig

    def plot_greek_surface(
        self,
        greek: str = 'delta',
        price_range: Optional[Tuple[float, float]] = None,
        price_points: int = 30,
        time_points: int = 30,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> Optional[Figure]:
        """
        Plot 3D surface of a Greek over price and time.

        Args:
            greek: Greek to plot ('delta', 'gamma', 'theta', 'vega')
            price_range: (min, max) price range
            price_points: Number of price points
            time_points: Number of time points
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure if available
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available")
            return None

        if price_range is None:
            price_range = self._get_price_range()

        prices = np.linspace(price_range[0], price_range[1], price_points)
        max_days = max(leg.option.expiry_days for leg in self.strategy.legs)
        days = np.linspace(max_days, 1, time_points)

        # Create meshgrid
        P, D = np.meshgrid(prices, days)
        Z = np.zeros_like(P)

        from optionstrat.models.option import Option, OptionLeg
        from optionstrat.models.strategy import Strategy

        # Calculate Greek at each point
        for i, d in enumerate(days):
            for j, p in enumerate(prices):
                new_legs = []
                for leg in self.strategy.legs:
                    new_option = Option(
                        option_type=leg.option.option_type,
                        strike=leg.option.strike,
                        expiry_days=max(int(d), 1),
                        underlying_price=p,
                        volatility=leg.option.volatility,
                        risk_free_rate=leg.option.risk_free_rate
                    )
                    new_legs.append(OptionLeg(new_option, leg.quantity, leg.entry_price))

                temp_strategy = Strategy(
                    name=self.strategy.name,
                    legs=new_legs,
                    underlying_price=p
                )

                greeks_obj = temp_strategy.greeks
                Z[i, j] = getattr(greeks_obj, greek, 0)

        # Create 3D figure
        fig = plt.figure(figsize=figsize, facecolor=self.style.background_color)
        ax = fig.add_subplot(111, projection='3d', facecolor=self.style.background_color)

        # Color map based on Greek type
        cmaps = {
            'delta': 'coolwarm',
            'gamma': 'YlOrRd',
            'theta': 'RdPu',
            'vega': 'YlGn'
        }

        surf = ax.plot_surface(P, D, Z, cmap=cmaps.get(greek, 'viridis'),
                               linewidth=0, antialiased=True, alpha=0.8)

        # Labels
        ax.set_xlabel('Stock Price ($)', fontsize=10, color=self.style.text_color)
        ax.set_ylabel('Days to Expiration', fontsize=10, color=self.style.text_color)
        ax.set_zlabel(greek.title(), fontsize=10, color=self.style.text_color)
        ax.set_title(f'{self.strategy.name} - {greek.title()} Surface',
                    fontsize=14, color=self.style.text_color, fontweight='bold')

        # Colorbar
        cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
        cbar.ax.tick_params(colors=self.style.text_color)

        ax.tick_params(colors=self.style.text_color)

        if save_path:
            plt.savefig(save_path, dpi=150, facecolor=self.style.background_color)

        return fig

    def plot_interactive_greeks(
        self,
        price_range: Optional[Tuple[float, float]] = None,
        price_points: int = 100
    ) -> Optional[Any]:
        """
        Create interactive Greeks chart with Plotly.

        Args:
            price_range: (min, max) price range
            price_points: Number of price points

        Returns:
            Plotly Figure if available
        """
        if not PLOTLY_AVAILABLE:
            print("plotly not available")
            return None

        if price_range is None:
            price_range = self._get_price_range()

        prices = np.linspace(price_range[0], price_range[1], price_points)

        # Calculate Greeks
        deltas, gammas, thetas, vegas = [], [], [], []

        for price in prices:
            updated_strategy = self.strategy.with_updated_price(price)
            g = updated_strategy.greeks
            deltas.append(g.delta)
            gammas.append(g.gamma)
            thetas.append(g.theta)
            vegas.append(g.vega)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Delta (Δ)', 'Gamma (Γ)', 'Theta (Θ)', 'Vega (ν)'],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Delta
        fig.add_trace(
            go.Scatter(x=prices.tolist(), y=deltas, name='Delta',
                      line=dict(color=self.style.delta_color, width=2),
                      fill='tozeroy', fillcolor=f'rgba(33, 150, 243, 0.2)'),
            row=1, col=1
        )

        # Gamma
        fig.add_trace(
            go.Scatter(x=prices.tolist(), y=gammas, name='Gamma',
                      line=dict(color=self.style.gamma_color, width=2),
                      fill='tozeroy', fillcolor=f'rgba(255, 152, 0, 0.2)'),
            row=1, col=2
        )

        # Theta
        fig.add_trace(
            go.Scatter(x=prices.tolist(), y=thetas, name='Theta',
                      line=dict(color=self.style.theta_color, width=2),
                      fill='tozeroy', fillcolor=f'rgba(233, 30, 99, 0.2)'),
            row=2, col=1
        )

        # Vega
        fig.add_trace(
            go.Scatter(x=prices.tolist(), y=vegas, name='Vega',
                      line=dict(color=self.style.vega_color, width=2),
                      fill='tozeroy', fillcolor=f'rgba(76, 175, 80, 0.2)'),
            row=2, col=2
        )

        # Add current price lines
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_vline(
                    x=self.strategy.underlying_price,
                    line=dict(color='#FFC107', width=2, dash='dash'),
                    row=row, col=col
                )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{self.strategy.name} - Greeks Dashboard',
                font=dict(size=20, color='white')
            ),
            template='plotly_dark',
            showlegend=False,
            height=700
        )

        fig.update_xaxes(title_text='Stock Price ($)')
        fig.update_yaxes(title_text='Value')

        return fig

    def greeks_summary_table(self) -> str:
        """
        Generate ASCII summary table of current Greeks.

        Returns:
            Formatted ASCII table string
        """
        g = self.strategy.greeks

        table = f"""
╔══════════════════════════════════════════════════════════════════╗
║           {self.strategy.name.upper():^52} ║
║                    GREEKS SUMMARY                                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   Delta (Δ)    {g.delta:>10.4f}    Price sensitivity             ║
║   Gamma (Γ)    {g.gamma:>10.4f}    Delta sensitivity             ║
║   Theta (Θ)    {g.theta:>10.4f}    Time decay per day            ║
║   Vega  (ν)    {g.vega:>10.4f}    Volatility sensitivity        ║
║   Rho   (ρ)    {g.rho:>10.4f}    Interest rate sensitivity     ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                    INTERPRETATION                                ║
╠══════════════════════════════════════════════════════════════════╣
"""

        # Add interpretations
        if g.delta > 0:
            table += f"║   Position is NET LONG (bullish bias)                           ║\n"
        else:
            table += f"║   Position is NET SHORT (bearish bias)                          ║\n"

        if g.gamma > 0:
            table += f"║   Gamma POSITIVE: Benefits from price movement                  ║\n"
        else:
            table += f"║   Gamma NEGATIVE: Hurt by price movement                        ║\n"

        if g.theta > 0:
            table += f"║   Theta POSITIVE: Earns from time decay                         ║\n"
        else:
            table += f"║   Theta NEGATIVE: Loses from time decay                         ║\n"

        if g.vega > 0:
            table += f"║   Vega POSITIVE: Benefits from IV increase                      ║\n"
        else:
            table += f"║   Vega NEGATIVE: Benefits from IV decrease                      ║\n"

        table += "╚══════════════════════════════════════════════════════════════════╝"

        return table
