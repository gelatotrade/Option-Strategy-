"""
Payoff Visualization Module

This module provides comprehensive visualization tools for options
payoff diagrams, P&L matrices, and probability distributions.

Features:
- Static payoff diagrams (matplotlib)
- Interactive charts (plotly)
- P&L heatmaps over price and time
- Probability distribution visualization
- Multi-strategy comparison charts
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
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
class ChartStyle:
    """Chart styling configuration."""
    profit_color: str = "#00C853"  # Green
    loss_color: str = "#FF1744"    # Red
    neutral_color: str = "#FFFFFF"
    line_color: str = "#2196F3"    # Blue
    grid_color: str = "#424242"
    background_color: str = "#1a1a2e"
    text_color: str = "#FFFFFF"
    font_family: str = "Arial"
    line_width: float = 2.0


class PayoffVisualizer:
    """
    Visualization tools for options payoff analysis.

    Provides multiple chart types:
    - Payoff at expiration
    - P&L over time
    - P&L heatmap
    - Probability distribution
    - Strategy comparison

    Example:
        from optionstrat import StrategyBuilder, PayoffVisualizer

        builder = StrategyBuilder("AAPL", 150)
        iron_condor = builder.iron_condor(140, 145, 155, 160, 30)

        viz = PayoffVisualizer(iron_condor)
        viz.plot_payoff()
        viz.plot_pnl_heatmap()
    """

    def __init__(
        self,
        strategy: Any,
        style: Optional[ChartStyle] = None
    ):
        """
        Initialize visualizer with a strategy.

        Args:
            strategy: Strategy object to visualize
            style: Optional custom chart style
        """
        self.strategy = strategy
        self.style = style or ChartStyle()

    def _get_price_range(
        self,
        margin_pct: float = 0.3
    ) -> Tuple[float, float]:
        """Calculate appropriate price range for charts."""
        strikes = [leg.option.strike for leg in self.strategy.legs]
        min_strike = min(strikes)
        max_strike = max(strikes)
        spread = max_strike - min_strike

        # Add margin based on spread or underlying price
        margin = max(spread * 0.5, self.strategy.underlying_price * margin_pct)

        return (min_strike - margin, max_strike + margin)

    def plot_payoff(
        self,
        price_range: Optional[Tuple[float, float]] = None,
        price_points: int = 200,
        show_breakevens: bool = True,
        show_max_profit_loss: bool = True,
        show_current_price: bool = True,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Optional[Figure]:
        """
        Plot payoff diagram at expiration.

        Args:
            price_range: (min, max) price range
            price_points: Number of price points
            show_breakevens: Show breakeven lines
            show_max_profit_loss: Annotate max profit/loss
            show_current_price: Show current price line
            figsize: Figure size
            title: Custom title
            save_path: Path to save figure

        Returns:
            matplotlib Figure if available
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available. Install with: pip install matplotlib")
            return None

        if price_range is None:
            price_range = self._get_price_range()

        # Generate data
        prices = np.linspace(price_range[0], price_range[1], price_points)
        profits = [self.strategy.profit_at_expiry(p) for p in prices]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.style.background_color)
        ax.set_facecolor(self.style.background_color)

        # Create gradient fill
        profits_array = np.array(profits)
        ax.fill_between(
            prices,
            profits_array,
            0,
            where=profits_array >= 0,
            color=self.style.profit_color,
            alpha=0.3,
            label='Profit'
        )
        ax.fill_between(
            prices,
            profits_array,
            0,
            where=profits_array < 0,
            color=self.style.loss_color,
            alpha=0.3,
            label='Loss'
        )

        # Plot main line
        ax.plot(
            prices,
            profits,
            color=self.style.line_color,
            linewidth=self.style.line_width,
            label='P&L at Expiration'
        )

        # Zero line
        ax.axhline(y=0, color=self.style.neutral_color, linewidth=0.8, linestyle='-', alpha=0.5)

        # Current price
        if show_current_price:
            ax.axvline(
                x=self.strategy.underlying_price,
                color='#FFC107',
                linewidth=1.5,
                linestyle='--',
                label=f'Current: ${self.strategy.underlying_price:.2f}'
            )

        # Breakevens
        if show_breakevens:
            breakevens = self.strategy.breakevens
            for be in breakevens:
                ax.axvline(
                    x=be,
                    color='#9C27B0',
                    linewidth=1,
                    linestyle=':',
                    alpha=0.8
                )
                ax.annotate(
                    f'BE: ${be:.2f}',
                    xy=(be, 0),
                    xytext=(be, max(profits) * 0.1),
                    fontsize=9,
                    color=self.style.text_color,
                    ha='center'
                )

        # Max profit/loss annotations
        if show_max_profit_loss:
            max_profit = max(profits)
            max_loss = min(profits)
            max_profit_price = prices[np.argmax(profits)]
            max_loss_price = prices[np.argmin(profits)]

            # Annotate max profit
            ax.annotate(
                f'Max Profit: ${max_profit:.2f}',
                xy=(max_profit_price, max_profit),
                xytext=(max_profit_price, max_profit + abs(max_profit) * 0.1),
                fontsize=10,
                color=self.style.profit_color,
                ha='center',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=self.style.profit_color, lw=1)
            )

            # Annotate max loss
            ax.annotate(
                f'Max Loss: ${max_loss:.2f}',
                xy=(max_loss_price, max_loss),
                xytext=(max_loss_price, max_loss - abs(max_loss) * 0.15),
                fontsize=10,
                color=self.style.loss_color,
                ha='center',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=self.style.loss_color, lw=1)
            )

        # Styling
        ax.set_xlabel('Stock Price at Expiration ($)', fontsize=12, color=self.style.text_color)
        ax.set_ylabel('Profit / Loss ($)', fontsize=12, color=self.style.text_color)
        ax.set_title(
            title or f'{self.strategy.name} - Payoff Diagram',
            fontsize=14,
            color=self.style.text_color,
            fontweight='bold'
        )

        ax.tick_params(colors=self.style.text_color)
        ax.spines['bottom'].set_color(self.style.grid_color)
        ax.spines['top'].set_color(self.style.grid_color)
        ax.spines['left'].set_color(self.style.grid_color)
        ax.spines['right'].set_color(self.style.grid_color)

        ax.grid(True, alpha=0.3, color=self.style.grid_color)
        ax.legend(loc='upper right', facecolor=self.style.background_color,
                  edgecolor=self.style.grid_color, labelcolor=self.style.text_color)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, facecolor=self.style.background_color)

        return fig

    def plot_pnl_heatmap(
        self,
        price_range: Optional[Tuple[float, float]] = None,
        price_points: int = 50,
        time_points: int = 20,
        figsize: Tuple[int, int] = (14, 8),
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Optional[Figure]:
        """
        Plot P&L heatmap over price and time.

        Shows how the strategy value changes as price and time change.

        Args:
            price_range: (min, max) price range
            price_points: Number of price points
            time_points: Number of time points
            figsize: Figure size
            title: Custom title
            save_path: Path to save figure

        Returns:
            matplotlib Figure if available
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available")
            return None

        if price_range is None:
            price_range = self._get_price_range()

        # Generate P&L matrix
        pnl_data = self.strategy.pnl_matrix(
            price_range=price_range,
            price_points=price_points,
            time_points=time_points
        )

        prices = np.array(pnl_data['prices'])
        days = np.array(pnl_data['days'])
        pnl = np.array(pnl_data['pnl'])

        # Create custom colormap (red for loss, white for zero, green for profit)
        colors = [self.style.loss_color, self.style.neutral_color, self.style.profit_color]
        n_bins = 100
        cmap = mcolors.LinearSegmentedColormap.from_list('pnl', colors, N=n_bins)

        # Normalize around zero
        max_abs = max(abs(pnl.min()), abs(pnl.max()))
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.style.background_color)
        ax.set_facecolor(self.style.background_color)

        # Create heatmap
        im = ax.imshow(
            pnl,
            cmap=cmap,
            norm=norm,
            aspect='auto',
            extent=[prices[0], prices[-1], days[-1], days[0]]
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='P&L ($)')
        cbar.ax.yaxis.label.set_color(self.style.text_color)
        cbar.ax.tick_params(colors=self.style.text_color)

        # Mark current price
        ax.axvline(
            x=self.strategy.underlying_price,
            color='#FFC107',
            linewidth=2,
            linestyle='--',
            label=f'Current: ${self.strategy.underlying_price:.2f}'
        )

        # Breakeven lines
        breakevens = self.strategy.breakevens
        for be in breakevens:
            if prices[0] <= be <= prices[-1]:
                ax.axvline(x=be, color='#9C27B0', linewidth=1, linestyle=':')

        # Styling
        ax.set_xlabel('Stock Price ($)', fontsize=12, color=self.style.text_color)
        ax.set_ylabel('Days to Expiration', fontsize=12, color=self.style.text_color)
        ax.set_title(
            title or f'{self.strategy.name} - P&L Heatmap',
            fontsize=14,
            color=self.style.text_color,
            fontweight='bold'
        )

        ax.tick_params(colors=self.style.text_color)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, facecolor=self.style.background_color)

        return fig

    def plot_interactive(
        self,
        price_range: Optional[Tuple[float, float]] = None,
        price_points: int = 200,
        show_legs: bool = True
    ) -> Optional[Any]:
        """
        Create interactive payoff chart with Plotly.

        Features:
        - Hover to see exact P&L at any price
        - Toggle individual leg contributions
        - Zoom and pan

        Args:
            price_range: (min, max) price range
            price_points: Number of price points
            show_legs: Show individual leg payoffs

        Returns:
            Plotly Figure if available
        """
        if not PLOTLY_AVAILABLE:
            print("plotly not available. Install with: pip install plotly")
            return None

        if price_range is None:
            price_range = self._get_price_range()

        prices = np.linspace(price_range[0], price_range[1], price_points)
        profits = [self.strategy.profit_at_expiry(p) for p in prices]

        # Create figure
        fig = go.Figure()

        # Add profit/loss fill areas
        positive_profits = [max(p, 0) for p in profits]
        negative_profits = [min(p, 0) for p in profits]

        fig.add_trace(go.Scatter(
            x=prices.tolist(),
            y=positive_profits,
            fill='tozeroy',
            fillcolor='rgba(0, 200, 83, 0.3)',
            line=dict(width=0),
            name='Profit Zone',
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=prices.tolist(),
            y=negative_profits,
            fill='tozeroy',
            fillcolor='rgba(255, 23, 68, 0.3)',
            line=dict(width=0),
            name='Loss Zone',
            hoverinfo='skip'
        ))

        # Main P&L line
        fig.add_trace(go.Scatter(
            x=prices.tolist(),
            y=profits,
            mode='lines',
            name='Strategy P&L',
            line=dict(color=self.style.line_color, width=3),
            hovertemplate='Price: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>'
        ))

        # Individual legs if requested
        if show_legs:
            for i, leg in enumerate(self.strategy.legs):
                leg_profits = [leg.profit_at_expiry(p) for p in prices]
                direction = "Long" if leg.is_long else "Short"
                leg_name = f"{direction} {leg.option.option_type.title()} ${leg.option.strike}"

                fig.add_trace(go.Scatter(
                    x=prices.tolist(),
                    y=leg_profits,
                    mode='lines',
                    name=leg_name,
                    line=dict(dash='dot', width=1),
                    visible='legendonly',
                    hovertemplate=f'{leg_name}<br>Price: $%{{x:.2f}}<br>P&L: $%{{y:.2f}}<extra></extra>'
                ))

        # Current price line
        fig.add_vline(
            x=self.strategy.underlying_price,
            line=dict(color='#FFC107', width=2, dash='dash'),
            annotation_text=f"Current: ${self.strategy.underlying_price:.2f}",
            annotation_position="top"
        )

        # Breakeven lines
        for be in self.strategy.breakevens:
            fig.add_vline(
                x=be,
                line=dict(color='#9C27B0', width=1, dash='dot'),
                annotation_text=f"BE: ${be:.2f}",
                annotation_position="bottom"
            )

        # Zero line
        fig.add_hline(y=0, line=dict(color='white', width=1))

        # Layout
        fig.update_layout(
            title=dict(
                text=f'{self.strategy.name} - Interactive Payoff',
                font=dict(size=20, color='white')
            ),
            xaxis_title='Stock Price at Expiration ($)',
            yaxis_title='Profit / Loss ($)',
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            ),
            margin=dict(r=150)
        )

        return fig

    def plot_pnl_table(
        self,
        price_points: List[float] = None,
        time_points: List[int] = None,
        as_percentage: bool = False
    ) -> str:
        """
        Generate ASCII P&L table.

        Args:
            price_points: List of prices to show
            time_points: List of days to show
            as_percentage: Show as percentage of max risk

        Returns:
            Formatted ASCII table string
        """
        if price_points is None:
            strikes = sorted([leg.option.strike for leg in self.strategy.legs])
            margin = (strikes[-1] - strikes[0]) * 0.5
            price_points = np.linspace(strikes[0] - margin, strikes[-1] + margin, 7).tolist()

        if time_points is None:
            max_days = self.strategy.legs[0].option.expiry_days
            time_points = [max_days, max_days // 2, max_days // 4, 1]

        max_risk = self.strategy.max_loss

        # Build table
        header = "Price     | " + " | ".join([f"T-{d:3d}" for d in time_points]) + " | At Exp"
        separator = "-" * len(header)

        rows = [header, separator]

        for price in price_points:
            row = f"${price:7.2f} | "
            for days in time_points:
                pnl = self.strategy.profit_at_price_and_time(price, days)
                if as_percentage and max_risk > 0:
                    value = pnl / max_risk * 100
                    row += f"{value:6.1f}% | "
                else:
                    row += f"${pnl:6.0f} | "

            # At expiration
            pnl = self.strategy.profit_at_expiry(price)
            if as_percentage and max_risk > 0:
                value = pnl / max_risk * 100
                row += f"{value:6.1f}%"
            else:
                row += f"${pnl:6.0f}"

            rows.append(row)

        return "\n".join(rows)

    def plot_probability_distribution(
        self,
        num_simulations: int = 10000,
        figsize: Tuple[int, int] = (12, 6),
        bins: int = 50,
        save_path: Optional[str] = None
    ) -> Optional[Figure]:
        """
        Plot Monte Carlo probability distribution of outcomes.

        Args:
            num_simulations: Number of MC simulations
            figsize: Figure size
            bins: Number of histogram bins
            save_path: Path to save figure

        Returns:
            matplotlib Figure if available
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available")
            return None

        # Run Monte Carlo simulation
        leg = self.strategy.legs[0]
        S = self.strategy.underlying_price
        T = leg.option.time_to_expiry
        r = leg.option.risk_free_rate
        sigma = leg.option.volatility

        np.random.seed(42)
        z = np.random.standard_normal(num_simulations)
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

        # Calculate profits for each simulation
        profits = np.array([self.strategy.profit_at_expiry(s) for s in ST])

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor=self.style.background_color)

        for ax in [ax1, ax2]:
            ax.set_facecolor(self.style.background_color)

        # Histogram of profits
        colors = [self.style.loss_color if p < 0 else self.style.profit_color for p in np.histogram_bin_edges(profits, bins=bins)[:-1]]

        n, bins_edges, patches = ax1.hist(profits, bins=bins, edgecolor='black', alpha=0.7)

        # Color bars based on profit/loss
        for i, patch in enumerate(patches):
            if bins_edges[i] < 0:
                patch.set_facecolor(self.style.loss_color)
            else:
                patch.set_facecolor(self.style.profit_color)

        ax1.axvline(x=0, color='white', linewidth=2, linestyle='--')
        ax1.axvline(x=np.mean(profits), color='#FFC107', linewidth=2, label=f'Mean: ${np.mean(profits):.2f}')
        ax1.axvline(x=np.median(profits), color='#00BCD4', linewidth=2, linestyle=':', label=f'Median: ${np.median(profits):.2f}')

        ax1.set_xlabel('Profit / Loss ($)', fontsize=12, color=self.style.text_color)
        ax1.set_ylabel('Frequency', fontsize=12, color=self.style.text_color)
        ax1.set_title('P&L Distribution', fontsize=14, color=self.style.text_color, fontweight='bold')
        ax1.tick_params(colors=self.style.text_color)
        ax1.legend(facecolor=self.style.background_color, labelcolor=self.style.text_color)

        # Cumulative distribution
        sorted_profits = np.sort(profits)
        cumulative = np.arange(1, len(sorted_profits) + 1) / len(sorted_profits)

        ax2.plot(sorted_profits, cumulative, color=self.style.line_color, linewidth=2)
        ax2.axvline(x=0, color='white', linewidth=1, linestyle='--')
        ax2.axhline(y=0.5, color='#FFC107', linewidth=1, linestyle=':', alpha=0.5)

        # Mark key percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            val = np.percentile(profits, p)
            ax2.plot(val, p/100, 'o', color='#FFC107', markersize=8)
            ax2.annotate(f'P{p}: ${val:.0f}', xy=(val, p/100), xytext=(5, 0),
                        textcoords='offset points', fontsize=9, color=self.style.text_color)

        ax2.set_xlabel('Profit / Loss ($)', fontsize=12, color=self.style.text_color)
        ax2.set_ylabel('Cumulative Probability', fontsize=12, color=self.style.text_color)
        ax2.set_title('Cumulative Distribution', fontsize=14, color=self.style.text_color, fontweight='bold')
        ax2.tick_params(colors=self.style.text_color)
        ax2.grid(True, alpha=0.3)

        # Add statistics text box
        prob_profit = np.mean(profits > 0)
        expected_value = np.mean(profits)
        std_dev = np.std(profits)
        max_sim_profit = np.max(profits)
        max_sim_loss = np.min(profits)

        stats_text = f"""
Probability of Profit: {prob_profit*100:.1f}%
Expected Value: ${expected_value:.2f}
Std Deviation: ${std_dev:.2f}
Max Profit (sim): ${max_sim_profit:.2f}
Max Loss (sim): ${max_sim_loss:.2f}
        """.strip()

        fig.text(0.02, 0.02, stats_text, fontsize=10, color=self.style.text_color,
                 family='monospace', verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor=self.style.background_color,
                          edgecolor=self.style.grid_color))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, facecolor=self.style.background_color)

        return fig


def compare_strategies(
    strategies: List[Any],
    price_range: Optional[Tuple[float, float]] = None,
    price_points: int = 200,
    figsize: Tuple[int, int] = (14, 8)
) -> Optional[Figure]:
    """
    Compare multiple strategies on the same chart.

    Args:
        strategies: List of Strategy objects
        price_range: (min, max) price range
        price_points: Number of price points
        figsize: Figure size

    Returns:
        matplotlib Figure if available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None

    if not strategies:
        return None

    # Determine price range from all strategies
    if price_range is None:
        all_strikes = []
        for s in strategies:
            all_strikes.extend([leg.option.strike for leg in s.legs])
        margin = (max(all_strikes) - min(all_strikes)) * 0.5
        price_range = (min(all_strikes) - margin, max(all_strikes) + margin)

    prices = np.linspace(price_range[0], price_range[1], price_points)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Color palette
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4']

    for i, strategy in enumerate(strategies):
        profits = [strategy.profit_at_expiry(p) for p in prices]
        color = colors[i % len(colors)]

        ax.plot(prices, profits, color=color, linewidth=2.5, label=strategy.name)

        # Mark breakevens
        for be in strategy.breakevens:
            if price_range[0] <= be <= price_range[1]:
                ax.axvline(x=be, color=color, linewidth=1, linestyle=':', alpha=0.5)

    # Zero line
    ax.axhline(y=0, color='white', linewidth=1, linestyle='-', alpha=0.5)

    # Current price (use first strategy)
    ax.axvline(x=strategies[0].underlying_price, color='#FFC107', linewidth=2,
               linestyle='--', label=f'Current: ${strategies[0].underlying_price:.2f}')

    ax.set_xlabel('Stock Price at Expiration ($)', fontsize=12, color='white')
    ax.set_ylabel('Profit / Loss ($)', fontsize=12, color='white')
    ax.set_title('Strategy Comparison', fontsize=14, color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3, color='#424242')
    ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#424242', labelcolor='white')

    plt.tight_layout()

    return fig
