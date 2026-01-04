#!/usr/bin/env python3
"""
Generate dashboard visualization images for README documentation.
Creates dark-themed images matching the Streamlit dashboard style.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import os

# Set dark theme
plt.style.use('dark_background')

# Color scheme matching Streamlit dark theme
COLORS = {
    'bg': '#0e1117',
    'card_bg': '#1a1d24',
    'border': '#2d3139',
    'text': '#fafafa',
    'text_secondary': '#8b949e',
    'accent': '#ff4b4b',
    'green': '#00d26a',
    'red': '#ff4b4b',
    'blue': '#0096ff',
    'yellow': '#ffc107',
    'purple': '#9d4edd',
}

def create_rounded_box(ax, x, y, width, height, color=COLORS['card_bg'],
                       border_color=COLORS['border'], text=None, text_color=COLORS['text'],
                       fontsize=10, text_y_offset=0):
    """Create a rounded rectangle box with optional text."""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.02",
                         facecolor=color, edgecolor=border_color, linewidth=1)
    ax.add_patch(box)
    if text:
        ax.text(x + width/2, y + height/2 + text_y_offset, text,
                ha='center', va='center', fontsize=fontsize, color=text_color,
                fontweight='bold')
    return box


def generate_dashboard_overview():
    """Generate Dashboard Overview visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(0.5, 9.5, '📈 Option-Strat Dashboard', fontsize=20, color=COLORS['text'],
            fontweight='bold')
    ax.text(0.5, 9.0, 'The Complete Options Trading Toolkit', fontsize=12,
            color=COLORS['text_secondary'])

    # Stock info card (top right)
    create_rounded_box(ax, 10, 8.2, 3.5, 1.5, text='AAPL\n$185.42 (+1.25%)', fontsize=14)

    # Navigation cards
    nav_items = [
        ('🔧 Strategy Builder', 'Build multi-leg strategies'),
        ('⚡ Optimizer', 'Find optimal strategies'),
        ('📊 Flow Scanner', 'Track unusual activity'),
        ('📈 Greeks Calculator', 'Analyze sensitivities'),
        ('📋 Templates', '50+ pre-built strategies'),
        ('🎯 P&L Heatmap', 'Visualize outcomes'),
    ]

    for i, (title, desc) in enumerate(nav_items):
        row = i // 3
        col = i % 3
        x = 0.5 + col * 4.5
        y = 5.5 - row * 2.5

        # Card background
        box = FancyBboxPatch((x, y), 4, 2,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             facecolor=COLORS['card_bg'], edgecolor=COLORS['border'], linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.2, y + 1.5, title, fontsize=12, color=COLORS['text'], fontweight='bold')
        ax.text(x + 0.2, y + 0.8, desc, fontsize=9, color=COLORS['text_secondary'])

    # Quick Stats section
    ax.text(0.5, 2.3, 'Quick Stats', fontsize=14, color=COLORS['text'], fontweight='bold')

    stats = [
        ('Strategies Analyzed', '1,247'),
        ('Avg. Return', '+23.5%'),
        ('Win Rate', '68.2%'),
        ('Active Trades', '12'),
    ]

    for i, (label, value) in enumerate(stats):
        x = 0.5 + i * 3.4
        box = FancyBboxPatch((x, 0.5), 3, 1.5,
                             boxstyle="round,pad=0.02,rounding_size=0.03",
                             facecolor=COLORS['card_bg'], edgecolor=COLORS['border'], linewidth=1)
        ax.add_patch(box)
        ax.text(x + 1.5, y=1.5, s=value, ha='center', fontsize=16,
                color=COLORS['green'] if '+' in value else COLORS['text'], fontweight='bold')
        ax.text(x + 1.5, y=0.9, s=label, ha='center', fontsize=9, color=COLORS['text_secondary'])

    plt.tight_layout()
    plt.savefig('docs/images/dashboard_overview.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("Generated: dashboard_overview.png")


def generate_strategy_builder():
    """Generate Strategy Builder visualization with payoff diagram."""
    fig = plt.figure(figsize=(14, 10), facecolor=COLORS['bg'])

    # Create grid for layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3,
                          left=0.05, right=0.95, top=0.92, bottom=0.05)

    # Title area
    ax_title = fig.add_subplot(gs[0, :2])
    ax_title.set_facecolor(COLORS['bg'])
    ax_title.axis('off')
    ax_title.text(0, 0.8, '🔧 Strategy Builder', fontsize=18, color=COLORS['text'],
                  fontweight='bold', transform=ax_title.transAxes)
    ax_title.text(0, 0.4, 'Build and analyze complex multi-leg options strategies',
                  fontsize=11, color=COLORS['text_secondary'], transform=ax_title.transAxes)

    # Strategy configuration panel
    ax_config = fig.add_subplot(gs[0, 2:])
    ax_config.set_facecolor(COLORS['card_bg'])
    ax_config.axis('off')
    for spine in ax_config.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['border'])

    config_text = """Strategy Type: Iron Condor
Days to Expiration: 30
Implied Volatility: 25%
Current Price: $150.00"""
    ax_config.text(0.05, 0.95, config_text, fontsize=10, color=COLORS['text'],
                   transform=ax_config.transAxes, va='top', family='monospace')

    # Legs display
    ax_legs = fig.add_subplot(gs[1, :2])
    ax_legs.set_facecolor(COLORS['card_bg'])
    ax_legs.axis('off')

    legs_data = [
        ('Put Buy', '$140', COLORS['green']),
        ('Put Sell', '$145', COLORS['red']),
        ('Call Sell', '$155', COLORS['red']),
        ('Call Buy', '$160', COLORS['green']),
    ]

    ax_legs.text(0.05, 0.9, 'Option Legs', fontsize=12, color=COLORS['text'],
                 fontweight='bold', transform=ax_legs.transAxes)

    for i, (leg_type, strike, color) in enumerate(legs_data):
        x = 0.05 + i * 0.24
        rect = FancyBboxPatch((x, 0.3), 0.2, 0.4,
                              boxstyle="round,pad=0.01,rounding_size=0.02",
                              facecolor=COLORS['bg'], edgecolor=color, linewidth=2,
                              transform=ax_legs.transAxes)
        ax_legs.add_patch(rect)
        ax_legs.text(x + 0.1, 0.6, leg_type, ha='center', fontsize=9,
                     color=COLORS['text'], transform=ax_legs.transAxes, fontweight='bold')
        ax_legs.text(x + 0.1, 0.45, strike, ha='center', fontsize=11,
                     color=color, transform=ax_legs.transAxes, fontweight='bold')

    # Strategy Metrics
    ax_metrics = fig.add_subplot(gs[1, 2:])
    ax_metrics.set_facecolor(COLORS['card_bg'])
    ax_metrics.axis('off')

    ax_metrics.text(0.05, 0.9, 'Strategy Metrics', fontsize=12, color=COLORS['text'],
                    fontweight='bold', transform=ax_metrics.transAxes)

    metrics = [
        ('Net Cost', 'CREDIT', '$245.00', COLORS['green']),
        ('Max Profit', '', '$245.00', COLORS['green']),
        ('Max Loss', '', '$255.00', COLORS['red']),
        ('P(Profit)', '', '68.5%', COLORS['blue']),
        ('Return/Risk', '', '96.1%', COLORS['yellow']),
    ]

    for i, (label, sublabel, value, color) in enumerate(metrics):
        x = 0.05 + (i % 5) * 0.19
        y = 0.55
        ax_metrics.text(x, y + 0.15, label, fontsize=8, color=COLORS['text_secondary'],
                        transform=ax_metrics.transAxes)
        if sublabel:
            ax_metrics.text(x, y + 0.05, sublabel, fontsize=7, color=color,
                            transform=ax_metrics.transAxes)
        ax_metrics.text(x, y - 0.1, value, fontsize=11, color=color,
                        transform=ax_metrics.transAxes, fontweight='bold')

    ax_metrics.text(0.05, 0.2, 'Breakevens: $145.00, $155.00', fontsize=10,
                    color=COLORS['text'], transform=ax_metrics.transAxes)

    # Payoff Diagram
    ax_payoff = fig.add_subplot(gs[2, :])
    ax_payoff.set_facecolor(COLORS['card_bg'])

    # Generate Iron Condor payoff
    stock_prices = np.linspace(120, 180, 200)
    put_buy_strike, put_sell_strike = 140, 145
    call_sell_strike, call_buy_strike = 155, 160

    # Calculate payoff for each leg
    put_buy = np.maximum(put_buy_strike - stock_prices, 0) - 1.50
    put_sell = -(np.maximum(put_sell_strike - stock_prices, 0) - 3.00)
    call_sell = -(np.maximum(stock_prices - call_sell_strike, 0) - 3.00)
    call_buy = np.maximum(stock_prices - call_buy_strike, 0) - 1.50

    total_payoff = (put_buy + put_sell + call_sell + call_buy) * 100

    # Create gradient fill
    ax_payoff.fill_between(stock_prices, total_payoff, 0,
                           where=(total_payoff > 0), color=COLORS['green'], alpha=0.3)
    ax_payoff.fill_between(stock_prices, total_payoff, 0,
                           where=(total_payoff <= 0), color=COLORS['red'], alpha=0.3)
    ax_payoff.plot(stock_prices, total_payoff, color=COLORS['blue'], linewidth=2.5)

    # Zero line
    ax_payoff.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', linewidth=0.5)

    # Current price line
    ax_payoff.axvline(x=150, color=COLORS['yellow'], linestyle='--', linewidth=1.5, label='Current: $150')

    # Breakeven lines
    ax_payoff.axvline(x=145, color=COLORS['text_secondary'], linestyle=':', linewidth=1, alpha=0.7)
    ax_payoff.axvline(x=155, color=COLORS['text_secondary'], linestyle=':', linewidth=1, alpha=0.7)

    ax_payoff.set_xlabel('Stock Price at Expiration ($)', color=COLORS['text'], fontsize=10)
    ax_payoff.set_ylabel('Profit/Loss ($)', color=COLORS['text'], fontsize=10)
    ax_payoff.set_title('P&L at Expiration', color=COLORS['text'], fontsize=12, fontweight='bold')
    ax_payoff.tick_params(colors=COLORS['text_secondary'])
    ax_payoff.grid(True, alpha=0.2, color=COLORS['border'])
    ax_payoff.legend(loc='upper right', facecolor=COLORS['card_bg'],
                     edgecolor=COLORS['border'], labelcolor=COLORS['text'])

    for spine in ax_payoff.spines.values():
        spine.set_color(COLORS['border'])

    plt.savefig('docs/images/strategy_builder.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("Generated: strategy_builder.png")


def generate_strategy_optimizer():
    """Generate Strategy Optimizer visualization."""
    fig = plt.figure(figsize=(14, 10), facecolor=COLORS['bg'])

    gs = fig.add_gridspec(3, 2, hspace=0.25, wspace=0.2,
                          left=0.05, right=0.95, top=0.92, bottom=0.05)

    # Title
    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.set_facecolor(COLORS['bg'])
    ax_title.axis('off')
    ax_title.text(0, 0.8, '⚡ Strategy Optimizer', fontsize=18, color=COLORS['text'],
                  fontweight='bold', transform=ax_title.transAxes)
    ax_title.text(0, 0.3, 'Find optimal strategies based on your criteria',
                  fontsize=11, color=COLORS['text_secondary'], transform=ax_title.transAxes)

    # Configuration
    ax_config = fig.add_subplot(gs[0, 1])
    ax_config.set_facecolor(COLORS['card_bg'])
    ax_config.axis('off')

    config_items = [
        ('Symbol:', 'AAPL'),
        ('Price:', '$185.42'),
        ('Optimize By:', 'Risk-Adjusted Return'),
        ('Max Risk:', '$500'),
    ]

    for i, (label, value) in enumerate(config_items):
        y = 0.85 - i * 0.22
        ax_config.text(0.05, y, label, fontsize=10, color=COLORS['text_secondary'],
                       transform=ax_config.transAxes)
        ax_config.text(0.4, y, value, fontsize=10, color=COLORS['text'],
                       transform=ax_config.transAxes, fontweight='bold')

    # Results table
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.set_facecolor(COLORS['card_bg'])
    ax_table.axis('off')

    ax_table.text(0.02, 0.95, 'Optimization Results', fontsize=14, color=COLORS['text'],
                  fontweight='bold', transform=ax_table.transAxes)

    # Table headers
    headers = ['Rank', 'Strategy', 'Max Profit', 'Max Loss', 'P(Profit)', 'Return/Risk', 'Score']
    header_x = [0.02, 0.08, 0.28, 0.42, 0.56, 0.70, 0.85]

    for x, header in zip(header_x, headers):
        ax_table.text(x, 0.82, header, fontsize=9, color=COLORS['text_secondary'],
                      transform=ax_table.transAxes, fontweight='bold')

    # Horizontal line (using plot instead of axhline for axes transform)
    ax_table.plot([0.02, 0.98], [0.78, 0.78], color=COLORS['border'],
                  linewidth=1, transform=ax_table.transAxes)

    # Table data
    results = [
        ('🥇', 'Iron Condor 145/150/160/165', '$420', '$80', '72.3%', '525.0%', '94.2'),
        ('🥈', 'Bull Put Spread 175/180', '$380', '$120', '68.5%', '316.7%', '87.5'),
        ('🥉', 'Iron Butterfly 180/185/190', '$650', '$350', '45.2%', '185.7%', '76.3'),
        ('4', 'Bear Call Spread 190/195', '$290', '$210', '71.8%', '138.1%', '72.1'),
        ('5', 'Jade Lizard 180/185/190', '$520', '$480', '62.4%', '108.3%', '68.9'),
    ]

    for i, row in enumerate(results):
        y = 0.68 - i * 0.13
        colors = [COLORS['yellow'] if i == 0 else COLORS['text'],
                  COLORS['text'], COLORS['green'], COLORS['red'],
                  COLORS['blue'], COLORS['green'], COLORS['yellow']]

        for j, (x, val, color) in enumerate(zip(header_x, row, colors)):
            ax_table.text(x, y, val, fontsize=9, color=color,
                          transform=ax_table.transAxes,
                          fontweight='bold' if j in [0, 6] else 'normal')

    # Comparison chart
    ax_chart = fig.add_subplot(gs[2, :])
    ax_chart.set_facecolor(COLORS['card_bg'])

    strategies = ['Iron Condor', 'Bull Put', 'Iron Butterfly', 'Bear Call', 'Jade Lizard']
    x_pos = np.arange(len(strategies))
    width = 0.25

    max_profit = [420, 380, 650, 290, 520]
    max_loss = [80, 120, 350, 210, 480]
    scores = [94.2, 87.5, 76.3, 72.1, 68.9]

    bars1 = ax_chart.bar(x_pos - width, max_profit, width, label='Max Profit',
                         color=COLORS['green'], alpha=0.8)
    bars2 = ax_chart.bar(x_pos, max_loss, width, label='Max Loss',
                         color=COLORS['red'], alpha=0.8)

    ax2 = ax_chart.twinx()
    ax2.plot(x_pos, scores, 'o-', color=COLORS['yellow'], linewidth=2,
             markersize=8, label='Score')
    ax2.set_ylabel('Optimization Score', color=COLORS['yellow'], fontsize=10)
    ax2.tick_params(axis='y', colors=COLORS['yellow'])
    ax2.set_ylim(60, 100)

    ax_chart.set_xlabel('Strategy', color=COLORS['text'], fontsize=10)
    ax_chart.set_ylabel('Amount ($)', color=COLORS['text'], fontsize=10)
    ax_chart.set_title('Strategy Comparison', color=COLORS['text'], fontsize=12, fontweight='bold')
    ax_chart.set_xticks(x_pos)
    ax_chart.set_xticklabels(strategies, color=COLORS['text'], fontsize=9)
    ax_chart.tick_params(colors=COLORS['text_secondary'])
    ax_chart.legend(loc='upper left', facecolor=COLORS['card_bg'],
                    edgecolor=COLORS['border'], labelcolor=COLORS['text'])
    ax2.legend(loc='upper right', facecolor=COLORS['card_bg'],
               edgecolor=COLORS['border'], labelcolor=COLORS['text'])
    ax_chart.grid(True, alpha=0.2, color=COLORS['border'], axis='y')

    for spine in ax_chart.spines.values():
        spine.set_color(COLORS['border'])
    for spine in ax2.spines.values():
        spine.set_color(COLORS['border'])

    plt.savefig('docs/images/strategy_optimizer.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("Generated: strategy_optimizer.png")


def generate_flow_scanner():
    """Generate Options Flow Scanner visualization."""
    fig = plt.figure(figsize=(14, 10), facecolor=COLORS['bg'])

    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25,
                          left=0.05, right=0.95, top=0.92, bottom=0.05)

    # Title
    ax_title = fig.add_subplot(gs[0, :2])
    ax_title.set_facecolor(COLORS['bg'])
    ax_title.axis('off')
    ax_title.text(0, 0.7, '📊 Options Flow Scanner', fontsize=18, color=COLORS['text'],
                  fontweight='bold', transform=ax_title.transAxes)
    ax_title.text(0, 0.2, 'Track unusual options activity and smart money flow',
                  fontsize=11, color=COLORS['text_secondary'], transform=ax_title.transAxes)

    # Summary stats
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.set_facecolor(COLORS['card_bg'])
    ax_stats.axis('off')

    stats = [
        ('Total Trades', '2,847', COLORS['text']),
        ('Bullish Flow', '$12.4M', COLORS['green']),
        ('Bearish Flow', '$8.2M', COLORS['red']),
        ('Put/Call Ratio', '0.67', COLORS['blue']),
    ]

    for i, (label, value, color) in enumerate(stats):
        y = 0.85 - i * 0.22
        ax_stats.text(0.1, y, label, fontsize=9, color=COLORS['text_secondary'],
                      transform=ax_stats.transAxes)
        ax_stats.text(0.7, y, value, fontsize=11, color=color,
                      transform=ax_stats.transAxes, fontweight='bold', ha='right')

    # Flow table
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.set_facecolor(COLORS['card_bg'])
    ax_table.axis('off')

    ax_table.text(0.02, 0.95, 'Unusual Options Activity', fontsize=14, color=COLORS['text'],
                  fontweight='bold', transform=ax_table.transAxes)

    headers = ['Time', 'Symbol', 'Type', 'Strike', 'Exp', 'Size', 'Premium', 'Sentiment']
    header_x = [0.02, 0.10, 0.20, 0.30, 0.42, 0.54, 0.66, 0.82]

    for x, header in zip(header_x, headers):
        ax_table.text(x, 0.82, header, fontsize=9, color=COLORS['text_secondary'],
                      transform=ax_table.transAxes, fontweight='bold')

    ax_table.plot([0.02, 0.98], [0.78, 0.78], color=COLORS['border'],
                  linewidth=1, transform=ax_table.transAxes)

    flows = [
        ('14:32:15', 'AAPL', 'CALL', '$190', '01/17', '5,000', '$1.2M', '🟢 BULLISH'),
        ('14:28:42', 'TSLA', 'PUT', '$240', '01/10', '3,200', '$890K', '🔴 BEARISH'),
        ('14:25:18', 'NVDA', 'CALL', '$500', '01/24', '2,500', '$2.1M', '🟢 BULLISH'),
        ('14:21:33', 'META', 'CALL', '$360', '01/17', '4,100', '$1.5M', '🟢 BULLISH'),
        ('14:18:55', 'SPY', 'PUT', '$470', '01/10', '8,000', '$3.2M', '🔴 BEARISH'),
        ('14:15:22', 'AMZN', 'CALL', '$195', '01/24', '2,800', '$980K', '🟡 NEUTRAL'),
    ]

    for i, row in enumerate(flows):
        y = 0.68 - i * 0.11
        for j, (x, val) in enumerate(zip(header_x, row)):
            if 'BULLISH' in val:
                color = COLORS['green']
            elif 'BEARISH' in val:
                color = COLORS['red']
            elif 'CALL' in val:
                color = COLORS['green']
            elif 'PUT' in val:
                color = COLORS['red']
            else:
                color = COLORS['text']
            ax_table.text(x, y, val, fontsize=9, color=color, transform=ax_table.transAxes)

    # Sentiment pie chart
    ax_pie = fig.add_subplot(gs[2, 0])
    ax_pie.set_facecolor(COLORS['card_bg'])

    sentiments = [55, 35, 10]
    labels = ['Bullish', 'Bearish', 'Neutral']
    colors_pie = [COLORS['green'], COLORS['red'], COLORS['yellow']]

    wedges, texts, autotexts = ax_pie.pie(sentiments, labels=labels, autopct='%1.1f%%',
                                           colors=colors_pie, startangle=90,
                                           textprops={'color': COLORS['text']})
    for autotext in autotexts:
        autotext.set_color(COLORS['text'])
        autotext.set_fontweight('bold')
    ax_pie.set_title('Sentiment Breakdown', color=COLORS['text'], fontsize=11, fontweight='bold')

    # Volume by symbol
    ax_vol = fig.add_subplot(gs[2, 1])
    ax_vol.set_facecolor(COLORS['card_bg'])

    symbols = ['AAPL', 'TSLA', 'NVDA', 'META', 'SPY']
    volumes = [5000, 3200, 2500, 4100, 8000]
    colors_bar = [COLORS['green'], COLORS['red'], COLORS['green'],
                  COLORS['green'], COLORS['red']]

    bars = ax_vol.barh(symbols, volumes, color=colors_bar, alpha=0.8)
    ax_vol.set_xlabel('Volume', color=COLORS['text'], fontsize=9)
    ax_vol.set_title('Volume by Symbol', color=COLORS['text'], fontsize=11, fontweight='bold')
    ax_vol.tick_params(colors=COLORS['text_secondary'])
    ax_vol.grid(True, alpha=0.2, color=COLORS['border'], axis='x')

    for spine in ax_vol.spines.values():
        spine.set_color(COLORS['border'])

    # Premium timeline
    ax_time = fig.add_subplot(gs[2, 2])
    ax_time.set_facecolor(COLORS['card_bg'])

    times = ['9:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00', '13:30', '14:00']
    bullish_prem = [2.1, 3.5, 4.2, 3.8, 5.1, 4.5, 6.2, 7.8, 9.5, 12.4]
    bearish_prem = [1.8, 2.2, 2.8, 3.1, 4.2, 4.8, 5.5, 6.2, 7.1, 8.2]

    ax_time.fill_between(range(len(times)), bullish_prem, alpha=0.3, color=COLORS['green'])
    ax_time.fill_between(range(len(times)), bearish_prem, alpha=0.3, color=COLORS['red'])
    ax_time.plot(bullish_prem, color=COLORS['green'], linewidth=2, label='Bullish')
    ax_time.plot(bearish_prem, color=COLORS['red'], linewidth=2, label='Bearish')

    ax_time.set_xticks(range(len(times)))
    ax_time.set_xticklabels(times, rotation=45, fontsize=7)
    ax_time.set_ylabel('Premium ($M)', color=COLORS['text'], fontsize=9)
    ax_time.set_title('Premium Flow', color=COLORS['text'], fontsize=11, fontweight='bold')
    ax_time.tick_params(colors=COLORS['text_secondary'])
    ax_time.legend(loc='upper left', facecolor=COLORS['card_bg'],
                   edgecolor=COLORS['border'], labelcolor=COLORS['text'], fontsize=8)
    ax_time.grid(True, alpha=0.2, color=COLORS['border'])

    for spine in ax_time.spines.values():
        spine.set_color(COLORS['border'])

    plt.savefig('docs/images/flow_scanner.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("Generated: flow_scanner.png")


def generate_greeks_calculator():
    """Generate Greeks Calculator visualization."""
    fig = plt.figure(figsize=(14, 10), facecolor=COLORS['bg'])

    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25,
                          left=0.05, right=0.95, top=0.92, bottom=0.05)

    # Title
    ax_title = fig.add_subplot(gs[0, :2])
    ax_title.set_facecolor(COLORS['bg'])
    ax_title.axis('off')
    ax_title.text(0, 0.7, '📈 Greeks Calculator', fontsize=18, color=COLORS['text'],
                  fontweight='bold', transform=ax_title.transAxes)
    ax_title.text(0, 0.2, 'Analyze option sensitivities and risk metrics',
                  fontsize=11, color=COLORS['text_secondary'], transform=ax_title.transAxes)

    # Position Greeks display
    ax_greeks = fig.add_subplot(gs[0, 2])
    ax_greeks.set_facecolor(COLORS['card_bg'])
    ax_greeks.axis('off')

    ax_greeks.text(0.5, 0.95, 'Position Greeks', fontsize=11, color=COLORS['text'],
                   fontweight='bold', transform=ax_greeks.transAxes, ha='center')

    greeks = [
        ('Delta (Δ)', '-0.0234', COLORS['blue']),
        ('Gamma (Γ)', '-0.0089', COLORS['purple']),
        ('Theta (Θ)', '+0.0456', COLORS['green']),
        ('Vega (ν)', '-0.1234', COLORS['yellow']),
        ('Rho (ρ)', '+0.0012', COLORS['text']),
    ]

    for i, (name, value, color) in enumerate(greeks):
        col = i % 3
        row = i // 3
        x = 0.1 + col * 0.33
        y = 0.65 - row * 0.35
        ax_greeks.text(x, y + 0.1, name, fontsize=8, color=COLORS['text_secondary'],
                       transform=ax_greeks.transAxes)
        ax_greeks.text(x, y - 0.05, value, fontsize=12, color=color,
                       transform=ax_greeks.transAxes, fontweight='bold')

    # Delta chart
    ax_delta = fig.add_subplot(gs[1, 0])
    ax_delta.set_facecolor(COLORS['card_bg'])

    stock_prices = np.linspace(130, 170, 100)
    delta = 1 / (1 + np.exp(-(stock_prices - 150) / 5)) - 0.5

    ax_delta.plot(stock_prices, delta, color=COLORS['blue'], linewidth=2)
    ax_delta.axhline(y=0, color=COLORS['text_secondary'], linestyle='--', alpha=0.5)
    ax_delta.axvline(x=150, color=COLORS['yellow'], linestyle='--', alpha=0.7)
    ax_delta.fill_between(stock_prices, delta, 0, where=(delta > 0),
                          color=COLORS['green'], alpha=0.2)
    ax_delta.fill_between(stock_prices, delta, 0, where=(delta < 0),
                          color=COLORS['red'], alpha=0.2)
    ax_delta.set_title('Delta (Δ)', color=COLORS['text'], fontsize=11, fontweight='bold')
    ax_delta.set_xlabel('Stock Price', color=COLORS['text_secondary'], fontsize=9)
    ax_delta.tick_params(colors=COLORS['text_secondary'])
    ax_delta.grid(True, alpha=0.2, color=COLORS['border'])
    for spine in ax_delta.spines.values():
        spine.set_color(COLORS['border'])

    # Gamma chart
    ax_gamma = fig.add_subplot(gs[1, 1])
    ax_gamma.set_facecolor(COLORS['card_bg'])

    gamma = 0.1 * np.exp(-((stock_prices - 150) ** 2) / 100)

    ax_gamma.plot(stock_prices, gamma, color=COLORS['purple'], linewidth=2)
    ax_gamma.axvline(x=150, color=COLORS['yellow'], linestyle='--', alpha=0.7)
    ax_gamma.fill_between(stock_prices, gamma, alpha=0.3, color=COLORS['purple'])
    ax_gamma.set_title('Gamma (Γ)', color=COLORS['text'], fontsize=11, fontweight='bold')
    ax_gamma.set_xlabel('Stock Price', color=COLORS['text_secondary'], fontsize=9)
    ax_gamma.tick_params(colors=COLORS['text_secondary'])
    ax_gamma.grid(True, alpha=0.2, color=COLORS['border'])
    for spine in ax_gamma.spines.values():
        spine.set_color(COLORS['border'])

    # Theta chart
    ax_theta = fig.add_subplot(gs[1, 2])
    ax_theta.set_facecolor(COLORS['card_bg'])

    days = np.linspace(30, 0, 100)
    theta = -0.05 * np.exp(-days / 10)

    ax_theta.plot(days, theta, color=COLORS['green'], linewidth=2)
    ax_theta.fill_between(days, theta, alpha=0.3, color=COLORS['green'])
    ax_theta.set_title('Theta Decay (Θ)', color=COLORS['text'], fontsize=11, fontweight='bold')
    ax_theta.set_xlabel('Days to Expiration', color=COLORS['text_secondary'], fontsize=9)
    ax_theta.tick_params(colors=COLORS['text_secondary'])
    ax_theta.grid(True, alpha=0.2, color=COLORS['border'])
    ax_theta.invert_xaxis()
    for spine in ax_theta.spines.values():
        spine.set_color(COLORS['border'])

    # Vega chart
    ax_vega = fig.add_subplot(gs[2, 0])
    ax_vega.set_facecolor(COLORS['card_bg'])

    iv = np.linspace(0.1, 0.5, 100)
    vega = 0.5 * np.exp(-((iv - 0.25) ** 2) / 0.02)

    ax_vega.plot(iv * 100, vega, color=COLORS['yellow'], linewidth=2)
    ax_vega.axvline(x=25, color=COLORS['blue'], linestyle='--', alpha=0.7, label='Current IV')
    ax_vega.fill_between(iv * 100, vega, alpha=0.3, color=COLORS['yellow'])
    ax_vega.set_title('Vega (ν)', color=COLORS['text'], fontsize=11, fontweight='bold')
    ax_vega.set_xlabel('Implied Volatility (%)', color=COLORS['text_secondary'], fontsize=9)
    ax_vega.tick_params(colors=COLORS['text_secondary'])
    ax_vega.legend(loc='upper right', facecolor=COLORS['card_bg'],
                   edgecolor=COLORS['border'], labelcolor=COLORS['text'], fontsize=8)
    ax_vega.grid(True, alpha=0.2, color=COLORS['border'])
    for spine in ax_vega.spines.values():
        spine.set_color(COLORS['border'])

    # Greeks surface
    ax_surface = fig.add_subplot(gs[2, 1:], projection='3d')
    ax_surface.set_facecolor(COLORS['card_bg'])

    S = np.linspace(130, 170, 30)
    T = np.linspace(1, 30, 30)
    S, T = np.meshgrid(S, T)

    # Simplified delta surface
    Z = 1 / (1 + np.exp(-(S - 150) / (5 + T/10))) - 0.5

    surf = ax_surface.plot_surface(S, T, Z, cmap='RdYlGn', alpha=0.8,
                                    linewidth=0, antialiased=True)
    ax_surface.set_xlabel('Stock Price', color=COLORS['text_secondary'], fontsize=8)
    ax_surface.set_ylabel('Days to Exp', color=COLORS['text_secondary'], fontsize=8)
    ax_surface.set_zlabel('Delta', color=COLORS['text_secondary'], fontsize=8)
    ax_surface.set_title('Delta Surface', color=COLORS['text'], fontsize=11, fontweight='bold')
    ax_surface.tick_params(colors=COLORS['text_secondary'])
    ax_surface.xaxis.pane.fill = False
    ax_surface.yaxis.pane.fill = False
    ax_surface.zaxis.pane.fill = False

    plt.savefig('docs/images/greeks_calculator.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("Generated: greeks_calculator.png")


def generate_strategy_templates():
    """Generate Strategy Templates visualization."""
    fig = plt.figure(figsize=(14, 10), facecolor=COLORS['bg'])

    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.2,
                          left=0.05, right=0.95, top=0.92, bottom=0.05)

    # Title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor(COLORS['bg'])
    ax_title.axis('off')
    ax_title.text(0.5, 0.7, '📋 Strategy Templates', fontsize=18, color=COLORS['text'],
                  fontweight='bold', transform=ax_title.transAxes, ha='center')
    ax_title.text(0.5, 0.2, '50+ Pre-built Options Strategies Organized by Market Outlook',
                  fontsize=11, color=COLORS['text_secondary'], transform=ax_title.transAxes, ha='center')

    # Strategy categories
    categories = [
        ('🟢 BULLISH', ['Long Call', 'Bull Call Spread', 'Bull Put Spread',
                        'Call Ratio Backspread', 'Synthetic Long'], COLORS['green']),
        ('🔴 BEARISH', ['Long Put', 'Bear Put Spread', 'Bear Call Spread',
                        'Put Ratio Backspread', 'Synthetic Short'], COLORS['red']),
        ('🟡 NEUTRAL', ['Iron Condor', 'Iron Butterfly', 'Short Straddle',
                        'Short Strangle', 'Calendar Spread'], COLORS['yellow']),
        ('🟣 VOLATILE', ['Long Straddle', 'Long Strangle', 'Reverse Iron Condor',
                         'Reverse Iron Butterfly', 'Double Diagonal'], COLORS['purple']),
        ('🔵 INCOME', ['Covered Call', 'Cash-Secured Put', 'Poor Man\'s CC',
                       'Wheel Strategy', 'Jade Lizard'], COLORS['blue']),
        ('⚪ HEDGE', ['Protective Put', 'Collar', 'Married Put',
                      'Risk Reversal', 'Fence'], COLORS['text']),
    ]

    for i, (title, strategies, color) in enumerate(categories):
        row = 1 + i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(COLORS['card_bg'])
        ax.axis('off')

        # Category title
        ax.text(0.5, 0.95, title, fontsize=12, color=color, fontweight='bold',
                transform=ax.transAxes, ha='center')

        # Strategy list
        for j, strategy in enumerate(strategies):
            y = 0.75 - j * 0.15
            ax.text(0.1, y, f"• {strategy}", fontsize=9, color=COLORS['text'],
                    transform=ax.transAxes)

        # Border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2)

    plt.savefig('docs/images/strategy_templates.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("Generated: strategy_templates.png")


def generate_pnl_heatmap():
    """Generate P&L Heatmap visualization."""
    fig = plt.figure(figsize=(14, 10), facecolor=COLORS['bg'])

    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2,
                          left=0.08, right=0.95, top=0.90, bottom=0.08)

    # Title
    fig.suptitle('🎯 P&L Heatmap Analysis', fontsize=18, color=COLORS['text'],
                 fontweight='bold', y=0.96)

    # Main heatmap
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_heat.set_facecolor(COLORS['card_bg'])

    # Generate P&L data for Iron Condor
    stock_prices = np.linspace(130, 170, 20)
    days_to_exp = np.linspace(30, 0, 15)

    S, D = np.meshgrid(stock_prices, days_to_exp)

    # Simplified Iron Condor P&L calculation
    pnl = np.zeros_like(S)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            s, d = S[i, j], D[i, j]
            time_factor = d / 30
            if 145 <= s <= 155:
                pnl[i, j] = 300 * (1 - time_factor * 0.3)
            elif 140 <= s < 145 or 155 < s <= 160:
                pnl[i, j] = 300 - abs(s - 150) * 30 * (1 - time_factor * 0.5)
            else:
                pnl[i, j] = -200 - abs(s - 150) * 10

    im = ax_heat.imshow(pnl, aspect='auto', cmap='RdYlGn',
                         extent=[130, 170, 0, 30], origin='lower')

    # Contour lines
    contours = ax_heat.contour(stock_prices, days_to_exp, pnl,
                                levels=[-200, -100, 0, 100, 200, 300],
                                colors='white', linewidths=0.5, alpha=0.5)
    ax_heat.clabel(contours, inline=True, fontsize=8, fmt='$%.0f')

    # Breakeven lines
    ax_heat.axvline(x=145, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    ax_heat.axvline(x=155, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    ax_heat.axvline(x=150, color=COLORS['yellow'], linestyle='-', linewidth=2)

    ax_heat.set_xlabel('Stock Price at Expiration ($)', color=COLORS['text'], fontsize=10)
    ax_heat.set_ylabel('Days to Expiration', color=COLORS['text'], fontsize=10)
    ax_heat.set_title('Iron Condor P&L Heatmap', color=COLORS['text'], fontsize=12, fontweight='bold')
    ax_heat.tick_params(colors=COLORS['text_secondary'])

    cbar = plt.colorbar(im, ax=ax_heat, label='Profit/Loss ($)')
    cbar.ax.yaxis.label.set_color(COLORS['text'])
    cbar.ax.tick_params(colors=COLORS['text_secondary'])

    for spine in ax_heat.spines.values():
        spine.set_color(COLORS['border'])

    # P&L by price (top right)
    ax_price = fig.add_subplot(gs[0, 1])
    ax_price.set_facecolor(COLORS['card_bg'])

    pnl_at_exp = pnl[-1, :]
    ax_price.fill_between(stock_prices, pnl_at_exp, 0,
                          where=(pnl_at_exp > 0), color=COLORS['green'], alpha=0.4)
    ax_price.fill_between(stock_prices, pnl_at_exp, 0,
                          where=(pnl_at_exp <= 0), color=COLORS['red'], alpha=0.4)
    ax_price.plot(stock_prices, pnl_at_exp, color=COLORS['blue'], linewidth=2)
    ax_price.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', linewidth=0.5)
    ax_price.axvline(x=150, color=COLORS['yellow'], linestyle='--', linewidth=1.5)

    ax_price.set_xlabel('Stock Price ($)', color=COLORS['text'], fontsize=9)
    ax_price.set_ylabel('P&L ($)', color=COLORS['text'], fontsize=9)
    ax_price.set_title('P&L at Expiration', color=COLORS['text'], fontsize=11, fontweight='bold')
    ax_price.tick_params(colors=COLORS['text_secondary'])
    ax_price.grid(True, alpha=0.2, color=COLORS['border'])
    for spine in ax_price.spines.values():
        spine.set_color(COLORS['border'])

    # P&L by time (bottom right)
    ax_time = fig.add_subplot(gs[1, 1])
    ax_time.set_facecolor(COLORS['card_bg'])

    # P&L at different price levels over time
    for price, label, color in [(145, '$145', COLORS['yellow']),
                                 (150, '$150 (ATM)', COLORS['green']),
                                 (155, '$155', COLORS['yellow']),
                                 (140, '$140', COLORS['red'])]:
        idx = np.argmin(np.abs(stock_prices - price))
        ax_time.plot(days_to_exp, pnl[:, idx], label=label, color=color, linewidth=2)

    ax_time.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', linewidth=0.5)
    ax_time.set_xlabel('Days to Expiration', color=COLORS['text'], fontsize=9)
    ax_time.set_ylabel('P&L ($)', color=COLORS['text'], fontsize=9)
    ax_time.set_title('P&L Over Time', color=COLORS['text'], fontsize=11, fontweight='bold')
    ax_time.tick_params(colors=COLORS['text_secondary'])
    ax_time.legend(loc='upper left', facecolor=COLORS['card_bg'],
                   edgecolor=COLORS['border'], labelcolor=COLORS['text'], fontsize=8)
    ax_time.grid(True, alpha=0.2, color=COLORS['border'])
    ax_time.invert_xaxis()
    for spine in ax_time.spines.values():
        spine.set_color(COLORS['border'])

    plt.savefig('docs/images/pnl_heatmap.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("Generated: pnl_heatmap.png")


def generate_probability_distribution():
    """Generate Probability Distribution visualization."""
    fig = plt.figure(figsize=(14, 10), facecolor=COLORS['bg'])

    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2,
                          left=0.08, right=0.95, top=0.90, bottom=0.08)

    # Title
    fig.suptitle('📊 Probability Analysis', fontsize=18, color=COLORS['text'],
                 fontweight='bold', y=0.96)

    # Main distribution
    ax_dist = fig.add_subplot(gs[0, :])
    ax_dist.set_facecolor(COLORS['card_bg'])

    # Generate lognormal distribution for stock prices
    current_price = 150
    volatility = 0.25
    days = 30

    np.random.seed(42)
    returns = np.random.normal(0, volatility * np.sqrt(days/252), 10000)
    future_prices = current_price * np.exp(returns)

    # Create histogram
    counts, bins, patches = ax_dist.hist(future_prices, bins=50, density=True,
                                          alpha=0.7, color=COLORS['blue'])

    # Color profitable/unprofitable regions
    for i, (patch, left_edge) in enumerate(zip(patches, bins[:-1])):
        if 145 <= left_edge <= 155:
            patch.set_facecolor(COLORS['green'])
        elif left_edge < 140 or left_edge > 160:
            patch.set_facecolor(COLORS['red'])

    # Breakeven lines
    ax_dist.axvline(x=145, color=COLORS['yellow'], linestyle='--', linewidth=2,
                     label='Lower Breakeven: $145')
    ax_dist.axvline(x=155, color=COLORS['yellow'], linestyle='--', linewidth=2,
                     label='Upper Breakeven: $155')
    ax_dist.axvline(x=150, color='white', linestyle='-', linewidth=2,
                     label='Current Price: $150')

    ax_dist.set_xlabel('Stock Price at Expiration ($)', color=COLORS['text'], fontsize=10)
    ax_dist.set_ylabel('Probability Density', color=COLORS['text'], fontsize=10)
    ax_dist.set_title('Price Distribution at Expiration (30 Days, 25% IV)',
                       color=COLORS['text'], fontsize=12, fontweight='bold')
    ax_dist.tick_params(colors=COLORS['text_secondary'])
    ax_dist.legend(loc='upper right', facecolor=COLORS['card_bg'],
                    edgecolor=COLORS['border'], labelcolor=COLORS['text'])
    ax_dist.grid(True, alpha=0.2, color=COLORS['border'])
    for spine in ax_dist.spines.values():
        spine.set_color(COLORS['border'])

    # Probability stats
    ax_stats = fig.add_subplot(gs[1, 0])
    ax_stats.set_facecolor(COLORS['card_bg'])
    ax_stats.axis('off')

    ax_stats.text(0.5, 0.95, 'Probability Statistics', fontsize=14, color=COLORS['text'],
                  fontweight='bold', transform=ax_stats.transAxes, ha='center')

    prob_profit = np.mean((future_prices >= 145) & (future_prices <= 155)) * 100
    prob_max_profit = np.mean((future_prices >= 147) & (future_prices <= 153)) * 100
    prob_loss = 100 - prob_profit
    expected_value = np.mean(np.where((future_prices >= 145) & (future_prices <= 155),
                                       300, -200))

    stats = [
        ('Probability of Profit', f'{prob_profit:.1f}%', COLORS['green']),
        ('Probability of Max Profit', f'{prob_max_profit:.1f}%', COLORS['green']),
        ('Probability of Loss', f'{prob_loss:.1f}%', COLORS['red']),
        ('Expected Value', f'${expected_value:.2f}',
         COLORS['green'] if expected_value > 0 else COLORS['red']),
        ('50th Percentile', f'${np.percentile(future_prices, 50):.2f}', COLORS['text']),
        ('1σ Range', f'${np.percentile(future_prices, 16):.0f} - ${np.percentile(future_prices, 84):.0f}',
         COLORS['blue']),
    ]

    for i, (label, value, color) in enumerate(stats):
        y = 0.75 - i * 0.12
        ax_stats.text(0.1, y, label + ':', fontsize=10, color=COLORS['text_secondary'],
                      transform=ax_stats.transAxes)
        ax_stats.text(0.9, y, value, fontsize=12, color=color, fontweight='bold',
                      transform=ax_stats.transAxes, ha='right')

    # Outcome pie chart
    ax_pie = fig.add_subplot(gs[1, 1])
    ax_pie.set_facecolor(COLORS['card_bg'])

    outcomes = [prob_profit, prob_loss]
    labels = ['Profit Zone\n(Max: $300)', 'Loss Zone\n(Max: $500)']
    colors_pie = [COLORS['green'], COLORS['red']]
    explode = (0.02, 0)

    wedges, texts, autotexts = ax_pie.pie(outcomes, labels=labels, autopct='%1.1f%%',
                                           colors=colors_pie, explode=explode,
                                           startangle=90, textprops={'color': COLORS['text']})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    ax_pie.set_title('Outcome Probabilities', color=COLORS['text'],
                      fontsize=12, fontweight='bold')

    plt.savefig('docs/images/probability_distribution.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print("Generated: probability_distribution.png")


def main():
    """Generate all dashboard images."""
    print("Generating dashboard visualization images...")
    print("=" * 50)

    os.makedirs('docs/images', exist_ok=True)

    generate_dashboard_overview()
    generate_strategy_builder()
    generate_strategy_optimizer()
    generate_flow_scanner()
    generate_greeks_calculator()
    generate_strategy_templates()
    generate_pnl_heatmap()
    generate_probability_distribution()

    print("=" * 50)
    print("All images generated successfully!")
    print(f"Images saved to: docs/images/")


if __name__ == "__main__":
    main()
