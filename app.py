"""
Option-Strat Web Dashboard

A comprehensive web interface for options analysis built with Streamlit.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Option-Strat | Options Trading Toolkit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #2196F3, #00BCD4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #0f3460;
    }
    .profit {
        color: #00C853;
    }
    .loss {
        color: #FF1744;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Import our modules
import sys
sys.path.insert(0, 'src')

try:
    from optionstrat.utils.data import get_stock_data, get_option_chain
    from optionstrat.strategies.builder import StrategyBuilder
    from optionstrat.strategies.templates import StrategyTemplates, MarketOutlook
    from optionstrat.optimizer.optimizer import StrategyOptimizer, OptimizationGoal
    from optionstrat.flow.scanner import FlowScanner, FlowFilter, Sentiment
    from optionstrat.models.option import Option
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    import_error = str(e)


def main():
    """Main application."""

    # Sidebar
    st.sidebar.markdown('<p class="main-header">Option-Strat</p>', unsafe_allow_html=True)
    st.sidebar.markdown("### The Options Trader's Toolkit")

    if not IMPORTS_OK:
        st.error(f"Import error: {import_error}")
        st.info("Make sure all dependencies are installed: pip install -r requirements.txt")
        return

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📊 Strategy Builder", "🎯 Optimizer", "📈 Options Flow",
         "🔬 Greeks Calculator", "📋 Templates", "⚙️ Settings"]
    )

    # Symbol input in sidebar
    st.sidebar.markdown("---")
    symbol = st.sidebar.text_input("Symbol", value="AAPL", max_chars=5).upper()

    # Get stock data
    with st.spinner("Loading market data..."):
        stock_data = get_stock_data(symbol)

    # Display stock info in sidebar
    st.sidebar.markdown(f"### {symbol}")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Price", f"${stock_data['price']:.2f}")
    change_pct = stock_data.get('change_percent', 0)
    col2.metric("Change", f"{change_pct:+.2f}%")

    # Page routing
    if page == "🏠 Home":
        show_home(stock_data)
    elif page == "📊 Strategy Builder":
        show_strategy_builder(symbol, stock_data)
    elif page == "🎯 Optimizer":
        show_optimizer(symbol, stock_data)
    elif page == "📈 Options Flow":
        show_flow()
    elif page == "🔬 Greeks Calculator":
        show_greeks_calculator(symbol, stock_data)
    elif page == "📋 Templates":
        show_templates(symbol, stock_data)
    elif page == "⚙️ Settings":
        show_settings()


def show_home(stock_data):
    """Home page with overview."""
    st.markdown('<p class="main-header">Welcome to Option-Strat</p>', unsafe_allow_html=True)

    st.markdown("""
    Your comprehensive toolkit for options trading analysis. This platform provides:

    - **Strategy Builder**: Build and analyze complex multi-leg options strategies
    - **Strategy Optimizer**: Find the best strategies for your market outlook
    - **Options Flow Scanner**: Track unusual options activity and smart money
    - **Greeks Calculator**: Calculate and visualize options Greeks
    - **50+ Strategy Templates**: Pre-configured strategies for any market condition
    """)

    st.markdown("---")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Strategies Available", "50+")
    with col2:
        st.metric("Greeks Calculated", "5 Primary + 4 Secondary")
    with col3:
        st.metric("Data Source", "Real-time" if stock_data.get('history') else "Simulated")
    with col4:
        st.metric("Analysis Tools", "6 Modules")

    st.markdown("---")

    # Feature cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Strategy Builder")
        st.markdown("""
        Build custom options strategies with our intuitive builder:
        - Support for all common strategies
        - Real-time P&L calculation
        - Interactive payoff diagrams
        - Greeks aggregation
        """)

        st.markdown("### 🎯 Strategy Optimizer")
        st.markdown("""
        Find the optimal strategy for your outlook:
        - Set target price and timeframe
        - Optimize by return or probability
        - Compare multiple strategies
        - Risk-adjusted rankings
        """)

    with col2:
        st.markdown("### 📈 Options Flow")
        st.markdown("""
        Track smart money movements:
        - Unusual activity detection
        - Multi-leg trade recognition
        - Sentiment analysis
        - Volume/OI tracking
        """)

        st.markdown("### 🔬 Greeks Analysis")
        st.markdown("""
        Deep dive into options Greeks:
        - Real-time Greek calculations
        - Greeks over price/time charts
        - 3D surface visualization
        - Position-level aggregation
        """)


def show_strategy_builder(symbol, stock_data):
    """Strategy builder page."""
    st.markdown("## 📊 Strategy Builder")

    price = stock_data['price']

    # Strategy selection
    col1, col2 = st.columns([1, 2])

    with col1:
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Iron Condor", "Iron Butterfly", "Bull Call Spread", "Bear Put Spread",
             "Long Straddle", "Long Strangle", "Covered Call", "Protective Put",
             "Long Call", "Long Put", "Short Call", "Short Put",
             "Call Butterfly", "Put Butterfly", "Calendar Spread"]
        )

        expiry_days = st.slider("Days to Expiration", 7, 90, 30)
        volatility = st.slider("Implied Volatility (%)", 10, 100, 25) / 100
        quantity = st.number_input("Contracts", 1, 100, 1)

    with col2:
        st.markdown(f"**Current Price:** ${price:.2f}")

        # Strategy-specific inputs
        if strategy_type == "Iron Condor":
            c1, c2, c3, c4 = st.columns(4)
            put_buy = c1.number_input("Put Buy", value=round(price * 0.90, 0))
            put_sell = c2.number_input("Put Sell", value=round(price * 0.95, 0))
            call_sell = c3.number_input("Call Sell", value=round(price * 1.05, 0))
            call_buy = c4.number_input("Call Buy", value=round(price * 1.10, 0))

        elif strategy_type == "Iron Butterfly":
            c1, c2, c3 = st.columns(3)
            wing_put = c1.number_input("Wing Put", value=round(price * 0.90, 0))
            body = c2.number_input("Body Strike", value=round(price, 0))
            wing_call = c3.number_input("Wing Call", value=round(price * 1.10, 0))

        elif strategy_type in ["Bull Call Spread", "Bear Call Spread"]:
            c1, c2 = st.columns(2)
            buy_strike = c1.number_input("Buy Strike", value=round(price, 0))
            sell_strike = c2.number_input("Sell Strike", value=round(price * 1.05, 0))

        elif strategy_type in ["Bear Put Spread", "Bull Put Spread"]:
            c1, c2 = st.columns(2)
            buy_strike = c1.number_input("Buy Strike", value=round(price, 0))
            sell_strike = c2.number_input("Sell Strike", value=round(price * 0.95, 0))

        elif strategy_type in ["Long Straddle", "Short Straddle"]:
            strike = st.number_input("Strike", value=round(price, 0))

        elif strategy_type in ["Long Strangle", "Short Strangle"]:
            c1, c2 = st.columns(2)
            put_strike = c1.number_input("Put Strike", value=round(price * 0.95, 0))
            call_strike = c2.number_input("Call Strike", value=round(price * 1.05, 0))

        else:
            strike = st.number_input("Strike", value=round(price, 0))

    # Build strategy
    if st.button("Analyze Strategy", type="primary"):
        builder = StrategyBuilder(symbol, price, volatility=volatility)

        try:
            # Create strategy based on type
            if strategy_type == "Iron Condor":
                strategy = builder.iron_condor(put_buy, put_sell, call_sell, call_buy, expiry_days, quantity)
            elif strategy_type == "Iron Butterfly":
                strategy = builder.iron_butterfly(wing_put, body, wing_call, expiry_days, quantity)
            elif strategy_type == "Bull Call Spread":
                strategy = builder.bull_call_spread(buy_strike, sell_strike, expiry_days, quantity)
            elif strategy_type == "Bear Put Spread":
                strategy = builder.bear_put_spread(buy_strike, sell_strike, expiry_days, quantity)
            elif strategy_type == "Long Straddle":
                strategy = builder.long_straddle(strike, expiry_days, quantity)
            elif strategy_type == "Long Strangle":
                strategy = builder.long_strangle(put_strike, call_strike, expiry_days, quantity)
            elif strategy_type == "Long Call":
                strategy = builder.long_call(strike, expiry_days, quantity)
            elif strategy_type == "Long Put":
                strategy = builder.long_put(strike, expiry_days, quantity)
            else:
                strategy = builder.long_call(strike, expiry_days, quantity)

            # Display results
            display_strategy_analysis(strategy, price)

        except Exception as e:
            st.error(f"Error building strategy: {e}")


def display_strategy_analysis(strategy, current_price):
    """Display strategy analysis results."""
    metrics = strategy.get_metrics()
    greeks = strategy.greeks

    # Metrics
    st.markdown("### Strategy Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        cost_type = "DEBIT" if strategy.is_debit else "CREDIT"
        st.metric("Net Cost", f"${abs(strategy.net_premium):.2f}", cost_type)

    with col2:
        st.metric("Max Profit", f"${metrics.max_profit:.2f}")

    with col3:
        st.metric("Max Loss", f"${metrics.max_loss:.2f}")

    with col4:
        st.metric("P(Profit)", f"{metrics.probability_of_profit*100:.1f}%")

    with col5:
        st.metric("Return on Risk", f"{metrics.return_on_risk*100:.1f}%")

    # Breakevens
    if metrics.breakevens:
        st.markdown(f"**Breakeven(s):** {', '.join(f'${be:.2f}' for be in metrics.breakevens)}")

    # Greeks
    st.markdown("### Position Greeks")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Delta (Δ)", f"{greeks.delta:.4f}")
    col2.metric("Gamma (Γ)", f"{greeks.gamma:.4f}")
    col3.metric("Theta (Θ)", f"{greeks.theta:.4f}")
    col4.metric("Vega (ν)", f"{greeks.vega:.4f}")
    col5.metric("Rho (ρ)", f"{greeks.rho:.4f}")

    # Payoff chart
    st.markdown("### Payoff Diagram")

    # Generate payoff data
    payoff_data = strategy.payoff_table(price_points=100)
    prices = payoff_data['prices']
    profits = payoff_data['profits']

    # Create DataFrame for plotting
    df = pd.DataFrame({'Price': prices, 'P&L': profits})

    # Plot with Streamlit
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add profit/loss areas
    pos_profits = [max(p, 0) for p in profits]
    neg_profits = [min(p, 0) for p in profits]

    fig.add_trace(go.Scatter(
        x=prices, y=pos_profits,
        fill='tozeroy',
        fillcolor='rgba(0, 200, 83, 0.3)',
        line=dict(width=0),
        name='Profit'
    ))

    fig.add_trace(go.Scatter(
        x=prices, y=neg_profits,
        fill='tozeroy',
        fillcolor='rgba(255, 23, 68, 0.3)',
        line=dict(width=0),
        name='Loss'
    ))

    # Add main line
    fig.add_trace(go.Scatter(
        x=prices, y=profits,
        mode='lines',
        name='P&L at Expiration',
        line=dict(color='#2196F3', width=3)
    ))

    # Add current price line
    fig.add_vline(x=current_price, line=dict(color='#FFC107', width=2, dash='dash'),
                  annotation_text=f"Current: ${current_price:.2f}")

    # Add breakevens
    for be in metrics.breakevens:
        fig.add_vline(x=be, line=dict(color='#9C27B0', width=1, dash='dot'))

    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Stock Price at Expiration ($)',
        yaxis_title='Profit / Loss ($)',
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # P&L Table
    st.markdown("### P&L Table")
    st.text(strategy.summary())


def show_optimizer(symbol, stock_data):
    """Strategy optimizer page."""
    st.markdown("## 🎯 Strategy Optimizer")

    price = stock_data['price']

    col1, col2 = st.columns(2)

    with col1:
        target_price = st.number_input(
            "Target Price",
            value=round(price * 1.05, 2),
            step=1.0
        )
        target_days = st.slider("Days to Target", 1, 60, 14)

    with col2:
        goal = st.selectbox(
            "Optimization Goal",
            ["Maximum Return", "Maximum Probability", "Minimum Risk",
             "Risk-Adjusted", "Balanced"]
        )
        max_results = st.slider("Max Results", 3, 20, 10)

    # Calculate move required
    move_pct = (target_price / price - 1) * 100
    st.markdown(f"**Move Required:** {move_pct:+.1f}%")

    if st.button("Find Optimal Strategies", type="primary"):
        goal_map = {
            "Maximum Return": OptimizationGoal.MAX_RETURN,
            "Maximum Probability": OptimizationGoal.MAX_PROBABILITY,
            "Minimum Risk": OptimizationGoal.MIN_RISK,
            "Risk-Adjusted": OptimizationGoal.RISK_ADJUSTED,
            "Balanced": OptimizationGoal.BALANCED,
        }

        optimizer = StrategyOptimizer(symbol, price)

        with st.spinner("Optimizing strategies..."):
            results = optimizer.optimize(
                target_price=target_price,
                target_days=target_days,
                goal=goal_map[goal],
                max_results=max_results
            )

        if results:
            st.markdown("### Top Strategies")

            for result in results[:5]:
                with st.expander(f"#{result.rank} - {result.strategy.name}"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Expected Return", f"${result.expected_return:.2f}")
                    col2.metric("P(Profit)", f"{result.probability_of_profit*100:.1f}%")
                    col3.metric("Max Loss", f"${result.max_loss:.2f}")
                    col4.metric("Score", f"{result.score:.2f}")

                    st.text(result.strategy.summary())
        else:
            st.warning("No strategies found matching criteria.")


def show_flow():
    """Options flow page."""
    st.markdown("## 📈 Options Flow Scanner")

    col1, col2, col3 = st.columns(3)

    with col1:
        flow_symbol = st.text_input("Filter Symbol", value="")

    with col2:
        min_premium = st.number_input("Min Premium ($)", value=50000, step=10000)

    with col3:
        only_unusual = st.checkbox("Only Unusual Activity", value=True)

    if st.button("Scan Flow", type="primary"):
        scanner = FlowScanner()

        flow_filter = FlowFilter(
            symbols=[flow_symbol.upper()] if flow_symbol else None,
            min_premium=min_premium,
            only_unusual=only_unusual
        )

        with st.spinner("Scanning options flow..."):
            trades = scanner.get_flow(flow_filter, limit=30)

        if trades:
            # Summary stats
            st.markdown(scanner.flow_summary(trades))

            # Sentiment breakdown
            sentiment_stats = scanner.aggregate_by_sentiment(trades)

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Bullish Premium",
                f"${sentiment_stats['bullish']['premium']:,.0f}",
                f"{sentiment_stats['bullish']['percentage']:.1f}%"
            )
            col2.metric(
                "Bearish Premium",
                f"${sentiment_stats['bearish']['premium']:,.0f}",
                f"{sentiment_stats['bearish']['percentage']:.1f}%"
            )
            col3.metric(
                "Neutral Premium",
                f"${sentiment_stats['neutral']['premium']:,.0f}",
                f"{sentiment_stats['neutral']['percentage']:.1f}%"
            )

            # Trade table
            st.markdown("### Recent Trades")

            trade_data = []
            for trade in trades[:20]:
                flags = []
                if trade.is_unusual:
                    flags.append("UNS")
                if trade.is_sweep:
                    flags.append("SWP")
                if trade.is_block:
                    flags.append("BLK")

                trade_data.append({
                    'Time': trade.timestamp.strftime('%H:%M:%S'),
                    'Symbol': trade.symbol,
                    'Type': f"{trade.side.value.upper()} {trade.trade_type.value.upper()}",
                    'Strike': f"${trade.strike:.0f}",
                    'Expiry': trade.expiry.strftime('%m/%d'),
                    'Premium': f"${trade.premium:,.0f}",
                    'IV': f"{trade.implied_volatility*100:.1f}%",
                    'Sentiment': trade.sentiment.value.upper(),
                    'Flags': ' '.join(flags)
                })

            st.dataframe(pd.DataFrame(trade_data), use_container_width=True)
        else:
            st.warning("No trades found matching criteria.")


def show_greeks_calculator(symbol, stock_data):
    """Greeks calculator page."""
    st.markdown("## 🔬 Greeks Calculator")

    price = stock_data['price']

    col1, col2 = st.columns(2)

    with col1:
        option_type = st.radio("Option Type", ["Call", "Put"])
        strike = st.number_input("Strike Price", value=round(price, 0))
        expiry_days = st.slider("Days to Expiration", 1, 365, 30)

    with col2:
        volatility = st.slider("Implied Volatility (%)", 5, 150, 25) / 100
        risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100

    # Calculate option
    option = Option(
        option_type=option_type.lower(),
        strike=strike,
        expiry_days=expiry_days,
        underlying_price=price,
        volatility=volatility,
        risk_free_rate=risk_free_rate
    )

    # Display results
    st.markdown("### Option Valuation")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Option Price", f"${option.price:.2f}")
    col2.metric("Intrinsic Value", f"${option.intrinsic_value:.2f}")
    col3.metric("Extrinsic Value", f"${option.extrinsic_value:.2f}")
    col4.metric("Moneyness", option.moneyness)

    st.markdown("### Greeks")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Delta (Δ)", f"{option.delta:.4f}")
    col2.metric("Gamma (Γ)", f"{option.gamma:.4f}")
    col3.metric("Theta (Θ)", f"{option.theta:.4f}")
    col4.metric("Vega (ν)", f"{option.vega:.4f}")
    col5.metric("Rho (ρ)", f"{option.rho:.4f}")

    st.markdown("### Probability Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("P(ITM)", f"{option.probability_itm*100:.1f}%")
    col2.metric("Breakeven", f"${option.breakeven:.2f}")
    col3.metric("Max Loss", f"${option.max_loss:.2f}")

    # Greeks visualization
    st.markdown("### Greeks Over Price")

    prices = np.linspace(price * 0.7, price * 1.3, 50)
    deltas, gammas, thetas, vegas = [], [], [], []

    for p in prices:
        temp_option = Option(
            option_type=option_type.lower(),
            strike=strike,
            expiry_days=expiry_days,
            underlying_price=p,
            volatility=volatility,
            risk_free_rate=risk_free_rate
        )
        deltas.append(temp_option.delta)
        gammas.append(temp_option.gamma)
        thetas.append(temp_option.theta)
        vegas.append(temp_option.vega)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['Delta', 'Gamma', 'Theta', 'Vega'])

    fig.add_trace(go.Scatter(x=prices, y=deltas, name='Delta',
                             line=dict(color='#2196F3')), row=1, col=1)
    fig.add_trace(go.Scatter(x=prices, y=gammas, name='Gamma',
                             line=dict(color='#FF9800')), row=1, col=2)
    fig.add_trace(go.Scatter(x=prices, y=thetas, name='Theta',
                             line=dict(color='#E91E63')), row=2, col=1)
    fig.add_trace(go.Scatter(x=prices, y=vegas, name='Vega',
                             line=dict(color='#4CAF50')), row=2, col=2)

    for i in range(1, 3):
        for j in range(1, 3):
            fig.add_vline(x=price, line=dict(color='#FFC107', dash='dash'),
                         row=i, col=j)

    fig.update_layout(template='plotly_dark', height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def show_templates(symbol, stock_data):
    """Strategy templates page."""
    st.markdown("## 📋 Strategy Templates")

    price = stock_data['price']

    templates = StrategyTemplates(symbol, price)
    all_templates = templates.list_all()

    # Filter by outlook
    outlook_filter = st.selectbox(
        "Filter by Market Outlook",
        ["All", "Bullish", "Bearish", "Neutral", "Volatile"]
    )

    # Display templates
    for name, info in all_templates.items():
        if outlook_filter != "All" and info['outlook'] != outlook_filter.lower():
            continue

        with st.expander(f"**{info['name']}** - {info['description'][:60]}..."):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Outlook:** {info['outlook'].title()}")
                st.markdown(f"**Risk Profile:** {info['risk_profile'].title()}")
                st.markdown(f"**Time Horizon:** {info['time_horizon'].title()}")

            with col2:
                st.markdown(f"**Max Profit:** {info['max_profit']}")
                st.markdown(f"**Max Loss:** {info['max_loss']}")

            st.markdown(f"**Ideal Conditions:** {info['ideal_conditions']}")

            if st.button(f"Create {name}", key=f"create_{name}"):
                try:
                    strategy = templates.create(name, expiry_days=30)
                    display_strategy_analysis(strategy, price)
                except Exception as e:
                    st.error(f"Error creating strategy: {e}")


def show_settings():
    """Settings page."""
    st.markdown("## ⚙️ Settings")

    st.markdown("### Default Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("Default Volatility (%)", value=25, key="default_vol")
        st.number_input("Risk-Free Rate (%)", value=5.0, key="default_rate")
        st.number_input("Default Expiry (days)", value=30, key="default_expiry")

    with col2:
        st.selectbox("Data Source", ["Simulated", "Yahoo Finance"], key="data_source")
        st.selectbox("Chart Theme", ["Dark", "Light"], key="chart_theme")
        st.number_input("Monte Carlo Simulations", value=10000, key="mc_sims")

    st.markdown("### About")
    st.markdown("""
    **Option-Strat** is a comprehensive options trading toolkit inspired by OptionStrat.com.

    Features:
    - Black-Scholes option pricing
    - Complete Greeks calculation
    - 50+ pre-defined strategies
    - Interactive P&L visualization
    - Options flow scanner
    - Strategy optimizer

    Built with Python, Streamlit, and Plotly.
    """)


if __name__ == "__main__":
    main()
