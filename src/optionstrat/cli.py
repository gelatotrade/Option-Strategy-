"""
Option-Strat CLI

Command-line interface for the Option-Strat toolkit.

Usage:
    optionstrat analyze SYMBOL [--strategy=STRATEGY] [--expiry=DAYS]
    optionstrat optimize SYMBOL --target=PRICE --days=DAYS
    optionstrat flow [--symbol=SYMBOL] [--min-premium=AMOUNT]
    optionstrat greeks SYMBOL --strike=STRIKE --type=TYPE --expiry=DAYS
    optionstrat compare SYMBOL --strategies=LIST

Examples:
    optionstrat analyze AAPL --strategy=iron_condor --expiry=30
    optionstrat optimize TSLA --target=280 --days=14
    optionstrat flow --symbol=SPY --min-premium=100000
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def main():
    """Option-Strat: The Options Trader's Toolkit"""
    pass


@main.command()
@click.argument('symbol')
@click.option('--strategy', '-s', default='iron_condor', help='Strategy type')
@click.option('--expiry', '-e', default=30, type=int, help='Days to expiration')
@click.option('--volatility', '-v', default=0.25, type=float, help='Implied volatility')
def analyze(symbol: str, strategy: str, expiry: int, volatility: float):
    """Analyze an options strategy for a symbol."""
    from optionstrat.utils.data import get_stock_data
    from optionstrat.strategies.builder import StrategyBuilder

    console.print(f"\n[bold blue]Analyzing {strategy} for {symbol}...[/bold blue]\n")

    # Get stock data
    with console.status("Fetching stock data..."):
        stock = get_stock_data(symbol)

    price = stock['price']
    console.print(f"[green]Current Price:[/green] ${price:.2f}")

    # Build strategy
    builder = StrategyBuilder(symbol, price, volatility=volatility)

    try:
        # Create strategy based on name
        if strategy == 'iron_condor':
            strat = builder.iron_condor(
                put_buy=price * 0.90,
                put_sell=price * 0.95,
                call_sell=price * 1.05,
                call_buy=price * 1.10,
                expiry_days=expiry
            )
        elif strategy == 'iron_butterfly':
            strat = builder.iron_butterfly(
                wing_put=price * 0.90,
                body_strike=price,
                wing_call=price * 1.10,
                expiry_days=expiry
            )
        elif strategy == 'bull_call_spread':
            strat = builder.bull_call_spread(
                buy_strike=price,
                sell_strike=price * 1.05,
                expiry_days=expiry
            )
        elif strategy == 'bear_put_spread':
            strat = builder.bear_put_spread(
                buy_strike=price,
                sell_strike=price * 0.95,
                expiry_days=expiry
            )
        elif strategy == 'long_straddle':
            strat = builder.long_straddle(price, expiry)
        elif strategy == 'long_strangle':
            strat = builder.long_strangle(
                put_strike=price * 0.95,
                call_strike=price * 1.05,
                expiry_days=expiry
            )
        elif strategy == 'long_call':
            strat = builder.long_call(price, expiry)
        elif strategy == 'long_put':
            strat = builder.long_put(price, expiry)
        else:
            console.print(f"[red]Unknown strategy: {strategy}[/red]")
            return

        # Display results
        console.print(strat.summary())

        # Create metrics table
        metrics = strat.get_metrics()

        table = Table(title="Strategy Metrics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Max Profit", f"${metrics.max_profit:.2f}")
        table.add_row("Max Loss", f"${metrics.max_loss:.2f}")
        table.add_row("Breakevens", ", ".join(f"${be:.2f}" for be in metrics.breakevens))
        table.add_row("P(Profit)", f"{metrics.probability_of_profit*100:.1f}%")
        table.add_row("Return on Risk", f"{metrics.return_on_risk*100:.1f}%")

        console.print(table)

        # Greeks
        greeks = strat.greeks
        greeks_table = Table(title="Position Greeks", box=box.ROUNDED)
        greeks_table.add_column("Greek", style="cyan")
        greeks_table.add_column("Value", style="yellow")

        greeks_table.add_row("Delta (Δ)", f"{greeks.delta:.4f}")
        greeks_table.add_row("Gamma (Γ)", f"{greeks.gamma:.4f}")
        greeks_table.add_row("Theta (Θ)", f"{greeks.theta:.4f} /day")
        greeks_table.add_row("Vega (ν)", f"{greeks.vega:.4f} /1%")

        console.print(greeks_table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@main.command()
@click.argument('symbol')
@click.option('--target', '-t', required=True, type=float, help='Target price')
@click.option('--days', '-d', required=True, type=int, help='Days to target')
@click.option('--goal', '-g', default='max_return', help='Optimization goal')
@click.option('--max-results', '-n', default=5, type=int, help='Max results')
def optimize(symbol: str, target: float, days: int, goal: str, max_results: int):
    """Find optimal strategies for a target price."""
    from optionstrat.utils.data import get_stock_data
    from optionstrat.optimizer.optimizer import StrategyOptimizer, OptimizationGoal

    console.print(f"\n[bold blue]Optimizing strategies for {symbol}...[/bold blue]")
    console.print(f"Target: ${target:.2f} in {days} days\n")

    # Get stock data
    with console.status("Fetching stock data..."):
        stock = get_stock_data(symbol)

    price = stock['price']
    console.print(f"[green]Current Price:[/green] ${price:.2f}")
    console.print(f"[yellow]Move Required:[/yellow] {(target/price - 1)*100:+.1f}%\n")

    # Map goal string to enum
    goal_map = {
        'max_return': OptimizationGoal.MAX_RETURN,
        'max_probability': OptimizationGoal.MAX_PROBABILITY,
        'min_risk': OptimizationGoal.MIN_RISK,
        'risk_adjusted': OptimizationGoal.RISK_ADJUSTED,
        'balanced': OptimizationGoal.BALANCED,
    }

    opt_goal = goal_map.get(goal, OptimizationGoal.MAX_RETURN)

    # Run optimization
    optimizer = StrategyOptimizer(symbol, price)

    with console.status("Running optimization..."):
        results = optimizer.optimize(
            target_price=target,
            target_days=days,
            goal=opt_goal,
            max_results=max_results
        )

    if not results:
        console.print("[red]No suitable strategies found.[/red]")
        return

    # Display results
    table = Table(title="Optimization Results", box=box.ROUNDED)
    table.add_column("#", style="dim")
    table.add_column("Strategy", style="cyan")
    table.add_column("Expected Return", style="green")
    table.add_column("P(Profit)", style="yellow")
    table.add_column("Max Loss", style="red")
    table.add_column("Score", style="magenta")

    for result in results:
        table.add_row(
            str(result.rank),
            result.strategy.name,
            f"${result.expected_return:.2f}",
            f"{result.probability_of_profit*100:.1f}%",
            f"${result.max_loss:.2f}",
            f"{result.score:.2f}"
        )

    console.print(table)

    # Show details of top result
    if results:
        console.print("\n[bold]Top Strategy Details:[/bold]")
        console.print(results[0].strategy.summary())


@main.command()
@click.option('--symbol', '-s', default=None, help='Filter by symbol')
@click.option('--min-premium', '-p', default=0, type=float, help='Minimum premium')
@click.option('--unusual', '-u', is_flag=True, help='Only unusual activity')
@click.option('--limit', '-n', default=20, type=int, help='Number of trades')
def flow(symbol: str, min_premium: float, unusual: bool, limit: int):
    """Monitor options flow activity."""
    from optionstrat.flow.scanner import FlowScanner, FlowFilter

    console.print("\n[bold blue]Options Flow Scanner[/bold blue]\n")

    scanner = FlowScanner()

    # Build filter
    flow_filter = FlowFilter(
        symbols=[symbol] if symbol else None,
        min_premium=min_premium,
        only_unusual=unusual
    )

    # Get flow
    with console.status("Scanning options flow..."):
        trades = scanner.get_flow(flow_filter, limit=limit)

    if not trades:
        console.print("[yellow]No trades matching criteria.[/yellow]")
        return

    # Summary
    console.print(scanner.flow_summary(trades))

    # Trade table
    table = Table(title="Recent Flow", box=box.ROUNDED)
    table.add_column("Time", style="dim")
    table.add_column("Symbol", style="cyan")
    table.add_column("Type", style="white")
    table.add_column("Strike", style="yellow")
    table.add_column("Expiry", style="white")
    table.add_column("Premium", style="green")
    table.add_column("Sentiment", style="magenta")
    table.add_column("Flags", style="red")

    for trade in trades[:15]:
        flags = []
        if trade.is_unusual:
            flags.append("UNS")
        if trade.is_sweep:
            flags.append("SWP")
        if trade.is_block:
            flags.append("BLK")

        sent_color = {
            'bullish': 'green',
            'bearish': 'red',
            'neutral': 'yellow'
        }.get(trade.sentiment.value, 'white')

        table.add_row(
            trade.timestamp.strftime('%H:%M:%S'),
            trade.symbol,
            f"{trade.side.value.upper()} {trade.trade_type.value.upper()}",
            f"${trade.strike:.0f}",
            trade.expiry.strftime('%m/%d'),
            f"${trade.premium:,.0f}",
            f"[{sent_color}]{trade.sentiment.value.upper()}[/{sent_color}]",
            " ".join(flags)
        )

    console.print(table)


@main.command()
@click.argument('symbol')
@click.option('--strike', '-k', required=True, type=float, help='Strike price')
@click.option('--type', '-t', 'option_type', required=True, help='call or put')
@click.option('--expiry', '-e', required=True, type=int, help='Days to expiry')
@click.option('--volatility', '-v', default=0.25, type=float, help='IV')
def greeks(symbol: str, strike: float, option_type: str, expiry: int, volatility: float):
    """Calculate Greeks for a single option."""
    from optionstrat.utils.data import get_stock_data
    from optionstrat.models.option import Option

    console.print(f"\n[bold blue]Greeks Calculator[/bold blue]\n")

    # Get stock data
    with console.status("Fetching stock data..."):
        stock = get_stock_data(symbol)

    price = stock['price']

    # Create option
    option = Option(
        option_type=option_type,
        strike=strike,
        expiry_days=expiry,
        underlying_price=price,
        volatility=volatility
    )

    # Display
    console.print(option.summary())


@main.command()
@click.argument('symbol')
@click.option('--strategies', '-s', default='iron_condor,bull_call_spread,long_straddle',
              help='Comma-separated strategies')
@click.option('--expiry', '-e', default=30, type=int, help='Days to expiry')
def compare(symbol: str, strategies: str, expiry: int):
    """Compare multiple strategies side by side."""
    from optionstrat.utils.data import get_stock_data
    from optionstrat.strategies.builder import StrategyBuilder

    console.print(f"\n[bold blue]Strategy Comparison for {symbol}[/bold blue]\n")

    # Get stock data
    with console.status("Fetching stock data..."):
        stock = get_stock_data(symbol)

    price = stock['price']
    console.print(f"[green]Current Price:[/green] ${price:.2f}\n")

    builder = StrategyBuilder(symbol, price)
    strategy_list = [s.strip() for s in strategies.split(',')]

    results = []

    for strat_name in strategy_list:
        try:
            if strat_name == 'iron_condor':
                strat = builder.iron_condor(
                    price * 0.90, price * 0.95,
                    price * 1.05, price * 1.10, expiry
                )
            elif strat_name == 'bull_call_spread':
                strat = builder.bull_call_spread(price, price * 1.05, expiry)
            elif strat_name == 'bear_put_spread':
                strat = builder.bear_put_spread(price, price * 0.95, expiry)
            elif strat_name == 'long_straddle':
                strat = builder.long_straddle(price, expiry)
            elif strat_name == 'long_strangle':
                strat = builder.long_strangle(price * 0.95, price * 1.05, expiry)
            elif strat_name == 'iron_butterfly':
                strat = builder.iron_butterfly(price * 0.90, price, price * 1.10, expiry)
            elif strat_name == 'long_call':
                strat = builder.long_call(price, expiry)
            elif strat_name == 'long_put':
                strat = builder.long_put(price, expiry)
            else:
                continue

            metrics = strat.get_metrics()
            results.append({
                'name': strat.name,
                'net_premium': strat.net_premium,
                'max_profit': metrics.max_profit,
                'max_loss': metrics.max_loss,
                'prob_profit': metrics.probability_of_profit,
                'delta': strat.delta,
                'theta': strat.theta,
            })
        except Exception as e:
            console.print(f"[yellow]Warning: Could not create {strat_name}: {e}[/yellow]")

    # Display comparison table
    table = Table(title="Strategy Comparison", box=box.ROUNDED)
    table.add_column("Strategy", style="cyan")
    table.add_column("Net Cost", style="white")
    table.add_column("Max Profit", style="green")
    table.add_column("Max Loss", style="red")
    table.add_column("P(Profit)", style="yellow")
    table.add_column("Delta", style="blue")
    table.add_column("Theta", style="magenta")

    for r in results:
        cost_str = f"{'DEBIT' if r['net_premium'] > 0 else 'CREDIT'} ${abs(r['net_premium']):.2f}"
        table.add_row(
            r['name'],
            cost_str,
            f"${r['max_profit']:.2f}",
            f"${r['max_loss']:.2f}",
            f"{r['prob_profit']*100:.1f}%",
            f"{r['delta']:.4f}",
            f"{r['theta']:.4f}"
        )

    console.print(table)


@main.command()
def templates():
    """List all available strategy templates."""
    from optionstrat.strategies.templates import StrategyTemplates, MarketOutlook

    console.print("\n[bold blue]Available Strategy Templates[/bold blue]\n")

    # Create dummy templates instance
    templates = StrategyTemplates("DUMMY", 100)
    all_templates = templates.list_all()

    # Group by outlook
    outlooks = {
        'bullish': [],
        'bearish': [],
        'neutral': [],
        'volatile': []
    }

    for name, info in all_templates.items():
        outlook = info['outlook']
        outlooks[outlook].append((name, info))

    for outlook, temps in outlooks.items():
        if temps:
            table = Table(title=f"{outlook.upper()} Strategies", box=box.ROUNDED)
            table.add_column("Template", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("Risk", style="yellow")

            for name, info in temps:
                table.add_row(
                    name,
                    info['description'][:50] + "..." if len(info['description']) > 50 else info['description'],
                    info['risk_profile']
                )

            console.print(table)
            console.print()


@main.command()
def interactive():
    """Start interactive mode."""
    console.print("\n[bold blue]Option-Strat Interactive Mode[/bold blue]")
    console.print("Type 'help' for commands, 'exit' to quit.\n")

    while True:
        try:
            cmd = console.input("[bold green]>>> [/bold green]").strip()

            if cmd.lower() in ('exit', 'quit', 'q'):
                console.print("[yellow]Goodbye![/yellow]")
                break

            if cmd.lower() == 'help':
                console.print("""
[bold]Available Commands:[/bold]
  analyze SYMBOL          - Analyze a stock
  price SYMBOL            - Get current price
  chain SYMBOL            - Show option chain
  build STRATEGY SYMBOL   - Build a strategy
  exit                    - Exit interactive mode
                """)
                continue

            parts = cmd.split()
            if not parts:
                continue

            action = parts[0].lower()

            if action == 'price' and len(parts) >= 2:
                from optionstrat.utils.data import get_stock_data
                stock = get_stock_data(parts[1].upper())
                console.print(f"[green]{parts[1].upper()}:[/green] ${stock['price']:.2f}")

            elif action == 'chain' and len(parts) >= 2:
                from optionstrat.utils.data import get_option_chain
                chain = get_option_chain(parts[1].upper())
                console.print(f"[green]Expirations:[/green] {', '.join(chain['expirations'][:5])}")
                console.print(f"[green]Calls:[/green] {len(chain['calls'])} strikes")
                console.print(f"[green]Puts:[/green] {len(chain['puts'])} strikes")

            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == '__main__':
    main()
