"""
Data Utilities

This module provides functions for fetching market data from
various sources (Yahoo Finance, simulated data, etc.)
"""

from typing import Optional, Dict, List, Any
from datetime import datetime, date, timedelta
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


def get_stock_data(
    symbol: str,
    period: str = "1mo",
    interval: str = "1d"
) -> Dict[str, Any]:
    """
    Fetch stock data for a symbol.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        period: Data period ('1d', '5d', '1mo', '3mo', '1y', 'max')
        interval: Data interval ('1m', '5m', '1h', '1d')

    Returns:
        Dictionary with stock data
    """
    if YFINANCE_AVAILABLE:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period=period, interval=interval)

            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'price': info.get('regularMarketPrice', hist['Close'].iloc[-1] if len(hist) > 0 else 100),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': info.get('regularMarketVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0) or 0,
                '52w_high': info.get('fiftyTwoWeekHigh', 0),
                '52w_low': info.get('fiftyTwoWeekLow', 0),
                'history': hist.to_dict() if len(hist) > 0 else {},
            }
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    # Return simulated data if yfinance not available
    return _simulate_stock_data(symbol)


def _simulate_stock_data(symbol: str) -> Dict[str, Any]:
    """Generate simulated stock data."""
    base_prices = {
        'AAPL': 185, 'TSLA': 250, 'NVDA': 480, 'AMD': 145,
        'MSFT': 380, 'GOOGL': 140, 'AMZN': 180, 'META': 360,
        'SPY': 475, 'QQQ': 410, 'IWM': 200, 'DIA': 380,
    }

    price = base_prices.get(symbol, 100 + np.random.uniform(-20, 50))
    change = np.random.uniform(-3, 3)

    return {
        'symbol': symbol,
        'name': f'{symbol} Inc.',
        'price': price,
        'change': change,
        'change_percent': change / price * 100,
        'volume': int(np.random.uniform(1e6, 1e8)),
        'market_cap': int(price * np.random.uniform(1e9, 1e12)),
        'pe_ratio': np.random.uniform(15, 40),
        'dividend_yield': np.random.uniform(0, 0.03),
        '52w_high': price * 1.3,
        '52w_low': price * 0.7,
        'history': {},
    }


def get_option_chain(
    symbol: str,
    expiry_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch options chain for a symbol.

    Args:
        symbol: Stock symbol
        expiry_date: Specific expiry date (YYYY-MM-DD), None for nearest

    Returns:
        Dictionary with option chain data
    """
    if YFINANCE_AVAILABLE:
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                return _simulate_option_chain(symbol, expiry_date)

            if expiry_date:
                target_expiry = expiry_date
            else:
                target_expiry = expirations[0]

            chain = ticker.option_chain(target_expiry)

            return {
                'symbol': symbol,
                'expiry': target_expiry,
                'expirations': list(expirations),
                'calls': chain.calls.to_dict('records'),
                'puts': chain.puts.to_dict('records'),
                'underlying_price': ticker.info.get('regularMarketPrice', 100),
            }
        except Exception as e:
            print(f"Error fetching options for {symbol}: {e}")

    return _simulate_option_chain(symbol, expiry_date)


def _simulate_option_chain(
    symbol: str,
    expiry_date: Optional[str] = None
) -> Dict[str, Any]:
    """Generate simulated option chain data."""
    stock_data = _simulate_stock_data(symbol)
    price = stock_data['price']

    if expiry_date:
        expiry = datetime.strptime(expiry_date, '%Y-%m-%d').date()
    else:
        expiry = date.today() + timedelta(days=30)

    days_to_expiry = (expiry - date.today()).days
    base_iv = 0.25 + np.random.uniform(-0.05, 0.10)

    strikes = [round(price * (0.8 + i * 0.025), 0) for i in range(17)]

    calls = []
    puts = []

    for strike in strikes:
        moneyness = price / strike

        # IV smile
        iv_adjustment = 0.1 * (1 - moneyness) ** 2
        iv = base_iv + iv_adjustment

        # Simplified BS for simulation
        from optionstrat.models.pricing import BlackScholes
        bs = BlackScholes(price, strike, days_to_expiry/365, 0.05, iv)

        call_price = bs.call_price()
        put_price = bs.put_price()

        calls.append({
            'strike': strike,
            'lastPrice': round(call_price, 2),
            'bid': round(call_price * 0.95, 2),
            'ask': round(call_price * 1.05, 2),
            'volume': int(np.random.uniform(100, 10000)),
            'openInterest': int(np.random.uniform(500, 50000)),
            'impliedVolatility': iv,
            'inTheMoney': price > strike,
        })

        puts.append({
            'strike': strike,
            'lastPrice': round(put_price, 2),
            'bid': round(put_price * 0.95, 2),
            'ask': round(put_price * 1.05, 2),
            'volume': int(np.random.uniform(100, 10000)),
            'openInterest': int(np.random.uniform(500, 50000)),
            'impliedVolatility': iv,
            'inTheMoney': price < strike,
        })

    # Generate multiple expirations
    expirations = []
    base_date = date.today()
    for i in range(12):
        exp = base_date + timedelta(days=7 * (i + 1))
        expirations.append(exp.strftime('%Y-%m-%d'))

    return {
        'symbol': symbol,
        'expiry': expiry.strftime('%Y-%m-%d'),
        'expirations': expirations,
        'calls': calls,
        'puts': puts,
        'underlying_price': price,
    }


def calculate_historical_volatility(
    symbol: str,
    period: str = "1y",
    window: int = 21
) -> float:
    """
    Calculate historical volatility for a symbol.

    Args:
        symbol: Stock symbol
        period: Data period
        window: Rolling window for calculation

    Returns:
        Annualized historical volatility
    """
    if YFINANCE_AVAILABLE:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if len(hist) > window:
                returns = np.log(hist['Close'] / hist['Close'].shift(1))
                rolling_std = returns.rolling(window=window).std()
                hv = rolling_std.iloc[-1] * np.sqrt(252)
                return float(hv)
        except Exception:
            pass

    # Return random realistic volatility
    return np.random.uniform(0.20, 0.50)
