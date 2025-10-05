from requests import get
import pandas as pd
import numpy as np
import datetime
from tvDatafeed import TvDatafeed, Interval
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale

import time
import sys
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')

import vectorbt as vbt
import arcticdb as adb
from scipy.stats import rankdata

import arcticdb as adb

from fredapi import Fred

import ta

#####################################################################################
# BUILT-IN FUNCTIONS
#####################################################################################
from hyperliquid.info import Info
from hyperliquid.utils import constants
import requests
import statsmodels.api as sm
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

info = Info(base_url=constants.MAINNET_API_URL, skip_ws=True)
meta, ctxs = info.meta_and_asset_ctxs()


def hyperliquid_price(coin):
    try:
        idx = next(i for i, asset in enumerate(meta['universe']) if asset['name'] == coin)
        # Obtener el precio de mercado actual (mark price)
        mark_price = float(ctxs[idx]['markPx'])
        return mark_price
    except StopIteration:
        return np.nan


def get_historical_klines(symbol: str, start_date: float, interval: str = '1h', spot: bool = True,
                          add_k: bool = False) -> pd.DataFrame:
    """
    Retrieve historical OHLC data from Binance Spot or Futures (Perpetual) API.
    Appends current price as final row with only 'close' filled.
    """
    start_ms = int(start_date * 1e3)
    end_ms = int((time.time() - 300) * 1e3)  # 5 min buffer

    base_url = 'https://api4.binance.com/api/v3/klines' if spot else 'https://fapi.binance.com/fapi/v1/klines'
    ticker_url = 'https://api4.binance.com/api/v3/ticker/price' if spot else 'https://fapi.binance.com/fapi/v1/ticker/price'

    all_data = []
    current_start = start_ms
    max_retries = 3

    def fetch_klines(start_ts: int, end_ts: int) -> pd.DataFrame:
        url = f"{base_url}?symbol={symbol}&interval={interval}&startTime={start_ts}&endTime={end_ts}&limit=1000"
        for attempt in range(max_retries):
            try:
                res = np.array(requests.get(url).json())
                if len(res) == 0:
                    return pd.DataFrame()
                df = pd.DataFrame({
                    'timestamp': res[:, 0],
                    'open': res[:, 1],
                    'high': res[:, 2],
                    'low': res[:, 3],
                    'close': res[:, 4],
                }).astype(float)
                return df
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to fetch data after {max_retries} attempts.") from e
                time.sleep(5)

    while current_start < end_ms:
        batch = fetch_klines(current_start, end_ms)
        if batch.empty:
            break
        all_data.append(batch)
        current_start = int(batch['timestamp'].max()) + 1
        if len(batch) < 1000:
            break

    df = pd.concat(all_data).drop_duplicates('timestamp')
    df.sort_values(by='timestamp', inplace=True)
    df['timestamp'] /= 1e3  # Convert ms to seconds
    df.index = range(len(df))

    # ─────── Add current price as last row ───────
    try:
        # price_resp = requests.get(ticker_url, params={'symbol': symbol}).json()
        # current_price = float(price_resp['price'])
        s = symbol.split('USD')[0]
        if add_k:
            current_price = hyperliquid_price(f'k{s}') / 1e3
        else:
            current_price = hyperliquid_price(s)
        current_time = time.time()

        last_row = {
            'timestamp': current_time,
            'open': current_price,
            'high': current_price,
            'low': current_price,
            'close': current_price
        }
        df = pd.concat([df, pd.DataFrame([last_row])], ignore_index=True)
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch current price for {symbol}: {e}")

    return df


def get_historical_klines_bybit(symbol: str, start_date: float, interval: str = '60',
                                spot: bool = False) -> pd.DataFrame:
    """
    Retrieve historical OHLC data from Bybit Spot or Perpetual market.
    Appends current price as final row with only 'close' filled.

    Parameters:
    - symbol: str (e.g., 'BTCUSDT')
    - start_date: float (Unix timestamp in seconds)
    - interval: str ('1', '3', '5', '15', '30', '60', '120', '240', 'D', etc.)
    - spot: bool, True for Spot market, False for Perpetual
    """
    start_sec = int(start_date)
    end_sec = int(time.time())

    base_url = 'https://api.bybit.com/spot/v3/public/quote/kline' if spot else 'https://api.bybit.com/v5/market/kline'
    ticker_url = 'https://api.bybit.com/spot/v3/public/quote/ticker/price' if spot else 'https://api.bybit.com/v5/market/tickers'

    all_data = []
    current_start = start_sec
    limit = 1000  # Max candles per call

    def fetch_klines(start_ts: int) -> pd.DataFrame:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
            'start': start_ts * 1000 if not spot else None
        }
        url = base_url
        for attempt in range(3):
            try:
                response = requests.get(url, params=params)
                data = response.json()
                candles = data.get('result', {}).get('list', []) if not spot else data.get('result', [])
                if not candles:
                    return pd.DataFrame()
                candles = np.array(candles, dtype=object)
                df = pd.DataFrame({
                    'timestamp': candles[:, 0].astype(float) / 1000,
                    'open': candles[:, 1].astype(float),
                    'high': candles[:, 2].astype(float),
                    'low': candles[:, 3].astype(float),
                    'close': candles[:, 4].astype(float)
                })
                return df
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(1)

    while current_start < end_sec:
        batch = fetch_klines(current_start)
        if batch.empty:
            break
        all_data.append(batch)
        current_start = int(batch['timestamp'].max()) + int(60 * int(interval))  # advance window
        if len(batch) < limit:
            break

    df = pd.concat(all_data).drop_duplicates('timestamp')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Add final row with live price
    try:
        resp = requests.get(ticker_url).json()
        if spot:
            # prices = resp.get('result', [])
            # price_entry = next((x for x in prices if x['symbol'] == symbol), None)
            # current_price = float(price_entry['price']) if price_entry else None
            current_price = hyperliquid_price(symbol.split('USD')[0])
        else:
            # prices = resp.get('result', {}).get('list', [])
            # price_entry = next((x for x in prices if x['symbol'] == symbol), None)
            # current_price = float(price_entry['markPrice']) if price_entry else None
            current_price = hyperliquid_price(symbol.split('USD')[0])

        if current_price:
            last_row = {
                'timestamp': time.time(),
                'open': current_price,
                'high': current_price,
                'low': current_price,
                'close': current_price
            }
            df = pd.concat([df, pd.DataFrame([last_row])], ignore_index=True)

    except Exception as e:
        print(f"⚠️ Warning: Couldn't fetch current price for {symbol}: {e}")

    return df


def get_price_coingecko(coin_id='bitcoin', vs_currency='usd'):
    url = f"https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": coin_id, "vs_currencies": vs_currency}
    response = requests.get(url, params=params)
    data = response.json()
    return data[coin_id][vs_currency]


def get_historical_klines_tv(symbol, tv_symbol, exchange, n_bars=6000, last_price_from_hl=True):
    tv = TvDatafeed()

    df = tv.get_hist(symbol=tv_symbol, exchange=exchange, interval=Interval.in_daily, n_bars=n_bars)
    df['date'] = pd.to_datetime(df.index)
    df['date'] = df['date'].dt.date - pd.Timedelta(days=1)
    df.index = range(len(df))
    df = df[['date', 'open', 'high', 'low', 'close']]

    if last_price_from_hl:
        # last_price = get_price_coingecko(cg_name)
        last_price = hyperliquid_price(symbol)
        last_price_df = pd.DataFrame({
            'date': [pd.to_datetime(time.time(), unit='s').date()],
            'open': [last_price],
            'high': [last_price],
            'low': [last_price],
            'close': [last_price]
        })
        df = pd.concat([df, last_price_df])
        df.index = range(len(df))
        return df
    else:
        return df


#####################################################################################
# DOWNLOAD DATA
#####################################################################################
# import logging
# logging.getLogger('tvDatafeed').setLevel(logging.CRITICAL)

def download_data():
    #  ---------------------------------------------------------------------------
    # Portfolio hardcoded
    #  ---------------------------------------------------------------------------
    # portfolio = pd.read_excel('portfolio.xlsx')
    portfolio_dict = {
        "DEX": {
            "Coin": ["AERO", "CRV", "PUMP", "RAY", "CAKE", "SUN", "UNI"],
            "ETH TVL": [0.0, 0.7, 0.0, 0.0, 0.014, 0.0, 0.6],
            "L2 TVL": [1.0, 0.04, 0.0, 0.0, 0.036, 0.0, 0.13],
            "SOL TVL": [0.0, 0.0, 1.0, 1.0, 0.027, 0.0, 0.0],
            "Other TVL": [0.0, 0.26, 0.0, 0.0, 0.923, 1.0, 0.27],
            "FDV [M$]": [1920.0, 1540.0, 5127.0, 988.0, 990.0, 493.0, 7548.0],
            "Size": ["Large", "Large", "Large", "Mid", "Mid", "Mid", "Large"],
            "Weight": [0.0204] * 7,
        },

        "LENDING": {
            "Coin": ["AAVE", "MORPHO", "SPK", "JST", "KMNO", "COMP", "SYRUP", "XVS", "EUL"],
            "ETH TVL": [0.48, 0.35, 0.817, 0.0, 0.0, 0.58, 0.592, 0.004, 0.318],
            "L2 TVL": [0.05, 0.18, 0.179, 0.0, 0.0, 0.08, 0.0, 0.004, 0.059],
            "SOL TVL": [0.0] * 9,
            "Other TVL": [0.47, 0.47, 0.004, 1.0, 0.0, 0.34, 0.038, 0.992, 0.623],
            "FDV [M$]": [4229.0, 1674.0, 600.0, 315.0, 624.0, 3246.0, 458.0, 180.0, 2851.0],
            "Size": ["Large", "Large", "Mid", "Mid", "Mid", "Large", "Mid", "Mid", "Large"],
            "Weight": [0.0159] * 9,
        },

        "STAKING": {
            "Coin": ["LDO", "RPL", "ETHFI", "JTO", "CLOUD", "SD", "LISTA"],
            "ETH TVL": [1.0, 1.0, 1.0, 0.0, 0.0, 0.86, 0.0],
            "L2 TVL": [0.0] * 7,
            "SOL TVL": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            "Other TVL": [0.0, 0.0, 0.0, 0.0, 0.0, 0.14, 1.0],
            "FDV [M$]": [1094.0, 106.0, 1460.0, 1558.0, 107.0, 65.0, 222.0],
            "Size": ["Large", "Mid", "Large", "Large", "Mid", "Micro", "Mid"],
            "Weight": [0.0204] * 7,
        },

        "PERPS": {
            "Coin": ["ASTER", "HYPE", "JUP", "ORDER", "DYDX", "GMX", "AVNT", "DRIFT"],
            "ETH TVL": [0.059, 0.0, 0.0, 1.0, 0.143, 0.0, 0.0, 0.0],
            "L2 TVL": [0.08, 0.903, 0.0, 0.0, 0.0, 0.752, 1.0, 0.0],
            "SOL TVL": [0.026, 0.0, 1.0, 0.0, 0.0, 0.01, 0.0, 1.0],
            "Other TVL": [0.835, 0.097, 0.0, 0.0, 0.856, 0.23, 0.0, 0.245],
            "FDV [M$]": [14440.0, 42370.0, 3035.0, 193.0, 550.0, 170.0, 1504.0, 699.0],
            "Size": ["Mega", "Mega", "Large", "Mid", "Mid", "Mid", "Large", "Mid"],
            "Weight": [0.0179] * 8,
        },
        "YIELD": {
            "Coin": ["PENDLE", "CVX", "YFI", "BIFI", "FLUID"],
            "ETH TVL": [0.709, 0.862, 0.499, 0.361, 0.382],
            "L2 TVL": [0.046, 0.026, 0.032, 0.192, 0.064],
            "SOL TVL": [0.0, 0.112, 0.0, 0.447, 0.0],
            "Other TVL": [0.245, 0.0, 0.469, 0.0, 0.554],
            "FDV [M$]": [1291.0, 335.0, 187.0, 13.0, 647.0],
            "Size": ["Mid", "Large", "Mid", "Mid", "Micro"],
            "Weight": [0.0286] * 5,
        },

        "STABLE": {
            "Coin": ["ENA", "FXS", "TRX", "SKY"],
            "ETH TVL": [1.0, 0.846, 0.0, 0.0],
            "L2 TVL": [0.019, 0.0, 0.0, 0.0],
            "SOL TVL": [0.0, 0.0, 0.0, 0.0],
            "Other TVL": [0.135, 1.0, 1.0, 1.0],
            "FDV [M$]": [8564.0, 203.0, 32020.0, 6136.0],
            "Size": ["Large", "Mid", "Mega", "Large"],
            "Weight": [0.0357] * 4,
        },

        "OTHERS": {  # ORACLE, RWA, CROSS
            "Coin": ["LINK", "ONDO", "STG", "PYTH"],
            "ETH TVL": [0.5, 0.799, 0.366, 0.0],
            "L2 TVL": [0.5, 0.003, 0.233, 0.0],
            "SOL TVL": [0.0, 0.145, 0.0, 1.0],
            "Other TVL": [0.0, 0.053, 0.401, 0.0],
            "FDV [M$]": [8799.0, 176.7, 1441.0, 0.0],
            "Size": ["Mega", "Large", "Mid", "Large"],
            "Weight": [0.0357] * 4,
        },
    }

    portfolio = []
    for protocol, info in portfolio_dict.items():
        temp_df = pd.DataFrame(info)
        temp_df["Protocol type"] = protocol
        portfolio.append(temp_df)

    portfolio = pd.concat(portfolio, ignore_index=True)

    #  ---------------------------------------------------------------------------
    #  ---------------------------------------------------------------------------
    tv_data = [
        ['AERO', 'AEROUSDT', 'BYBIT'],
        ['CRV', 'CRVUSDT', 'BYBIT'],
        ['PUMP', 'PUMPUSDT', 'BYBIT'],
        ['RAY', 'RAYUSDT', 'BINANCE'],
        ['CAKE', 'CAKEUSDT', 'BYBIT'],
        ['SUN', 'SUNUSDT', 'BYBIT'],
        ['UNI', 'UNIUSDT', 'BYBIT'],
        ['AAVE', 'AAVEUSDT', 'BYBIT'],
        ['MORPHO', 'MORPHOUSDT', 'BYBIT'],
        ['SPK', 'SPKUSDT', 'BYBIT'],
        ['JST', 'JSTUSDT', 'BYBIT'],
        ['KMNO', 'KMNOUSDT', 'BINANCE'],
        ['COMP', 'COMPUSDT', 'BYBIT'],
        ['SYRUP', 'SYRUPUSDT', 'BINANCE'],
        ['XVS', 'XVSUSDT', 'BINANCE'],
        ['EUL', 'EULUSDT', 'KUCOIN'],
        ['FLUID', 'FLUIDUSDT', 'BYBIT'],
        ['LDO', 'LDOUSDT', 'BYBIT'],
        ['RPL', 'RPLUSDT', 'BYBIT'],
        ['JUP', 'JUPUSDT', 'BYBIT'],
        ['ETHFI', 'ETHFIUSDT', 'BYBIT'],
        ['JTO', 'JTOUSDT', 'BYBIT'],
        ['CLOUD', 'CLOUDUSDT', 'BYBIT'],
        ['SD', 'SDUSDT', 'BYBIT'],
        ['LISTA', 'LISTAUSDT', 'BINANCE'],
        ['ASTER', 'ASTERUSDT', 'BYBIT'],
        ['HYPE', 'HYPEUSDT', 'KUCOIN'],
        ['ORDER', 'ORDERUSDT', 'BYBIT'],
        ['DYDX', 'DYDXUSDT', 'BYBIT'],
        ['GMX', 'GMXUSDT', 'BYBIT'],
        ['AVNT', 'AVNTUSDT', 'BYBIT'],
        ['DRIFT', 'DRIFTUSDT', 'BYBIT'],
        ['PENDLE', 'PENDLEUSDT', 'BYBIT'],
        ['CVX', 'CVXUSDT', 'BINANCE'],
        ['YFI', 'YFIUSDT', 'BYBIT'],
        ['BIFI', 'BIFIUSDT', 'BINANCE'],
        ['ENA', 'ENAUSDT', 'BYBIT'],
        ['FXS', 'FXSUSDT', 'BYBIT'],
        ['TRX', 'TRXUSDT', 'BYBIT'],
        ['SKY', 'SKYUSDT', 'BYBIT'],
        ['LINK', 'LINKUSDT', 'BYBIT'],
        ['ONDO', 'ONDOUSDT', 'BYBIT'],
        ['STG', 'STGUSDT', 'BYBIT'],
        ['PYTH', 'PYTHUSDT', 'BYBIT']
    ]

    dfs = {}
    errors = []
    for c in tqdm(tv_data):
        while True:
            try:
                df = get_historical_klines_tv(c[0], c[1], c[2], n_bars=6000)
                dfs[c[0]] = df
                time.sleep(1)
                break
            except:
                print(f'Error downloading {c[0]}')
                time.sleep(5.)
                continue
    return portfolio, dfs


#####################################################################################
# DOWNLOAD DATA FOR RISK-FACTOR MODEL
#####################################################################################


def tvl_ex_price(chain, coin):
    # DeFi total TVl
    url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}"
    tvl = requests.get(url).json()
    tvl = pd.DataFrame(tvl)
    tvl["date"] = pd.to_datetime(tvl["date"], unit="s")
    tvl['date'] = tvl['date'].dt.date

    # Coin price
    coin = get_historical_klines_tv(coin.upper(), f'{coin.upper()}USDT', 'BYBIT', 6000, False)
    coin = coin[['date', 'close']]

    # TVL ex-price: TVL / price
    df = pd.merge(tvl, coin, on='date', how='inner')
    df["tvl_ex_price"] = df["tvl"]  # / df["close"]

    return df


def dex_volume():
    # Resumen DEX (histórico total) - endpoint típico de overview
    url = "https://api.llama.fi/overview/dexs?chain=All&period=all"
    dex = requests.get(url).json()
    dex_vol = pd.DataFrame(dex["totalDataChart"], columns=["ts", "dex_volume_usd"])
    dex_vol["date"] = pd.to_datetime(dex_vol["ts"], unit="s")
    dex_vol = dex_vol[["date", "dex_volume_usd"]]
    dex_vol['dex_vol'] = dex_vol['dex_volume_usd'].ewm(span=7).mean()

    return dex_vol[['date', 'dex_vol']]


def univariate_beta(coin, risk_factors, dfs, risk_factor):
    df = dfs[coin].copy()
    df = pd.merge(df, risk_factors, on='date', how='inner')

    window = 7
    y = df['close'].pct_change(window)
    X = df[risk_factor].pct_change(window)

    Xy = pd.concat([y.rename('y'), X.rename('x')], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    Y = Xy['y'].values
    Xmat = sm.add_constant(Xy['x'].values)

    ols = sm.OLS(Y, Xmat).fit()
    beta = float(ols.params[1])
    alpha0 = float(ols.params[0])

    return beta


def compute_risk_factors():
    while True:
        try:
            # ETH TVL, ex-price
            eth_tvl = tvl_ex_price('Ethereum', 'eth')
            eth_tvl['eth_tvl'] = eth_tvl['tvl_ex_price']
            eth_tvl = eth_tvl[['date', 'eth_tvl']]
            break
        except:
            time.sleep(5.)
            continue

    while True:
        try:
            # SOL TVL, ex-price
            sol_tvl = tvl_ex_price('Solana', 'sol')
            sol_tvl['sol_tvl'] = sol_tvl['tvl_ex_price']
            sol_tvl = sol_tvl[['date', 'sol_tvl']]
            break
        except:
            time.sleep(5.)
            continue

    while True:
        try:
            # DEX volume
            dex_vol = dex_volume()
            dex_vol['date'] = dex_vol['date'].dt.date
            break
        except:
            time.sleep(5.)
            continue

    while True:
        try:
            # Market beta
            btc = get_historical_klines_tv('BTC', 'BTCUSDT', 'BYBIT', 6000, False)
            btc['btc'] = btc['close']
            btc = btc[['date', 'btc']]
            break
        except:
            time.sleep(5.)
            continue

    while True:
        try:
            # Flight-to-qualith
            btc_d = get_historical_klines_tv('BTC.D', 'BTC.D', 'CRYPTOCAP', 6000, False)
            btc_d['btc_d'] = btc_d['close']
            btc_d = btc_d[['date', 'btc_d']]
            break
        except:
            time.sleep(5.)
            continue

    while True:
        try:
            # Rotation to majors
            ethbtc = get_historical_klines_tv('ETHBTC', 'ETHBTC', 'BINANCE', 6000, False)
            ethbtc['ethbtc'] = ethbtc['close']
            ethbtc = ethbtc[['date', 'ethbtc']]
            break
        except:
            time.sleep(5.)
            continue

    while True:
        try:
            # SOL ecosystem
            sol = get_historical_klines_tv('SOL', 'SOLUSDT', 'BYBIT', 6000, False)
            sol['sol'] = sol['close']
            sol = sol[['date', 'sol']]
            break
        except:
            time.sleep(5.)
            continue

    while True:
        try:
            # ETH ecosystem
            eth = get_historical_klines_tv('ETH', 'ETHUSDT', 'BYBIT', 6000, False)
            eth['eth'] = eth['close']
            eth = eth[['date', 'eth']]
            break
        except:
            time.sleep(5.)
            continue

    while True:
        try:
            # Others, altseason
            others = get_historical_klines_tv('OTHERS.D', 'OTHERS.D', 'CRYPTOCAP', 6000, False)
            others['others'] = others['close']
            others = others[['date', 'others']]
            break
        except:
            time.sleep(5.)
            continue

    risk_factors = pd.merge(eth_tvl, sol_tvl, on='date', how='inner')
    risk_factors = pd.merge(risk_factors, dex_vol, on='date', how='inner')
    risk_factors = pd.merge(risk_factors, btc, on='date', how='inner')
    risk_factors = pd.merge(risk_factors, btc_d, on='date', how='inner')
    risk_factors = pd.merge(risk_factors, ethbtc, on='date', how='inner')
    risk_factors = pd.merge(risk_factors, sol, on='date', how='inner')
    risk_factors = pd.merge(risk_factors, eth, on='date', how='inner')
    risk_factors = pd.merge(risk_factors, others, on='date', how='inner')

    return risk_factors


def compute_betas(dfs, portfolio, risk_factors):
    #####################################################################################
    # CALCULATE BETAS, FROM RISK-FACTOR MODEL
    #####################################################################################

    # --- Run ---
    betas = {
        'coin': []
    }
    for risk_factor in risk_factors.keys()[1:]:
        betas[risk_factor] = []

    for asset_name, df_asset in dfs.items():
        betas['coin'].append(asset_name)
        for risk_factor in risk_factors.keys()[1:]:
            if len(df_asset) < 100:
                betas[risk_factor].append(0)
                continue

            beta = univariate_beta(asset_name, risk_factors, dfs, risk_factor)
            betas[risk_factor].append(beta)

    betas = pd.DataFrame(betas)

    # --- Portfolio weighted beta by risk factor ---
    df_betas = {
        'risk_factor': [],
        'beta': []
    }
    for risk_factor in risk_factors.keys()[1:]:
        beta = 0
        for i, row in portfolio.iterrows():
            beta += betas.loc[betas['coin'] == row['Coin'], risk_factor].values[0] * row['Weight']

        df_betas['risk_factor'].append(risk_factor)
        df_betas['beta'].append(beta)

    df_betas = pd.DataFrame(df_betas)
    return df_betas


def build_portfolio_table(portfolio: pd.DataFrame, dfs: dict, invested_capital: float = 5e3,
                          total_vol: float = 0.5) -> pd.DataFrame:
    """
    Construye y formatea la tabla de portfolio con asignación de capital y ajustes de volatilidad.

    Args:
        portfolio (pd.DataFrame): Portfolio base (de download_data).
        dfs (dict): Diccionario {coin: DataFrame con precios}.
        invested_capital (float): Capital total invertido.
        total_vol (float): Volatilidad objetivo del portfolio.

    Returns:
        pd.DataFrame: Tabla del portfolio con métricas y formato listo para dashboard.
    """
    portfolio_table = portfolio.copy()

    capital_allocations, vol_adj = [], []
    for _, row in portfolio_table.iterrows():
        rv = dfs[row['Coin']]['close'].pct_change().ewm(span=100).std().values[-1] * np.sqrt(365)
        vol_adj.append(total_vol / rv)
        capital_allocation = total_vol * row['Weight'] / rv
        capital_allocations.append(int(capital_allocation * invested_capital))

    portfolio_table['Volatility adjustment'] = vol_adj
    portfolio_table['Capital allocation'] = capital_allocations

    # --- Aesthetics ---
    portfolio_table.rename(
        columns={
            'ETH TVL': 'ETH TVL [%]',
            'L2 TVL': 'L2 TVL [%]',
            'SOL TVL': 'SOL TVL [%]',
            'Other TVL': 'Other TVL [%]'
        },
        inplace=True
    )

    for col in ['ETH TVL [%]', 'L2 TVL [%]', 'SOL TVL [%]', 'Other TVL [%]']:
        portfolio_table[col] = portfolio_table[col].apply(lambda x: round(100 * x, 0))

    portfolio_table['FDV [M$]'] = portfolio_table['FDV [M$]'].astype(int)

    portfolio_table.rename(columns={'Weight': 'Weight [%]'}, inplace=True)
    portfolio_table['Weight [%]'] = portfolio_table['Weight [%]'].apply(lambda x: round(100 * x, 2))

    portfolio_table.rename(columns={'Volatility adjustment': 'Vol adj'}, inplace=True)
    portfolio_table['Vol adj'] = portfolio_table['Vol adj'].apply(lambda x: round(x, 2))

    portfolio_table.rename(columns={'Capital allocation': 'Capital [$]'}, inplace=True)
    portfolio_table['Capital [$]'] = portfolio_table['Capital [$]'].apply(int)

    return portfolio_table


def generate_dfs_for_plots(portfolio):
    protocol_types = np.unique(portfolio['Protocol type'])
    data = []
    for protocol_type in protocol_types:
        pf = portfolio[portfolio['Protocol type'] == protocol_type]
        data.append({
            "Protocol type": protocol_type,
            "ETH": (pf['ETH TVL'] * pf['Weight']).sum(),
            "L2":  (pf['L2 TVL']  * pf['Weight']).sum(),
            "SOL": (pf['SOL TVL'] * pf['Weight']).sum(),
            "Other": (pf['Other TVL'] * pf['Weight']).sum(),
        })
    df_protocol = pd.DataFrame(data)
    df_protocol['total'] = df_protocol[['ETH', 'L2', 'SOL', 'Other']].sum(axis=1)
    df_protocol.sort_values('total', ascending=False, inplace=True)
    
    # 2) Risk by Chain
    chains = ['ETH', 'L2', 'SOL', 'Other']
    risks_chain = [(portfolio[f'{chain} TVL'] * portfolio['Weight']).sum() for chain in chains]
    df_chain = pd.DataFrame({"Label": chains, "Risk": risks_chain})
    
    # 3) Risk by FDV
    sizes = ['Mega', 'Large', 'Mid', 'Micro']
    xaxes_titles = ['Mega (>$10B)', 'Large ($1B-10B)', 'Mid ($100M-1B)', 'Micro (<$100M)']
    risks_size = [portfolio[portfolio['Size'] == s]['Weight'].sum() for s in sizes]
    df_size = pd.DataFrame({"Label": xaxes_titles, "Risk": risks_size})

    return df_size, df_chain, df_protocol
    
    
def portfolio_backtest(dfs, portfolio_table):
    """
    Performs a backtest to plot historical performance of current portfolio, grouped by Protocol type
    """
    from datetime import datetime

    all_dates = []
    for k, df in dfs.items():
        all_dates += list(df['date'].values)
    
    all_dates = np.unique(all_dates)
    df_all = pd.DataFrame({'date': all_dates})
    for k, df in dfs.items():
        df_copy = df.copy()
        df_copy[k] = df_copy['close']
        df_all = pd.merge(df_all, df_copy[['date', k]], on='date', how='outer')
    
    df_all = df_all.iloc[:-1]
    df_all = df_all.iloc[::7]
    df_all = df_all[df_all['date'] > datetime(2025,1,1).date()]
    
    for coin in df_all.keys()[1:]:
        df_all[f'{coin}_r'] = df_all[coin].pct_change()
    
    historical_performance = pd.DataFrame({'date': df_all['date'].values})
    protocol_types = np.unique(portfolio_table['Protocol type'].values)
    for protocol_type in tqdm(protocol_types):
        mask = portfolio_table['Protocol type'] == protocol_type
        basket = portfolio_table[mask]['Coin'].values
    
        basket_pnl = []
        for coin in basket:
            mask = portfolio_table['Coin'] == coin
            real_weight = (portfolio_table[mask]['Vol adj'] * portfolio_table[mask]['Weight [%]'] / 100).values[0]
            real_pnl = df_all[f'{coin}_r'].fillna(0) * real_weight
            basket_pnl.append(real_pnl.values)
        basket_pnl = np.array(basket_pnl).sum(axis=0)
        basket_equity = (1 + basket_pnl).cumprod()
        historical_performance[protocol_type] = basket_equity

    return historical_performance


def get_returns_by_coin(dfs, portfolio_table):
    """
    Calculates returns by asset for previous 1-week, 1-month, 3-months
    """
    returns_by_coin = {
        'coin': [],
        'protocol_type': [],
        'return_1w': [],
        'return_1m': [],
        'return_3m': [],
    }
    
    dfs = results['dfs']
    for k, df in dfs.items():
        coin = k
        df_copy = pd.DataFrame(df)
        return_1w = df_copy['close'].pct_change(7).values[-1]
        return_1m = df_copy['close'].pct_change(30).values[-1]
        return_3m = df_copy['close'].pct_change(90).values[-1]

        protocol_type = portfolio_table[portfolio_table['Coin'] == coin]['Protocol type'].values[0]
    
        returns_by_coin['coin'].append(coin)
        returns_by_coin['protocol_type'].append(protocol_type)
        returns_by_coin['return_1w'].append(return_1w)
        returns_by_coin['return_1m'].append(return_1m)
        returns_by_coin['return_3m'].append(return_3m)
    
    returns_by_coin = pd.DataFrame(returns_by_coin)
    returns_by_coin.sort_values(['protocol_type', 'coin'], inplace=True)
    
    return returns_by_coin


def run_portfolio_analysis(invested_capital: float = 5e3, total_vol: float = 0.5):
    """
    Ejecuta el pipeline completo de análisis de portfolio:
    descarga datos, calcula factores de riesgo, betas y construye la tabla final.

    Args:
        invested_capital (float): Capital total invertido.
        total_vol (float): Volatilidad objetivo.

    Returns:
        dict: Resultados con portfolio, risk_factors, betas y tabla de portfolio.
    """
    # 1. Descarga portfolio + datos de mercado
    portfolio, dfs = download_data()

    # 2. Calcula factores de riesgo
    risk_factors = compute_risk_factors()

    # 3. Calcula betas
    df_betas = compute_betas(dfs, portfolio, risk_factors)

    # 4. Construye tabla del portfolio
    portfolio_table = build_portfolio_table(portfolio, dfs, invested_capital, total_vol)

    # 5. Genera dfs para plots
    df_size, df_chain, df_protocol = generate_dfs_for_plots(portfolio)
    dfs_for_plots = {
        'protocol': df_protocol,
        'size': df_size,
        'chain': df_chain,
        'betas': df_betas
    }
    
    # 6. Backtest para ver performance histórico del portfolio
    historical_performance = portfolio_backtest(dfs, portfolio_table)

    # 7. Returns for every coin
    returns_by_coin = get_returns_by_coin(dfs, portfolio_table)

    return {
        "portfolio": portfolio,
        "dfs": dfs,
        "risk_factors": risk_factors,
        "portfolio_table": portfolio_table,
        "dfs_for_plots": dfs_for_plots,
        "historical_performance": historical_performance,
        "returns_by_coin": returns_by_coin
    }
