import pandas as pd
import numpy as np
import time, requests, logging
from tqdm import tqdm
from pathlib import Path
import uuid
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tvDatafeed import TvDatafeed, Interval
from hyperliquid.info import Info
from hyperliquid.utils import constants
# built-in functions
info = Info(base_url=constants.MAINNET_API_URL, skip_ws=True)
meta, ctxs = info.meta_and_asset_ctxs()
BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)
def hyperliquid_price(coin):
    try:
        idx = next(i for i, asset in enumerate(meta['universe']) if asset['name'] == coin)
        # Obtener el precio de mercado actual (mark price)
        mark_price = float(ctxs[idx]['markPx'])
        return mark_price
    except StopIteration:
        return np.nan


def get_historical_klines(symbol: str, start_date: float, interval: str = '1h', spot: bool = True, add_k: bool = False) -> pd.DataFrame:
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


def get_historical_klines_bybit(symbol: str, start_date: float, interval: str = '60', spot: bool = False) -> pd.DataFrame:
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

#############DOWNLOAD DATA#########################

def download_data(portfolio_path: str = "app/portfolio.xlsx"):
    portfolio = pd.read_excel(portfolio_path)
    
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
    for c in tqdm(tv_data):
        while True:
            try:
                df = get_historical_klines_tv(c[0], c[1], c[2], n_bars=6000)
                dfs[c[0]] = df
                time.sleep(1)
                break
            except:
                time.sleep(5.)
                continue
    return portfolio, dfs

############ RISK FACTOR ################3

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
def univariate_beta(coin, risk_factor):
    global risk_factors

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

def compute_risk_factors(dfs, portfolio):

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

            beta = univariate_beta(asset_name, risk_factor)
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
    return risk_factors, df_betas


# -------------------- PLOTS --------------------
STATIC_DIR = Path("static")

def generate_plots(portfolio, df_betas):
    portfolio = pd.read_excel('portfolio.xlsx')

    # ----------------------------------------------------------------------
    # Risk by Protocol Type
    # ----------------------------------------------------------------------
    protocol_types = np.unique(portfolio['Protocol type'])

    data = []
    for protocol_type in protocol_types:
        portfolio_filt = portfolio[portfolio['Protocol type'] == protocol_type]
        data.append({
            "Protocol type": protocol_type,
            "ETH": (portfolio_filt['ETH TVL'] * portfolio_filt['Weight']).sum(),
            "L2": (portfolio_filt['L2 TVL'] * portfolio_filt['Weight']).sum(),
            "SOL": (portfolio_filt['SOL TVL'] * portfolio_filt['Weight']).sum(),
            "Other": (portfolio_filt['Other TVL'] * portfolio_filt['Weight']).sum(),
        })

    df_protocol = pd.DataFrame(data)
    df_protocol['total'] = df_protocol[['ETH', 'L2', 'SOL', 'Other']].sum(axis=1)
    df_protocol.sort_values('total', ascending=False, inplace=True)

    # ----------------------------------------------------------------------
    # Risk by Chain
    # ----------------------------------------------------------------------
    chains = ['ETH', 'L2', 'SOL', 'Other']
    risks_chain = [(portfolio[f'{chain} TVL'] * portfolio['Weight']).sum() for chain in chains]
    df_chain = pd.DataFrame({"Label": chains, "Risk": risks_chain})

    # ----------------------------------------------------------------------
    # Risk by Market Cap
    # ----------------------------------------------------------------------
    sizes = ['Mega', 'Large', 'Mid', 'Micro']
    xaxes_titles = ['Mega (>$10B)', 'Large ($1B-10B)', 'Mid ($100M-1B)', 'Micro (<$100M)']
    risks_size = [portfolio[portfolio['Size'] == s]['Weight'].sum() for s in sizes]
    df_size = pd.DataFrame({"Label": xaxes_titles, "Risk": risks_size})

    # ----------------------------------------------------------------------
    # Combine all in one subplot (3 rows, 2 cols)
    # ----------------------------------------------------------------------
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2}, None],  # Row 1: Protocol type (colspan=2)
            [{}, {}],                # Row 2: Chain / FDV
            [{"colspan": 2}, None]   # Row 3: Betas (colspan=2)
        ],
        subplot_titles=(
            "Portfolio Risk by Protocol Type",
            "Portfolio Risk by Chain",
            "Portfolio Risk by FDV",
            "Risk Factor Betas"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    # --- Color palette ---
    colors = {
        "ETH": "#1f77b4",
        "L2": "#2ca02c",
        "SOL": "#d62728",
        "Other": "#9467bd"
    }

    # --- Plot 1: Risk by Protocol Type ---
    for chain in ["ETH", "L2", "SOL", "Other"]:
        fig.add_trace(
            go.Bar(
                x=df_protocol["Protocol type"],
                y=df_protocol[chain],
                name=chain,
                marker_color=colors[chain]
            ),
            row=1, col=1
        )

    # --- Plot 2: Risk by Chain ---
    fig.add_trace(
        go.Bar(
            x=df_chain["Label"],
            y=df_chain["Risk"],
            name="By Chain",
            marker_color="#1f77b4"
        ),
        row=2, col=1
    )

    # --- Plot 3: Risk by FDV ---
    fig.add_trace(
        go.Bar(
            x=df_size["Label"],
            y=df_size["Risk"],
            name="By FDV",
            marker_color="#17becf"
        ),
        row=2, col=2
    )

    # --- Plot 4: Betas by Risk Factor ---
    fig.add_trace(
        go.Bar(
            x=df_betas["risk_factor"],
            y=df_betas["beta"],
            name="Risk Factor Beta",
            marker_color="#636EFA"
        ),
        row=3, col=1
    )

    # ----------------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------------
    fig.update_layout(
        height=1000,
        width=1000,
        barmode="stack",
        title=dict(
            text="Portfolio Risk Overview",
            x=0.5,
            xanchor="center",
            font=dict(size=22, family="Arial", color="black")
        ),
        font=dict(size=13, family="Arial"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,          # Legend below all subplots
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=70, r=30, t=100, b=150)
    )

    # --- Axes formatting ---
    for r in [1, 2, 3]:
        for c in [1, 2]:
            fig.update_xaxes(
                showgrid=False,
                tickangle=0,
                tickfont=dict(size=12),
                title_font=dict(size=14)
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=0.5,
                gridcolor="lightgray",
                zeroline=False,
                title_font=dict(size=14)
            )

    # --- Axis titles ---
    fig.update_xaxes(title_text="Protocol Type", row=1, col=1)
    fig.update_yaxes(title_text="Risk Allocation", row=1, col=1)
    fig.update_xaxes(title_text="Chain", row=2, col=1)
    fig.update_yaxes(title_text="Risk Allocation", row=2, col=1)
    fig.update_xaxes(title_text="FDV Category", row=2, col=2)
    fig.update_yaxes(title_text="Risk Allocation", row=2, col=2)
    fig.update_xaxes(title_text="Risk Factor", row=3, col=1)
    fig.update_yaxes(title_text="Beta", row=3, col=1)

    fig.show()

    STATIC_DIR.mkdir(exist_ok=True)
    filename = f"{uuid.uuid4().hex}.png"
    out_path = STATIC_DIR / filename
    fig.write_image(out_path)

    return filename

def run_analysis():
    portfolio, dfs = download_data(BASE_DIR / "portfolio.xlsx")
    risk_factors, df_betas = compute_risk_factors(dfs, portfolio)
    plot_file = generate_plots(portfolio, df_betas)
    return {
        "ordenes": [],   # aquí si quieres añadir lógica para órdenes
        "plot_file": plot_file
    }









