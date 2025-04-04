import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf # For demonstration, may not work well for BVRM

# --- Page Configuration ---
st.set_page_config(
    page_title="BVRM Backtesting App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

@st.cache_data # Cache data loading
def load_data_yf(ticker, start_date, end_date):
    """Fetches data from Yahoo Finance."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            st.error(f"Could not download data for {ticker}. Check the ticker symbol or date range.")
            return pd.DataFrame()
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error fetching data from yfinance: {e}")
        return pd.DataFrame()

@st.cache_data
def load_data_csv(uploaded_file):
    """Loads data from user-uploaded CSV file."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # --- Data Validation ---
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close'] # Volume is optional but good
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                return pd.DataFrame()

            # Attempt to parse date column (common formats)
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except Exception:
                st.warning("Could not automatically parse 'Date' column. Ensure it's in a recognizable format (e.g., YYYY-MM-DD). Trying common formats...")
                try:
                    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
                except Exception as e:
                     st.error(f"Failed to parse 'Date' column. Please check format. Error: {e}")
                     return pd.DataFrame()

            df = df.sort_values(by='Date').reset_index(drop=True)
            # Ensure numeric types for OHLCV
            for col in ['Open', 'High', 'Low', 'Close']:
                 if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'Volume' in df.columns:
                 df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close'], inplace=True) # Drop rows with essential missing data
            return df
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def calculate_indicators(df, strategy, params):
    """Calculates technical indicators based on the selected strategy."""
    df_indicators = df.copy()
    if strategy == "MA Crossover":
        df_indicators[f'SMA_{params["ma_short"]}'] = ta.sma(df_indicators["Close"], length=params["ma_short"])
        df_indicators[f'SMA_{params["ma_long"]}'] = ta.sma(df_indicators["Close"], length=params["ma_long"])
    elif strategy == "RSI":
        df_indicators['RSI'] = ta.rsi(df_indicators["Close"], length=params["rsi_period"])
    elif strategy == "MACD":
        macd = ta.macd(df_indicators["Close"], fast=params["macd_fast"], slow=params["macd_slow"], signal=params["macd_signal"])
        if macd is not None and not macd.empty:
             df_indicators = pd.concat([df_indicators, macd], axis=1) # Concatenate MACD columns
    elif strategy == "Bollinger Bands":
        bbands = ta.bbands(df_indicators["Close"], length=params["bb_period"], std=params["bb_std_dev"])
        if bbands is not None and not bbands.empty:
             df_indicators = pd.concat([df_indicators, bbands], axis=1) # Concatenate BBands columns

    df_indicators.dropna(inplace=True) # Drop rows with NaNs created by indicators
    return df_indicators

def generate_signals(df, strategy, params):
    """Generates buy and sell signals based on the strategy."""
    signals = pd.DataFrame(index=df.index)
    signals['Price'] = df['Close']
    signals['Buy'] = np.nan
    signals['Sell'] = np.nan
    position = 0 # 0 = out, 1 = in

    if strategy == "MA Crossover":
        short_ma_col = f'SMA_{params["ma_short"]}'
        long_ma_col = f'SMA_{params["ma_long"]}'
        if short_ma_col not in df.columns or long_ma_col not in df.columns:
             st.warning("MA columns not found for signal generation.")
             return signals # Return empty signals if MAs aren't calculated

        # Buy when short MA crosses above long MA
        signals.loc[(df[short_ma_col] > df[long_ma_col]) & (df[short_ma_col].shift(1) <= df[long_ma_col].shift(1)), 'Buy'] = df['Close']
        # Sell when short MA crosses below long MA
        signals.loc[(df[short_ma_col] < df[long_ma_col]) & (df[short_ma_col].shift(1) >= df[long_ma_col].shift(1)), 'Sell'] = df['Close']

    elif strategy == "RSI":
        if 'RSI' not in df.columns:
             st.warning("RSI column not found for signal generation.")
             return signals
        # Buy when RSI crosses below lower threshold
        signals.loc[(df['RSI'] < params["rsi_lower"]) & (df['RSI'].shift(1) >= params["rsi_lower"]), 'Buy'] = df['Close']
        # Sell when RSI crosses above upper threshold
        signals.loc[(df['RSI'] > params["rsi_upper"]) & (df['RSI'].shift(1) <= params["rsi_upper"]), 'Sell'] = df['Close']

    elif strategy == "MACD":
        macd_line = 'MACD_12_26_9' # Default names from pandas_ta
        signal_line = 'MACDs_12_26_9'
        if macd_line not in df.columns or signal_line not in df.columns:
             st.warning("MACD columns not found for signal generation. Check column names.")
             return signals
        # Buy when MACD crosses above Signal line
        signals.loc[(df[macd_line] > df[signal_line]) & (df[macd_line].shift(1) <= df[signal_line].shift(1)), 'Buy'] = df['Close']
        # Sell when MACD crosses below Signal line
        signals.loc[(df[macd_line] < df[signal_line]) & (df[macd_line].shift(1) >= df[signal_line].shift(1)), 'Sell'] = df['Close']

    elif strategy == "Bollinger Bands":
        lower_band = 'BBL_20_2.0' # Default names
        upper_band = 'BBU_20_2.0'
        if lower_band not in df.columns or upper_band not in df.columns:
             st.warning("Bollinger Band columns not found for signal generation. Check column names.")
             return signals
        # Buy when Price crosses below Lower Band
        signals.loc[(df['Close'] < df[lower_band]) & (df['Close'].shift(1) >= df[lower_band].shift(1)), 'Buy'] = df['Close']
        # Sell when Price crosses above Upper Band
        signals.loc[(df['Close'] > df[upper_band]) & (df['Close'].shift(1) <= df[upper_band].shift(1)), 'Sell'] = df['Close']

    # --- Basic Backtest Simulation (Simplified) ---
    # This is a basic event-driven backtest simulation.
    # It assumes buying/selling happens at the closing price of the signal day.
    # It doesn't account for slippage, commissions, or partial fills.
    # It forces selling before buying if a buy signal appears while holding.
    # It forces buying before selling if a sell signal appears while out.

    signals['Position'] = 0 # 0 = cash, 1 = holding stock
    signals['Portfolio_Value'] = np.nan
    signals['Return'] = 0.0

    initial_capital = 10000.0 # Starting capital
    current_capital = initial_capital
    shares = 0
    last_signal = None # Track last action 'Buy' or 'Sell'

    for i in range(len(signals)):
        buy_signal = not pd.isna(signals['Buy'].iloc[i])
        sell_signal = not pd.isna(signals['Sell'].iloc[i])
        current_position = signals['Position'].iloc[i-1] if i > 0 else 0
        current_price = signals['Price'].iloc[i]

        action_taken = False

        # --- Sell Logic ---
        # Sell if signal occurs AND we are currently holding OR if it's the last day and holding
        if (sell_signal and current_position == 1) or (i == len(signals) - 1 and current_position == 1):
            if shares > 0: # Ensure we have shares to sell
                current_capital += shares * current_price
                st.write(f"Debug: Selling {shares:.4f} shares at {current_price:.2f} on {df['Date'].iloc[i].date()}")
                shares = 0
                signals.loc[signals.index[i], 'Position'] = 0
                last_signal = 'Sell'
                action_taken = True
            # If sell signal occurs but not holding (or no shares), just mark position as 0
            elif current_position == 1:
                signals.loc[signals.index[i], 'Position'] = 0
                last_signal = 'Sell'
                action_taken = True


        # --- Buy Logic ---
        # Buy if signal occurs AND we are currently out of the market
        if buy_signal and current_position == 0 and not action_taken: # Prevent buy on same bar as sell
            if current_capital > 0: # Ensure we have capital
                shares_to_buy = current_capital / current_price
                current_capital = 0 # Invest all capital (simple approach)
                shares = shares_to_buy
                st.write(f"Debug: Buying {shares:.4f} shares at {current_price:.2f} on {df['Date'].iloc[i].date()}")
                signals.loc[signals.index[i], 'Position'] = 1
                last_signal = 'Buy'
                action_taken = True
            # If buy signal occurs but no capital, just mark position as 1 (though shouldn't happen with this logic)
            elif current_position == 0:
                 signals.loc[signals.index[i], 'Position'] = 1
                 last_signal = 'Buy'
                 action_taken = True

        # --- Update Position and Portfolio Value ---
        if not action_taken:
             # Carry forward previous position if no signal triggered an action
             signals.loc[signals.index[i], 'Position'] = current_position

        # Calculate portfolio value for the day
        current_holding_value = shares * current_price
        signals.loc[signals.index[i], 'Portfolio_Value'] = current_capital + current_holding_value

        # Calculate daily return (log return often preferred, but simple % for now)
        if i > 0 and signals['Portfolio_Value'].iloc[i-1] != 0:
            signals.loc[signals.index[i], 'Return'] = (signals['Portfolio_Value'].iloc[i] / signals['Portfolio_Value'].iloc[i-1]) - 1
        elif i==0:
            signals.loc[signals.index[i], 'Portfolio_Value'] = initial_capital # Start portfolio value

    # Ensure first portfolio value is set
    if not signals.empty:
         signals['Portfolio_Value'].fillna(method='ffill', inplace=True) # Fill NaNs if any

    return signals


def plot_results(df, signals, strategy, params):
    """Creates an interactive Plotly chart."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, row_heights=[0.7, 0.3])

    # --- Candlestick Chart ---
    fig.add_trace(go.Candlestick(x=df['Date'],
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'), row=1, col=1)

    # --- Add Strategy Specific Indicators ---
    if strategy == "MA Crossover":
        fig.add_trace(go.Scatter(x=df['Date'], y=df[f'SMA_{params["ma_short"]}'], mode='lines', name=f'SMA {params["ma_short"]}', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df[f'SMA_{params["ma_long"]}'], mode='lines', name=f'SMA {params["ma_long"]}', line=dict(color='purple')), row=1, col=1)
    elif strategy == "Bollinger Bands":
        lower_band = f'BBL_{params["bb_period"]}_{params["bb_std_dev"]}' # pandas_ta default name
        upper_band = f'BBU_{params["bb_period"]}_{params["bb_std_dev"]}'
        middle_band = f'BBM_{params["bb_period"]}_{params["bb_std_dev"]}'
        if upper_band in df.columns:
             fig.add_trace(go.Scatter(x=df['Date'], y=df[upper_band], mode='lines', name='Upper Band', line=dict(color='rgba(173,204,255,0.5)')), row=1, col=1)
        if lower_band in df.columns:
             fig.add_trace(go.Scatter(x=df['Date'], y=df[lower_band], mode='lines', name='Lower Band', line=dict(color='rgba(173,204,255,0.5)'), fill='tonexty', fillcolor='rgba(173,204,255,0.2)'), row=1, col=1)
        if middle_band in df.columns:
             fig.add_trace(go.Scatter(x=df['Date'], y=df[middle_band], mode='lines', name='Middle Band (SMA)', line=dict(color='rgba(173,204,255,0.5)', dash='dash')), row=1, col=1)

    # --- Add Buy/Sell Signals ---
    buy_signals = signals[signals['Buy'].notna()]
    sell_signals = signals[signals['Sell'].notna()]

    fig.add_trace(go.Scatter(x=buy_signals.index.map(df['Date']), y=buy_signals['Buy'], mode='markers',
                             name='Buy Signal', marker=dict(symbol='triangle-up', size=10, color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals.index.map(df['Date']), y=sell_signals['Sell'], mode='markers',
                             name='Sell Signal', marker=dict(symbol='triangle-down', size=10, color='red')), row=1, col=1)

    # --- Add Second Row Indicator ---
    if strategy == "RSI":
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
            fig.add_hline(y=params["rsi_upper"], line_dash="dash", line_color="red", row=2, col=1, name=f'Upper Threshold ({params["rsi_upper"]})')
            fig.add_hline(y=params["rsi_lower"], line_dash="dash", line_color="green", row=2, col=1, name=f'Lower Threshold ({params["rsi_lower"]})')
            fig.update_yaxes(title_text="RSI", row=2, col=1)
    elif strategy == "MACD":
        macd_line = 'MACD_12_26_9' # Default names
        signal_line = 'MACDs_12_26_9'
        hist_line = 'MACDh_12_26_9'
        if macd_line in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df[macd_line], mode='lines', name='MACD', line=dict(color='blue')), row=2, col=1)
        if signal_line in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df[signal_line], mode='lines', name='Signal Line', line=dict(color='orange')), row=2, col=1)
        if hist_line in df.columns:
             colors = ['green' if val >= 0 else 'red' for val in df[hist_line]]
             fig.add_trace(go.Bar(x=df['Date'], y=df[hist_line], name='Histogram', marker_color=colors), row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)

    # --- Layout Updates ---
    fig.update_layout(
        title=f'{st.session_state.stock_ticker if "stock_ticker" in st.session_state else "Stock"} Backtest ({strategy})',
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600,
        legend_title="Legend",
        hovermode='x unified' # Better hover experience
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=1, col=1)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=2, col=1)

    return fig

def calculate_performance(signals, initial_capital):
    """Calculates basic performance metrics."""
    if signals.empty or 'Portfolio_Value' not in signals.columns or signals['Portfolio_Value'].isna().all():
        return {"Total Return (%)": 0, "Final Portfolio Value": initial_capital}

    final_value = signals['Portfolio_Value'].iloc[-1]
    total_return = ((final_value / initial_capital) - 1) * 100

    # Basic metrics - can be expanded significantly (Sharpe, Sortino, Max Drawdown etc.)
    metrics = {
        "Initial Capital": f"${initial_capital:,.2f}",
        "Final Portfolio Value": f"${final_value:,.2f}",
        "Total Return (%)": f"{total_return:.2f}%",
        # Add more metrics here (e.g., number of trades, win rate)
    }
    return metrics

# --- Streamlit Sidebar ---
st.sidebar.header("⚙️ Configuration")

# Data Source Selection
data_source = st.sidebar.radio("Select Data Source", ("Upload CSV", "Yahoo Finance (Demo)"), index=0)

uploaded_file = None
ticker = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload BVRM Stock CSV", type=['csv'])
    st.sidebar.info("CSV Requirements: Columns named 'Date', 'Open', 'High', 'Low', 'Close'. 'Volume' is optional. Ensure 'Date' is parseable (e.g., YYYY-MM-DD).")
else:
    ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., ^GSPC, AAPL)", value="AAPL")
    st.sidebar.warning("Yahoo Finance may not have reliable data for BVRM stocks. Use CSV upload for BVRM.")

# Date Range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
with col2:
    end_date = st.date_input("End Date", pd.to_datetime("today"))

# Strategy Selection
strategy = st.sidebar.selectbox(
    "Select Strategy",
    ("MA Crossover", "RSI", "MACD", "Bollinger Bands")
)

# Strategy Parameters
params = {}
st.sidebar.subheader(f"{strategy} Parameters")

if strategy == "MA Crossover":
    params["ma_short"] = st.sidebar.slider("Short MA Period", 5, 100, 20)
    params["ma_long"] = st.sidebar.slider("Long MA Period", 20, 200, 50)
    if params["ma_short"] >= params["ma_long"]:
        st.sidebar.warning("Short MA period should be less than Long MA period.")
elif strategy == "RSI":
    params["rsi_period"] = st.sidebar.slider("RSI Period", 7, 50, 14)
    params["rsi_upper"] = st.sidebar.slider("RSI Overbought Threshold", 50, 90, 70)
    params["rsi_lower"] = st.sidebar.slider("RSI Oversold Threshold", 10, 50, 30)
    if params["rsi_lower"] >= params["rsi_upper"]:
         st.sidebar.warning("RSI Lower Threshold must be below Upper Threshold.")
elif strategy == "MACD":
    params["macd_fast"] = st.sidebar.slider("MACD Fast Period", 5, 50, 12)
    params["macd_slow"] = st.sidebar.slider("MACD Slow Period", 10, 100, 26)
    params["macd_signal"] = st.sidebar.slider("MACD Signal Period", 5, 50, 9)
    if params["macd_fast"] >= params["macd_slow"]:
         st.sidebar.warning("MACD Fast Period must be less than Slow Period.")
elif strategy == "Bollinger Bands":
    params["bb_period"] = st.sidebar.slider("BB Period", 5, 100, 20)
    params["bb_std_dev"] = st.sidebar.slider("BB Standard Deviation", 1.0, 4.0, 2.0, step=0.1)


# Backtest Button
run_button = st.sidebar.button("🚀 Run Backtest")

# --- Main Application Area ---
st.title("📊 BVRM Stock Backtesting Tool")
st.markdown("Use the sidebar to configure the data source, date range, strategy, and parameters.")

if run_button:
    # --- 1. Load Data ---
    df = pd.DataFrame()
    stock_display_name = "Uploaded Data"
    if data_source == "Upload CSV":
        if uploaded_file:
            df = load_data_csv(uploaded_file)
            if not df.empty:
                 # Filter by date range if specified
                 df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))].copy()
                 st.session_state.stock_ticker = uploaded_file.name # Store filename for title
                 stock_display_name = uploaded_file.name
            else:
                 st.error("Failed to load or process CSV data.")
        else:
            st.warning("Please upload a CSV file.")
    else: # Yahoo Finance
        if ticker:
            df = load_data_yf(ticker, start_date, end_date)
            st.session_state.stock_ticker = ticker # Store ticker for title
            stock_display_name = ticker
        else:
            st.warning("Please enter a ticker symbol.")

    if not df.empty:
        st.subheader(f"Backtest Results for: {stock_display_name}")
        st.markdown(f"**Strategy:** {strategy} | **Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # --- 2. Calculate Indicators ---
        try:
            df_processed = calculate_indicators(df, strategy, params)
        except Exception as e:
            st.error(f"Error calculating indicators: {e}")
            st.write("Input Data (first 5 rows):")
            st.dataframe(df.head())
            df_processed = pd.DataFrame() # Prevent further errors


        if not df_processed.empty:
            # --- 3. Generate Signals & Run Basic Backtest ---
            try:
                 signals_df = generate_signals(df_processed.copy(), strategy, params) # Use copy to avoid modifying df_processed
            except Exception as e:
                 st.error(f"Error generating signals or running backtest: {e}")
                 st.write("Processed Data with Indicators (first 5 rows):")
                 st.dataframe(df_processed.head())
                 signals_df = pd.DataFrame() # Prevent further errors


            if not signals_df.empty:
                # --- 4. Calculate Performance ---
                initial_capital = 10000.0 # Set initial capital here
                performance_metrics = calculate_performance(signals_df, initial_capital)

                st.subheader("Performance Metrics")
                perf_cols = st.columns(len(performance_metrics))
                for idx, (key, value) in enumerate(performance_metrics.items()):
                    perf_cols[idx].metric(label=key, value=str(value)) # Ensure value is string for display

                # --- 5. Plot Results ---
                st.subheader("Interactive Chart")
                try:
                     fig = plot_results(df_processed, signals_df, strategy, params)
                     st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                     st.error(f"Error plotting results: {e}")

                # --- 6. Display Data and Signals (Optional) ---
                with st.expander("View Data and Signals"):
                    st.subheader("Raw Data (Filtered by Date)")
                    st.dataframe(df.head())
                    st.subheader("Data with Indicators")
                    st.dataframe(df_processed.head())
                    st.subheader("Signals and Portfolio")
                    st.dataframe(signals_df) # Show signals and portfolio value over time

            else:
                st.warning("No signals were generated for the selected strategy and parameters.")
                st.write("Processed Data with Indicators (first 5 rows):")
                st.dataframe(df_processed.head())

        else:
            st.warning("Could not calculate indicators. Check data integrity and indicator parameters.")
            st.write("Input Data (first 5 rows):")
            st.dataframe(df.head())

    else:
        st.info("Load data and click 'Run Backtest' to see results.")

else:
    st.info("Configure parameters in the sidebar and click 'Run Backtest'.")
