import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io # Needed for reading uploaded file buffer

# --- Page Configuration ---
st.set_page_config(
    page_title="BVRM Backtesting App (File Upload)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

@st.cache_data # Cache data loading
def load_data_csv(uploaded_file):
    """Loads data from user-uploaded CSV file.

    Expects a CSV file with columns: 'Date', 'Open', 'High', 'Low', 'Close'.
    'Volume' is optional but recommended.
    Date column should be parseable (e.g., YYYY-MM-DD, MM/DD/YYYY).
    """
    if uploaded_file is not None:
        try:
            # Read the file buffer into a DataFrame
            df = pd.read_csv(uploaded_file)

            # --- Data Validation ---
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must contain columns: {', '.join(required_cols)}. Please check your file.")
                return pd.DataFrame()

            # Attempt to parse date column
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except Exception:
                st.warning("Attempting to parse 'Date' column with common formats...")
                try:
                    # More robust date parsing attempt
                    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
                    if df['Date'].isna().any():
                         st.error("Some dates could not be parsed. Please ensure 'Date' column has a consistent format (e.g., YYYY-MM-DD or MM/DD/YYYY).")
                         return pd.DataFrame()
                except Exception as e:
                     st.error(f"Failed to parse 'Date' column. Please check format. Error: {e}")
                     return pd.DataFrame()

            df = df.sort_values(by='Date').reset_index(drop=True)

            # Ensure numeric types for OHLCV and handle potential errors
            for col in ['Open', 'High', 'Low', 'Close']:
                 if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN
            if 'Volume' in df.columns:
                 df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce') # Coerce errors to NaN

            # Drop rows with missing essential data (including those coerced to NaN)
            initial_rows = len(df)
            df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close'], inplace=True)
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                st.warning(f"Dropped {dropped_rows} rows due to missing/invalid essential data (Date, Open, High, Low, Close).")

            if df.empty:
                 st.error("No valid data remaining after cleaning. Please check your CSV file content.")
                 return pd.DataFrame()

            return df
        except Exception as e:
            st.error(f"Error processing CSV file: {e}. Ensure it's a valid CSV.")
            return pd.DataFrame()
    return pd.DataFrame()


def calculate_indicators(df, strategy, params):
    """Calculates technical indicators based on the selected strategy."""
    df_indicators = df.copy()
    try:
        if strategy == "MA Crossover":
            df_indicators[f'SMA_{params["ma_short"]}'] = ta.sma(df_indicators["Close"], length=params["ma_short"])
            df_indicators[f'SMA_{params["ma_long"]}'] = ta.sma(df_indicators["Close"], length=params["ma_long"])
        elif strategy == "RSI":
            df_indicators['RSI'] = ta.rsi(df_indicators["Close"], length=params["rsi_period"])
        elif strategy == "MACD":
            macd = ta.macd(df_indicators["Close"], fast=params["macd_fast"], slow=params["macd_slow"], signal=params["macd_signal"])
            if macd is not None and not macd.empty:
                 df_indicators = pd.concat([df_indicators, macd], axis=1) # Concatenate MACD columns
            else:
                 st.warning("Could not calculate MACD. Check data length and parameters.")
        elif strategy == "Bollinger Bands":
            bbands = ta.bbands(df_indicators["Close"], length=params["bb_period"], std=params["bb_std_dev"])
            if bbands is not None and not bbands.empty:
                 df_indicators = pd.concat([df_indicators, bbands], axis=1) # Concatenate BBands columns
            else:
                 st.warning("Could not calculate Bollinger Bands. Check data length and parameters.")

        df_indicators.dropna(inplace=True) # Drop rows with NaNs created by indicators
        if df_indicators.empty:
            st.warning(f"No data left after calculating {strategy} indicators. This might happen if the data period is shorter than the indicator requires.")

    except Exception as e:
        st.error(f"Error calculating indicators for {strategy}: {e}")
        return pd.DataFrame()

    return df_indicators

def generate_signals(df, strategy, params):
    """Generates buy and sell signals based on the strategy."""
    signals = pd.DataFrame(index=df.index)
    signals['Price'] = df['Close']
    signals['Buy'] = np.nan
    signals['Sell'] = np.nan

    if df.empty:
        st.warning("Cannot generate signals: Input data is empty.")
        return signals

    # Check if required columns exist before proceeding
    required_data_cols = ['Date', 'Close'] # Minimum needed
    if not all (c in df.columns for c in required_data_cols):
        st.error("Cannot generate signals: Missing required columns (Date, Close) in input data.")
        return signals

    # --- Strategy specific signal logic ---
    try:
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
             # Infer column names from params used in calculation
            macd_line = f'MACD_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
            signal_line = f'MACDs_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
            if macd_line not in df.columns or signal_line not in df.columns:
                 # Try default pandas_ta names as fallback
                 macd_line = 'MACD_12_26_9'
                 signal_line = 'MACDs_12_26_9'
                 if macd_line not in df.columns or signal_line not in df.columns:
                      st.warning(f"MACD columns ({macd_line}, {signal_line}) not found for signal generation. Check indicator calculation.")
                      return signals
                 else:
                      st.info("Using default MACD column names (MACD_12_26_9).")

            # Buy when MACD crosses above Signal line
            signals.loc[(df[macd_line] > df[signal_line]) & (df[macd_line].shift(1) <= df[signal_line].shift(1)), 'Buy'] = df['Close']
            # Sell when MACD crosses below Signal line
            signals.loc[(df[macd_line] < df[signal_line]) & (df[macd_line].shift(1) >= df[signal_line].shift(1)), 'Sell'] = df['Close']

        elif strategy == "Bollinger Bands":
            # Infer column names from params
            lower_band = f'BBL_{params["bb_period"]}_{float(params["bb_std_dev"])}' # Ensure std dev is float for name matching
            upper_band = f'BBU_{params["bb_period"]}_{float(params["bb_std_dev"])}'
            if lower_band not in df.columns or upper_band not in df.columns:
                 st.warning(f"Bollinger Band columns ({lower_band}, {upper_band}) not found for signal generation. Check indicator calculation.")
                 return signals

            # Buy when Price crosses below Lower Band
            signals.loc[(df['Close'] < df[lower_band]) & (df['Close'].shift(1) >= df[lower_band].shift(1)), 'Buy'] = df['Close']
            # Sell when Price crosses above Upper Band
            signals.loc[(df['Close'] > df[upper_band]) & (df['Close'].shift(1) <= df[upper_band].shift(1)), 'Sell'] = df['Close']

    except Exception as e:
         st.error(f"Error generating signals for {strategy}: {e}")
         return pd.DataFrame() # Return empty on error

    # --- Basic Backtest Simulation (Simplified) ---
    signals['Position'] = 0 # 0 = cash, 1 = holding stock
    signals['Portfolio_Value'] = np.nan
    signals['Return'] = 0.0

    initial_capital = 10000.0 # Starting capital
    current_capital = initial_capital
    shares = 0
    last_signal = None # Track last action 'Buy' or 'Sell'
    entry_price = 0

    # Use df index directly if available, otherwise rely on iloc
    signal_indices = signals.index

    for i in range(len(signals)):
        idx = signal_indices[i] # Get the actual index label
        buy_signal = not pd.isna(signals.loc[idx, 'Buy'])
        sell_signal = not pd.isna(signals.loc[idx, 'Sell'])
        current_position = signals.loc[signal_indices[i-1], 'Position'] if i > 0 else 0
        current_price = signals.loc[idx, 'Price']

        action_taken = False

        # --- Sell Logic ---
        if (sell_signal and current_position == 1) or (i == len(signals) - 1 and current_position == 1): # Force exit on last day if holding
            if shares > 0:
                current_capital += shares * current_price
                # Optional: Record trade profit/loss here if needed
                # st.write(f"Debug: Selling {shares:.4f} shares at {current_price:.2f} on {df.loc[idx, 'Date'].date()}")
                shares = 0
                signals.loc[idx, 'Position'] = 0
                last_signal = 'Sell'
                action_taken = True
            elif current_position == 1: # If position was 1 but shares somehow 0, still mark position as 0
                signals.loc[idx, 'Position'] = 0
                last_signal = 'Sell'
                action_taken = True

        # --- Buy Logic ---
        if buy_signal and current_position == 0 and not action_taken: # Prevent buy on same bar as sell
            if current_capital > 0 and current_price > 0:
                shares_to_buy = current_capital / current_price
                current_capital = 0
                shares = shares_to_buy
                entry_price = current_price # Record entry for potential future P/L calc
                # st.write(f"Debug: Buying {shares:.4f} shares at {current_price:.2f} on {df.loc[idx, 'Date'].date()}")
                signals.loc[idx, 'Position'] = 1
                last_signal = 'Buy'
                action_taken = True
            elif current_position == 0: # Mark intended position even if no capital / zero price
                 signals.loc[idx, 'Position'] = 1
                 last_signal = 'Buy'
                 action_taken = True


        # --- Update Position and Portfolio Value ---
        if not action_taken:
             # Carry forward previous position if no signal triggered an action
             signals.loc[idx, 'Position'] = current_position

        # Calculate portfolio value for the day
        current_holding_value = shares * current_price
        signals.loc[idx, 'Portfolio_Value'] = current_capital + current_holding_value

        # Calculate daily return
        if i > 0 and signals.loc[signal_indices[i-1], 'Portfolio_Value'] != 0:
            signals.loc[idx, 'Return'] = (signals.loc[idx, 'Portfolio_Value'] / signals.loc[signal_indices[i-1], 'Portfolio_Value']) - 1
        elif i==0:
            signals.loc[idx, 'Portfolio_Value'] = initial_capital # Start portfolio value

    # Ensure first portfolio value is set correctly and fill forward any NaNs
    if not signals.empty:
         if pd.isna(signals['Portfolio_Value'].iloc[0]):
              signals['Portfolio_Value'].iloc[0] = initial_capital
         signals['Portfolio_Value'].ffill(inplace=True)

    return signals


def plot_results(df, signals, strategy, params, file_name):
    """Creates an interactive Plotly chart."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, row_heights=[0.7, 0.3])

    if df.empty:
        st.warning("Cannot plot results: Input data for plotting is empty.")
        return fig # Return empty figure

    # --- Candlestick Chart ---
    fig.add_trace(go.Candlestick(x=df['Date'],
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'), row=1, col=1)

    # --- Add Strategy Specific Indicators ---
    try:
        if strategy == "MA Crossover":
            short_ma_col = f'SMA_{params["ma_short"]}'
            long_ma_col = f'SMA_{params["ma_long"]}'
            if short_ma_col in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df[short_ma_col], mode='lines', name=f'SMA {params["ma_short"]}', line=dict(color='orange')), row=1, col=1)
            if long_ma_col in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df[long_ma_col], mode='lines', name=f'SMA {params["ma_long"]}', line=dict(color='purple')), row=1, col=1)

        elif strategy == "Bollinger Bands":
            lower_band = f'BBL_{params["bb_period"]}_{float(params["bb_std_dev"])}'
            upper_band = f'BBU_{params["bb_period"]}_{float(params["bb_std_dev"])}'
            middle_band = f'BBM_{params["bb_period"]}_{float(params["bb_std_dev"])}'
            if upper_band in df.columns:
                 fig.add_trace(go.Scatter(x=df['Date'], y=df[upper_band], mode='lines', name='Upper Band', line=dict(color='rgba(173,204,255,0.5)')), row=1, col=1)
            if lower_band in df.columns: # Plot lower band first for filling
                 fig.add_trace(go.Scatter(x=df['Date'], y=df[lower_band], mode='lines', name='Lower Band', line=dict(color='rgba(173,204,255,0.5)')), row=1, col=1) # No fill here
            if middle_band in df.columns:
                 fig.add_trace(go.Scatter(x=df['Date'], y=df[middle_band], mode='lines', name='Middle Band (SMA)', line=dict(color='rgba(173,204,255,0.5)', dash='dash')), row=1, col=1)
            # Add fill separately if both bands exist
            if upper_band in df.columns and lower_band in df.columns:
                  fig.add_trace(go.Scatter(x=df['Date'], y=df[upper_band], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1) # Upper boundary trace
                  fig.add_trace(go.Scatter(x=df['Date'], y=df[lower_band], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(173,204,255,0.2)', showlegend=False), row=1, col=1) # Lower boundary trace with fill

        # --- Add Buy/Sell Signals to Price Chart ---
        # Map signals index to the corresponding Date in df for plotting
        buy_signals = signals[signals['Buy'].notna()]
        sell_signals = signals[signals['Sell'].notna()]

        if not buy_signals.empty:
            buy_dates = df.loc[buy_signals.index, 'Date']
            fig.add_trace(go.Scatter(x=buy_dates, y=buy_signals['Buy'], mode='markers',
                                     name='Buy Signal', marker=dict(symbol='triangle-up', size=10, color='green')), row=1, col=1)
        if not sell_signals.empty:
            sell_dates = df.loc[sell_signals.index, 'Date']
            fig.add_trace(go.Scatter(x=sell_dates, y=sell_signals['Sell'], mode='markers',
                                     name='Sell Signal', marker=dict(symbol='triangle-down', size=10, color='red')), row=1, col=1)

        # --- Add Second Row Indicator ---
        if strategy == "RSI":
            if 'RSI' in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
                fig.add_hline(y=params["rsi_upper"], line_dash="dash", line_color="red", row=2, col=1, annotation_text=f'Overbought ({params["rsi_upper"]})', annotation_position="bottom right")
                fig.add_hline(y=params["rsi_lower"], line_dash="dash", line_color="green", row=2, col=1, annotation_text=f'Oversold ({params["rsi_lower"]})', annotation_position="bottom right")
                fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1) # Set RSI range
        elif strategy == "MACD":
            macd_line = f'MACD_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
            signal_line = f'MACDs_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
            hist_line = f'MACDh_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
            # Fallback to defaults if specific names aren't found
            if macd_line not in df.columns: macd_line = 'MACD_12_26_9'
            if signal_line not in df.columns: signal_line = 'MACDs_12_26_9'
            if hist_line not in df.columns: hist_line = 'MACDh_12_26_9'

            if macd_line in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df[macd_line], mode='lines', name='MACD', line=dict(color='blue')), row=2, col=1)
            if signal_line in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df[signal_line], mode='lines', name='Signal Line', line=dict(color='orange')), row=2, col=1)
            if hist_line in df.columns:
                 colors = ['green' if val >= 0 else 'red' for val in df[hist_line]]
                 fig.add_trace(go.Bar(x=df['Date'], y=df[hist_line], name='Histogram', marker_color=colors), row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=2, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="grey", row=2, col=1) # Zero line for MACD

    except Exception as e:
        st.error(f"Error adding indicators/signals to plot: {e}")

    # --- Layout Updates ---
    fig.update_layout(
        title=f'Backtest Results for "{file_name}" ({strategy})',
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600,
        legend_title="Legend",
        hovermode='x unified'
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=1, col=1)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=2, col=1)

    return fig

def calculate_performance(signals, initial_capital):
    """Calculates basic performance metrics."""
    if signals.empty or 'Portfolio_Value' not in signals.columns or signals['Portfolio_Value'].isna().all():
        return {"Total Return (%)": 0, "Final Portfolio Value": initial_capital}

    # Ensure calculation starts from the actual first valid portfolio value
    first_valid_index = signals['Portfolio_Value'].first_valid_index()
    if first_valid_index is None: # Should not happen if initialized correctly, but safeguard
         return {"Total Return (%)": 0, "Final Portfolio Value": initial_capital}

    # Use the first valid value as the effective starting point if backtest didn't start at index 0
    effective_initial_capital = signals.loc[first_valid_index, 'Portfolio_Value']

    final_value = signals['Portfolio_Value'].iloc[-1]
    # Handle case where effective_initial_capital might be zero or NaN (though unlikely now)
    if pd.isna(effective_initial_capital) or effective_initial_capital == 0:
         total_return = 0
    else:
         total_return = ((final_value / effective_initial_capital) - 1) * 100


    metrics = {
        "Initial Capital": f"${initial_capital:,.2f}",
        "Final Portfolio Value": f"${final_value:,.2f}",
        "Total Return (%)": f"{total_return:.2f}%",
        # Add more metrics here (e.g., number of trades, win rate) - requires tracking trades in generate_signals
    }
    return metrics

# --- Streamlit Sidebar ---
st.sidebar.header("⚙️ Configuration")

# File Uploader - Now the only data source
uploaded_file = st.sidebar.file_uploader(
    "Upload Stock Data CSV",
    type=['csv'],
    help="Requires CSV with columns: Date, Open, High, Low, Close (Volume optional). Dates like YYYY-MM-DD or MM/DD/YYYY."
    )

# Date Range Input (Still useful for filtering uploaded data)
st.sidebar.subheader("Date Range Filter")
col1, col2 = st.sidebar.columns(2)
with col1:
    # Sensible defaults, adjust as needed
    default_start = pd.to_datetime("2020-01-01")
    start_date = st.date_input("Start Date", default_start)
with col2:
    default_end = pd.to_datetime("today")
    end_date = st.date_input("End Date", default_end)

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
st.title("📊 Stock Backtesting Tool (File Upload Only)")
st.markdown("""
Upload your historical stock data as a **clean CSV file** using the sidebar.
Ensure the CSV has columns: `Date`, `Open`, `High`, `Low`, `Close` (and optionally `Volume`).
The `Date` column needs a consistent format (e.g., YYYY-MM-DD).
Then, configure the strategy, parameters, and date range filter, and click 'Run Backtest'.
""")

if run_button:
    if uploaded_file is not None:
        # --- 1. Load Data ---
        df = load_data_csv(uploaded_file)
        file_name = uploaded_file.name
        st.session_state.stock_ticker = file_name # Use filename for display

        if not df.empty:
            # --- 1b. Filter Data by Date Range ---
            try:
                mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
                df_filtered = df.loc[mask].copy()
                if df_filtered.empty:
                     st.warning(f"No data available for the selected date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}. Check data file and date range.")
                else:
                     st.success(f"Loaded and filtered data for: {file_name} ({len(df_filtered)} rows)")
                     df = df_filtered # Use the filtered data from now on
            except Exception as e:
                 st.error(f"Error filtering data by date: {e}")
                 # Proceed with unfiltered data as a fallback? Or stop? Stop for now.
                 df = pd.DataFrame() # Clear df to prevent proceeding


        if not df.empty:
            # --- 2. Calculate Indicators ---
            df_processed = calculate_indicators(df.copy(), strategy, params) # Pass copy to avoid modifying filtered df

            if not df_processed.empty:
                # --- 3. Generate Signals & Run Basic Backtest ---
                signals_df = generate_signals(df_processed.copy(), strategy, params)

                if not signals_df.empty and not signals_df['Portfolio_Value'].isna().all():
                    # --- 4. Calculate Performance ---
                    initial_capital = 10000.0 # Set initial capital here
                    performance_metrics = calculate_performance(signals_df, initial_capital)

                    st.subheader("Performance Metrics")
                    perf_cols = st.columns(len(performance_metrics))
                    for idx, (key, value) in enumerate(performance_metrics.items()):
                        perf_cols[idx].metric(label=key, value=str(value))

                    # --- 5. Plot Results ---
                    st.subheader("Interactive Chart")
                    fig = plot_results(df_processed, signals_df, strategy, params, file_name)
                    st.plotly_chart(fig, use_container_width=True)

                    # --- 6. Display Data and Signals (Optional) ---
                    with st.expander("View Data and Signals"):
                        st.subheader("Filtered Data")
                        st.dataframe(df.head()) # Show head of the filtered data used
                        st.subheader("Data with Indicators")
                        st.dataframe(df_processed.head())
                        st.subheader("Signals and Portfolio")
                        st.dataframe(signals_df)

                elif signals_df.empty:
                     st.warning("No signals were generated. Check data and parameters.")
                     st.write("Data with Indicators (first 5 rows):")
                     st.dataframe(df_processed.head())
                else: # signals_df not empty, but portfolio value is all NaN (or similar issue)
                    st.warning("Backtest simulation did not produce valid portfolio values. Check data or backtest logic.")
                    st.dataframe(signals_df)


            else:
                st.warning("Could not calculate indicators or no data remained after calculation. Check data range and parameters.")
                st.write("Filtered Input Data (first 5 rows):")
                st.dataframe(df.head())

        # else: # df was empty after loading or filtering - error messages handled above
            # st.info("Load data using the sidebar and click 'Run Backtest'.")

    else:
        st.warning("Please upload a CSV file using the sidebar.")

else:
    st.info("Upload a CSV file, configure parameters in the sidebar, and click 'Run Backtest'.")
