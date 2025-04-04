# BVRM Stock Backtesting Application

This Streamlit application provides a basic framework for backtesting simple technical analysis trading strategies on stock data, with a focus on enabling analysis for BVRM (Bourse Régionale des Valeurs Mobilières - West Africa) stocks via CSV upload.

## Features

* **Data Input:**
    * Upload historical stock data via CSV (Recommended for BVRM).
    * Fetch data from Yahoo Finance (Demonstration purposes, may not work for BVRM tickers).
* **Date Range Selection:** Define the start and end dates for the backtest period.
* **Strategy Selection:** Choose from common technical strategies:
    * Moving Average (MA) Crossover
    * Relative Strength Index (RSI)
    * Moving Average Convergence Divergence (MACD)
    * Bollinger Bands
* **Parameter Customization:** Adjust indicator parameters (periods, thresholds, standard deviations) via sliders in the sidebar.
* **Signal Generation:** Automatically identifies basic Buy and Sell signals based on the chosen strategy and parameters.
* **Visualization:** Interactive Plotly chart displaying:
    * Candlestick price data
    * Selected indicators (MAs, RSI, MACD, Bollinger Bands)
    * Buy/Sell signal markers
* **Basic Backtesting:** Simulates trades based on signals (simplified logic: buy/sell on signal day's close, no costs/slippage).
* **Performance Metrics:** Displays basic results like Initial/Final Capital and Total Return %.

## Important Note on BVRM Data

Reliable, free, and easily accessible API data for BVRM stocks is often unavailable through standard libraries like `yfinance`.

**Therefore, the recommended way to use this tool for BVRM stocks is:**

1.  **Obtain Data:** Acquire historical daily stock data for your desired BVRM ticker(s) from your broker, a data vendor, or potentially the BVRM exchange website.
2.  **Format Data:** Ensure the data is in a CSV file with at least the following columns: `Date`, `Open`, `High`, `Low`, `Close`. A `Volume` column is also beneficial. The `Date` column should be in a standard, parseable format (e.g., `YYYY-MM-DD`, `MM/DD/YYYY`).
3.  **Upload CSV:** Use the "Upload CSV" option in the application's sidebar.

## Setup and Running

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <repository-folder>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

5.  **Access the App:** Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## How to Use

1.  Select your data source (Upload CSV or Yahoo Finance).
2.  If uploading, browse and select your CSV file. If using Yahoo Finance, enter a ticker symbol (e.g., `AAPL`, `^GSPC` - BVRM tickers unlikely to work here).
3.  Choose the date range for your backtest.
4.  Select the trading strategy you want to test.
5.  Adjust the parameters for the selected strategy in the sidebar.
6.  Click the "Run Backtest" button.
7.  Analyze the generated chart, performance metrics, and signal data.

## Limitations & Future Improvements

* **Backtesting Engine:** The current backtest simulation is highly simplified. It doesn't account for:
    * Transaction Costs (commissions, fees)
    * Slippage (difference between expected and execution price)
    * Partial Fills
    * Realistic Position Sizing (currently uses all capital on buy)
    * Dividend Adjustments
* **Data:** Relies on user-provided CSV for BVRM or potentially incomplete `yfinance` data.
* **Metrics:** Only basic performance metrics are calculated. Adding Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, etc., would be valuable.
* **Strategy Complexity:** Only basic implementations of standard indicators are included. More complex entry/exit logic or combined strategies could be added.
* **Risk Management:** No stop-loss or take-profit mechanisms are implemented.

This application serves as a starting point and educational tool. For serious trading decisions, use more robust backtesting libraries (like `backtrader`, `Zipline`, `VectorBT`) and consult professional financial advice.
