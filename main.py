import pandas as pd
import talib
from backtester import BackTester


def process_data(data):
    # RSI
    delta = data['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # SMA
    data['sma_50'] = data['close'].rolling(window=50).mean()

    # MACD
    ema_12 = data['close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema_12 - ema_26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

    # Drop rows with any NaN values from indicators
    return data.dropna().reset_index(drop=True)


def strat(data):
    data['signals'] = 0  # default: no trade
    position = 0  # 1 = long, 0 = neutral

    for i in range(1, len(data)):
        rsi_prev = data['rsi'].iloc[i - 1]
        rsi_curr = data['rsi'].iloc[i]
        macd_prev = data['macd'].iloc[i - 1]
        macd_signal_prev = data['macd_signal'].iloc[i - 1]
        macd_curr = data['macd'].iloc[i]
        macd_signal_curr = data['macd_signal'].iloc[i]
        sma_50 = data['sma_50'].iloc[i]
        close = data['close'].iloc[i]

        # Entry
        if (position == 0 and
                rsi_curr >= 30 and
                macd_curr > macd_signal_curr and
                close > sma_50):

            data.at[i, 'signals'] = 1  # buy
            position = 1

        # Exit
        elif (position == 1 and
              rsi_curr <= 70 and
              macd_curr < macd_signal_curr and
              close < sma_50):

            data.at[i, 'signals'] = -1  # sell
            position = 0

        # Do nothing otherwise (no need to emit repeated 1s)
        # Keep signal = 0

    return data


def main():
    data = pd.read_csv("BTC_2019_2023_1d.csv")
    processed_data = process_data(data)  # process the data
    result_data = strat(processed_data)  # Apply the strategy
    csv_file_path = "final_data.csv"
    result_data.to_csv(csv_file_path, index=False)

    bt = BackTester("BTC", signal_data_path="final_data.csv", master_file_path="final_data.csv", compound_flag=1)
    bt.get_trades(1000)

    # print trades and their PnL
    if not bt.trades:
        print("No trades were executed. Please check your strategy logic or data.")
        return

    for trade in bt.trades:
        print(trade)
        print(trade.pnl())

    # Print results
    stats = bt.get_statistics()
    if stats is None:
        print("No statistics available. Possibly no trades were executed.")
    else:
        for key, val in stats.items():
            print(key, ":", val)

    # Check for lookahead bias
    print("Checking for lookahead bias...")
    lookahead_bias = False
    for i in range(len(result_data)):
        if result_data.iloc[i]['signals'] != 0:  # If there's a signal
            temp_data = data.iloc[:i + 1].copy()  # Use raw data up to index i
            temp_data = process_data(temp_data)  # Re-process
            temp_data = strat(temp_data)  # Re-run strategy

            if len(temp_data) > i:
                if temp_data.iloc[i]['signals'] != result_data.iloc[i]['signals']:
                    print(f"Lookahead bias detected at index {i}")
                    lookahead_bias = True

    if not lookahead_bias:
        print("No lookahead bias detected.")

    # Generate the PnL graph
    bt.make_trade_graph()
    bt.make_pnl_graph()

if __name__ == "__main__":
    main()