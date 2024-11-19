import streamlit as st

from ticker import Ticker

st.markdown(
    """
    <style>
    /* Change background color */
    .stApp {
        background-color: #1f1e1e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define the app
def main():
    st.title("Stock Analysis")

    # User inputs
    ticker_symbol = st.text_input("Enter ticker symbol", "")
    if len(ticker_symbol) == 0:
        st.write('Warning no ticker selected')
    ticker_period = st.text_input("Select period", "6mo")
    forecast = st.slider("Forecast", min_value=1, max_value=10, value=3)
    RSI_period_main = st.slider("RSI Period", min_value=1, max_value=50, value=14)
    MFI_period = st.slider("MFI Period", min_value=1, max_value=50, value=14)
    BB_period = st.slider("Bollinger Bands Period", min_value=1, max_value=50, value=20)
    CH_period = st.slider("Chandelier Period", min_value=1, max_value=50, value=22)
    ATR_multiplier = st.slider("ATR Multiplier", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
    MACD_short = st.slider("MACD Short", min_value=1, max_value=50, value=12)
    MACD_long = st.slider("MACD Long", min_value=1, max_value=50, value=26)
    MACD_signal = st.slider("MACD Signal", min_value=1, max_value=50, value=9)
    DI_period = st.slider("DI Period", min_value=1, max_value=50, value=14)
    ADX_smoothing = st.slider("ADX Smoothing", min_value=1, max_value=50, value=14)
    ADX_tresh = st.slider("ADX Threshold", min_value=1, max_value=100, value=25)
    BB_smoothing = st.text_input("Bollinger Band smoother", "SMA")
    MACD_smoothing = st.text_input("MACD smoother", "EMA")


    
    # Initialize Ticker and plot
    try:
        data = Ticker(ticker_symbol, forecast, ticker_period)
        data.generate_triangle(0)
        
        # Generate plot
        fig = data.plot_ticker(forecast, RSI_period_main, MFI_period, BB_period, BB_smoothing, CH_period, ATR_multiplier,
                                MACD_short, MACD_long, MACD_signal, MACD_smoothing, DI_period, ADX_smoothing, ADX_tresh)
        if fig:
            st.plotly_chart(fig)
        else:
            st.write('Figure not generated. Please check your inputs or method implementation.')

    except Exception as e:
        st.write(f'An error occurred: {e}')


    st.markdown('''
                An explanation of the indicators and possible trading strategies can be found [here](https://docs.google.com/document/d/1_W_0cu40TYkQCfuK1ZGxy0sux4MIu6b4-8NYZkdMMFA/edit?usp=sharing).
                ''')
    

    st.markdown('''
                *Some tickers of interest*
                * AAPL: Apple
                * MSFT: Microsoft
                * NVDA: Nvidia
                * GOOGL: Alphabet
                * META: Meta
                * AMZN: Amazon
                * AMD: AMD
                * INTC: Intel
                * GM: General Motors
                * BA: Boing
                * AIR: Airbus
                * CAT: Caterpillar
                * AXP: American Express
                * V: Visa
                * KO: Coca-Cola
                * JNJ: Johnson & Johnson
                * NKE: Nike
                * RIVN: Rivian
                * TSLA: Tesla
                * LCID: Lucid
                * ^IXIC: NASDAQ
                * ^GSPC: S&P 500
                * ^DJI: Dow Jones
                * ^GDAXI: DAX
                * GC=F: Gold
                * UDVD.L: NASDA-100 UCTIS ETF CHF
                * EQCH.SW: SPDR S&P US Dividend Aristicrates ETF
                ''')

# Run the app
if __name__ == "__main__":
    main()