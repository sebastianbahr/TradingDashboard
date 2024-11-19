import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import logging
import time


class Ticker:
    def __init__(self, ticker:str, n_pivot_points: int, ticker_period: str):
        self.ticker = ticker
        self.n_pivot_points = n_pivot_points
        self.ticker_period = ticker_period


    def get_data(self, retries=3) -> pd.DataFrame:
        for attempt in range(retries):
            try:
                df = yf.Ticker(self.ticker)
                df = df.history(period=self.ticker_period, interval='1d')

                # Check if DataFrame is empty
                if df.empty:
                    raise ValueError(f"No data returned for ticker '{self.ticker}'")
                
                df = df[df.Volume > 0]
                self.data = df.reset_index()
                #return self.data
                
            except Exception as e:
                if attempt < retries -1:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(1)  # Wait before retrying
                else:
                    print(f"Failed to retrieve data for ticker '{self.ticker}' after {retries} attempts.")    


    #def get_data(self) -> pd.DataFrame: 
    #    df = yf.Ticker(self.ticker)
    #    df = df.history(period=self.ticker_period, interval='1d')
    #    df = df[df.Volume > 0]
    #    self.data = df.reset_index()


    def pivotid(self, c: int, n_left: int, n_right: int): 
        """
        The higher n_left and n_right the more certain it is pivot point represents trend change
        """
        # handle start and end values of time series
        if c-n_left < 0 or c+n_right >= len(self.data):
            return 0
        
        pividlow=True
        pividhigh=True
        # check if it is not a pivot low or high and there are higher highs or lower lows nearby
        for i in range(c-n_left, c+n_right+1):
            if(self.data.Low[c] > self.data.Low[i]) == True:
                pividlow=False
            if(self.data.High[c] < self.data.High[i]) == True:
                pividhigh=False
        if pividlow == True and pividhigh == True:
            return 3
        elif pividlow == True:
            return 1
        elif pividhigh == True:
            return 2
        else:
            return 0


    def points_position(self, x):
        if x['pivot']==1:
            return x['Low']-1e-3
        elif x['pivot']==2:
            return x['High']+1e-3
        else:
            return np.nan 
        

    def calculate_pivot_points(self):
        try:
            self.get_data()
        except Exception as error:
            print(error)

        self.data['pivot'] = self.data.apply(lambda x: self.pivotid(x.name, 4, 4), axis=1)
        self.data['point_position'] = self.data.apply(lambda row: self.points_position(row), axis=1)


    def detect_slope(self, type_: str) -> float:
        slopes = []
        rs = []
        data = np.array([])

        if type_ == 'Low':
            n = 1
        elif type_ == 'High':
            n = 2
        else:
            logging.error(f'type_ must be Low or High') 

        for i in range(3, 6):
            slope, _, r, _, _ = linregress(self.data[self.data['pivot']==n].iloc[-i:].index, self.data[self.data['pivot']==n].iloc[-i:][type_])
            slopes.append(slope)
            rs.append(abs(r))
    
        data = np.array([rs, slopes]).T
        # returns slope of fit with highest correlation
        slope = data[np.argmax(data[:,0]),1]
        
        return slope
    

    def pivot_points_lows(self, lows: list, slope:float, n_pivot_points: int):
        mins_x = []
        mins_y = []    
        
        # For each pivot point
        for i in range(0, len(lows)-n_pivot_points):

            min_y = np.array([])
            min_x = np.array([])
            start_idx = lows.get(i)
            previous_low = self.data.iloc[start_idx].Low 
            min_x = np.append(min_x, start_idx)
            min_y = np.append(min_y, previous_low)

            if slope > 0:

                # Loop through all past pivot points and select the ones that guarantee a positive slope
                for j in range(i+1, len(lows)):

                    idx = lows.get(j)
                    # Lows positive slope
                    if previous_low >= self.data.iloc[idx].Low:
                        min_x = np.append(min_x, idx)
                        min_y = np.append(min_y, self.data.iloc[idx].Low)
                        previous_low = self.data.iloc[idx].Low

                        # break out of for loop if three suitable minimas were found
                        if min_x.size==n_pivot_points:
                            break
                
            else:

                # Loop through all past pivot points and select the ones that guarantee a negative slope
                for j in range(i+1, len(lows)):

                    idx = lows.get(j)
                    # Lows negative slope
                    if previous_low <= self.data.iloc[idx].Low:
                        min_x = np.append(min_x, idx)
                        min_y = np.append(min_y, self.data.iloc[idx].Low)
                        previous_low = self.data.iloc[idx].Low

                        # break out of for loop if three suitable minimas were found
                        if min_x.size==n_pivot_points:
                            break

            if min_x.size >= n_pivot_points:
                mins_x.append(min_x)           
                mins_y.append(min_y)

        return mins_x, mins_y


    def pivot_points_highs(self, highs:list, slope:float, n_pivot_points: int):
        maxs_x = []
        maxs_y = [] 

        # For each pivot point
        for i in range(0, len(highs)-n_pivot_points):

            max_y = np.array([])
            max_x = np.array([])
            start_idx = highs.get(i)
            previous_high = self.data.iloc[start_idx].High 
            max_x = np.append(max_x, start_idx)
            max_y = np.append(max_y, previous_high)

            if slope > 0:

                # Loop through all past pivot points and select the ones that guarantee a positive slope
                for j in range(i+1, len(highs)):

                    idx = highs.get(j)
                    # Highs positive slope
                    if previous_high >= self.data.iloc[idx].High:
                        max_x = np.append(max_x, idx)
                        max_y = np.append(max_y, self.data.iloc[idx].High)
                        previous_high = self.data.iloc[idx].High

                        # break out of for loop if n suitable minimas were found
                        if max_x.size==n_pivot_points:
                            break
                
            else:

                # Loop through all past pivot points and select the ones that guarantee a negative slope
                for j in range(i+1, len(highs)):

                    idx = highs.get(j)
                    # Highs negative slope
                    if previous_high <= self.data.iloc[idx].High:
                        max_x = np.append(max_x, idx)
                        max_y = np.append(max_y, self.data.iloc[idx].High)
                        previous_high = self.data.iloc[idx].High

                        # break out of for loop if n suitable minimas were found
                        if max_x.size==n_pivot_points:
                            break

            if max_x.size >= n_pivot_points:
                maxs_x.append(max_x)           
                maxs_y.append(max_y)

        return maxs_x, maxs_y


    def select_pivot_points_highs(self):
        # sort data by date in descending order
        highs = self.data[self.data['pivot']==2].sort_values(by='Date', ascending=False).index.to_list()
        highs = dict(enumerate(highs))   
        slope = self.detect_slope('High')

        maxs_x, maxs_y = self.pivot_points_highs(highs, slope, self.n_pivot_points)

        if len(maxs_x) == 0:
            logging.error(f'Not {self.n_pivot_points} points found that match desired lows pattern. N pivot points reduced to {self.n_pivot_points-1}')

            # try to find patter with reducing requirement for pivot points
            for i in range(1, self.n_pivot_points):

                # check that at least two pivot points exist
                if self.n_pivot_points-i >= 2:
                    maxs_x, maxs_y = self.pivot_points_highs(highs, slope, self.n_pivot_points-i)

                    if len(maxs_x) != 0:
                        break

                else:
                    logging.error(f'At least two pivot points needed to fit model. Increase n_pivot_points')

        self.maxs_x = maxs_x
        self.maxs_y = maxs_y


    def select_pivot_points_lows(self):
        # sort data by date in descending order
        lows = self.data[self.data['pivot']==1].sort_values(by='Date', ascending=False).index.to_list()
        lows = dict(enumerate(lows)) 
        slope = self.detect_slope('Low')

        mins_x, mins_y = self.pivot_points_lows(lows, slope, self.n_pivot_points)

        if len(mins_x) == 0:
            logging.error(f'Not {self.n_pivot_points} points found that match desired lows pattern. N pivot points reduced to {self.n_pivot_points-1}')

            # try to find patter with reducing requirement for pivot points
            for i in range(1, self.n_pivot_points):

                # check that at least two pivot points exist
                if self.n_pivot_points-i >= 2:
                    mins_x, mins_y = self.pivot_points_lows(lows, slope, self.n_pivot_points-i)

                    if len(mins_x) != 0:
                        break

                else:
                    logging.error(f'At least two pivot points needed to fit model. Increase n_pivot_points')

        self.mins_x = mins_x
        self.mins_y = mins_y


    def value_lower_support(self, mins_x: np.array, mins_y: np.array, i: int):
        #min_x_reversed = mins_x.copy()
        #min_x_reversed.sort()
        #min_y_reversed = mins_y.copy()
        #min_y_reversed.sort()

        #x = np.array([min_x_reversed[0], min_x_reversed[i]])
        #y = np.array([min_y_reversed[0], min_y_reversed[i]])
        x = np.array([mins_x[0], mins_x[i]])
        y = np.array([mins_y[0], mins_y[i]])
        slope_min, intercep_min, _, _, _ = linregress(x, y)
        min_y_hat = slope_min*mins_x + intercep_min
        # if 0 no value outside boundary
        return np.where(mins_y - min_y_hat < -0.1, True, False).sum(), slope_min, intercep_min
    

    def value_higher_resistance(self, maxs_x: np.array, maxs_y: np.array, i: int):
        x = np.array([maxs_x[0], maxs_x[i]])
        y = np.array([maxs_y[0], maxs_y[i]])

        slope_max, intercep_max, _, _, _ = linregress(x, y)
        max_y_hat = slope_max*maxs_x + intercep_max

        # if 0 no value outside boundary
        return np.where(maxs_y - max_y_hat > 0.1, True, False).sum(), slope_max, intercep_max
    
    
    def generate_triangle(self, triangle_i: int):
        fit_min_x = None
        fit_min_y = None
        self.triangle_i = triangle_i
        self.calculate_pivot_points()
        self.select_pivot_points_lows()
        self.select_pivot_points_highs()
        mins_x_sub = self.mins_x[self.triangle_i]
        mins_y_sub = self.mins_y[self.triangle_i]

        for i in range(1, len(mins_x_sub)):
            check, slope_min, intercep_min = self.value_lower_support(mins_x_sub, mins_y_sub, i)

            if check==0:
                idx = len(mins_x_sub)-1-i
                fit_min_x = np.array([mins_x_sub[-1], mins_x_sub[idx]])
                fit_min_y = np.array([mins_y_sub[-1], mins_y_sub[idx]])
                break

        if fit_min_x is None:
            logging.error(f'No support found that suits conditions')


        fit_max_x = None
        fit_max_y = None
        maxs_x_sub = self.maxs_x[self.triangle_i]
        maxs_y_sub = self.maxs_y[self.triangle_i]

        for i in range(1, len(maxs_x_sub)):
            check, slope_max, intercep_max = self.value_higher_resistance(maxs_x_sub, maxs_y_sub, i)

            if check==0:
                idx = len(maxs_x_sub)-1-i
                fit_max_x = np.array([maxs_x_sub[-1], maxs_x_sub[idx]])
                fit_max_y = np.array([maxs_y_sub[-1], maxs_y_sub[idx]])
                break

        if fit_max_x is None:
            logging.error(f'No resistance found that suits conditions')

        self.slope_min = slope_min
        self.slope_max = slope_max
        self.intercep_min = intercep_min
        self.intercep_max = intercep_max


    def calculate_rsi(self, period:int):
        delta = self.data.Close.diff()

        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gain = gains.rolling(period).mean()
        avg_losses = losses.rolling(period).mean()

        rs = avg_gain / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi


    def calculate_bollinger_bands(self, period:int, smoothing: str):
        if smoothing == 'EMA':
            moving_avg = self.data.Close.ewm(span=period, adjust=False).mean()
            moving_sd = self.data.Close.ewm(span=period, adjust=False).std()
            upper_bband = moving_avg + (2 * moving_sd)
            lower_bband = moving_avg - (2 * moving_sd)
            BB_market = 100 * (upper_bband - lower_bband) / upper_bband

        elif smoothing == 'SMA':
            moving_avg = self.data.Close.rolling(period).mean()
            moving_sd = self.data.Close.rolling(period).std()
            upper_bband = moving_avg + (2 * moving_sd)
            lower_bband = moving_avg - (2 * moving_sd)
        
        else:
            logging.warning(f'Smoothing must be "EMA" or "SMA" but you selected {smoothing}')

        return moving_avg, upper_bband, lower_bband, (upper_bband - lower_bband)
    

    def calculate_MFI(self, period: int):
        self.data['typical_price'] = self.data.High + self.data.Low + self.data.Close
        self.data['money_flow'] = self.data.typical_price + self.data.Volume
        self.data['pos_money_flow'] = np.where(self.data.typical_price > self.data.typical_price.shift(1), self.data.money_flow, 0)
        self.data['neg_money_flow'] = np.where(self.data.typical_price < self.data.typical_price.shift(1), self.data.money_flow, 0)
        self.data['money_flow_ration'] = self.data.pos_money_flow.rolling(window=period).sum() / self.data.neg_money_flow.rolling(period).sum()
        MFI = 100 - (100 / (1 + self.data.money_flow_ration))

        return MFI


    def calculate_MACD(self, short_period: int, long_period: int, signal_period: int, smoothing: str):
        if smoothing == 'EMA':
            self.data['short'] = self.data.Close.ewm(span=short_period, adjust=False).mean()
            self.data['long'] = self.data.Close.ewm(span=long_period, adjust=False).mean()
            MACD = self.data.short - self.data.long
            MACD_signal = MACD.ewm(span=signal_period, adjust=False).mean()
            MACD_histogram = MACD - MACD_signal

        elif smoothing == 'SMA':
            self.data['short'] = self.data.Close.rolling(window=short_period).mean()
            self.data['long'] = self.data.Close.rolling(window=long_period).mean()
            MACD = self.data.short - self.data.long
            MACD_signal = MACD.rolling(window=signal_period).mean()
            MACD_histogram = MACD - MACD_signal

        else:
            logging.warning(f'Smoothing must be "EMA" or "SMA" but you selected {smoothing}')

        return MACD, MACD_signal, MACD_histogram
    

    def calculate_adx(self, DI_period: int, ADX_period: int):
        self.data['high_low'] = self.data.High - self.data.Low
        self.data['high_prev_close'] = abs(self.data.High - self.data.Close.shift(1))
        self.data['low_prev_close'] = abs(self.data.Low - self.data.Close.shift(1))
        self.data['true_range'] = self.data[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)

        self.data['+DM'] = np.where((self.data.High - self.data.High.shift(1) > self.data.Low.shift(1) - self.data.Low),
                                    np.maximum(self.data.High - self.data.High.shift(1), 0), 0)
        self.data['-DM'] = np.where((self.data.Low.shift(1) - self.data.Low > self.data.High - self.data.High.shift(1)),
                                    np.maximum(self.data.Low.shift(1) - self.data.Low, 0), 0)
         
        self.data['TR_smoothed'] = self.data['true_range'].rolling(window=DI_period).sum()
        self.data['+DM_smoothed'] = self.data['+DM'].rolling(window=DI_period).sum()
        self.data['-DM_smoothed'] = self.data['-DM'].rolling(window=DI_period).sum()

        self.data['pos_DI'] = 100 * (self.data['+DM_smoothed'] / self.data['TR_smoothed'])
        self.data['neg_DI'] = 100 * (self.data['-DM_smoothed'] / self.data['TR_smoothed'])

        trend_direction = np.where(self.data.pos_DI > self.data.neg_DI, 1, 0)

        DMX = 100 * (abs(self.data.pos_DI - self.data.neg_DI) / (self.data.pos_DI + self.data.neg_DI)) 
        ADX = DMX.rolling(window=ADX_period).mean() 

        return ADX, trend_direction
    

    def calculate_KC(self, period: int):
        self.data['high_low'] = self.data.High - self.data.Low
        self.data['high_prev_close'] = abs(self.data.High - self.data.Close.shift(1))
        self.data['low_prev_close'] = abs(self.data.Low - self.data.Close.shift(1))
        self.data['true_range'] = self.data[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
        ATR_KC = self.data.true_range.rolling(window=period).mean()
        EMA = self.data.Close.ewm(span=period, adjust=False).mean()
        KC_upper = EMA + (2 * ATR_KC) 
        KC_lower = EMA - (2 * ATR_KC) 

        return KC_upper, KC_lower


    def chandelier_exit_long(self, period: int, atr_multiplier: float):
        self.data['high_low'] = self.data.High - self.data.Low
        self.data['high_prev_close'] = abs(self.data.High - self.data.Close.shift(1))
        self.data['low_prev_close'] = abs(self.data.Low - self.data.Close.shift(1))
        self.data['true_range'] = self.data[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
        self.data['ATR'] = self.data.true_range.rolling(window=period, min_periods=1).mean()

        self.data['highest_high'] = self.data.High.rolling(period).max()
        chandelier_long = self.data.highest_high - (self.data.ATR * atr_multiplier)

        return chandelier_long
    

    def set_trade(self, n_days: int, rrr: float, quantity: int, resistance_buffer=1) -> dict:
        data = self.data.sort_values('Date').tail(n_days)
        resistance_x = np.array(data.index.to_list())
        data['resistance'] = self.slope_max*resistance_x + self.intercep_max

        #day = df[pd.to_datetime(df.Date)==date]
        data['stop_loss'] = (data.resistance - resistance_buffer)
        data['loss'] = (data.Close - data.stop_loss) * quantity
        data['gain'] = data['loss'] * rrr
        data['sell_limit'] = data.Close + (data.gain / quantity)

        output = {}

        if len(data[data.loss > 0]) > 0:
            output['set_trade'] = True

            for i in range(0, len(data)):
                output[data.Date.iloc[i].strftime('%Y-%m-%d')] = {'loss': round(data.iloc[i].loss, 3),
                                                            'gain': round(data.iloc[i].gain, 3),
                                                            'stop_loss': round(data.iloc[i].stop_loss, 3),
                                                            'sell_limit': round(data.iloc[i].sell_limit, 3)}

        else:
            output['set_trade'] = False

        print(output)


    def plot_ticker(self, forecast_days: int, RSI_period_main:int, MFI_period: int, BB_period:int, BB_smoothing: str, CH_period:int, ATR_multiplier:float,
                    MACD_short: int, MACD_long: int, MACD_signal: int, MACD_smoothing: str, DI_period: int, ADX_period: int, ADX_tresh: int):
        ma_50_days = self.data.Close.rolling(window=50).mean()
        ma_highs_20_days = self.data.High.rolling(window=20).mean()
        ma_lows_20_days = self.data.Low.rolling(window=20).mean()
        RSI_main = self.calculate_rsi(RSI_period_main) 
        RSI_short = self.calculate_rsi(5)
        RSI_long = self.calculate_rsi(50)
        MFI = self.calculate_MFI(MFI_period)
        ma_bollinger, BB_upper, BB_lower, BBW = self.calculate_bollinger_bands(BB_period, BB_smoothing)
        MACD, MACD_signal, MACD_histogram = self.calculate_MACD(MACD_short, MACD_long, MACD_signal, MACD_smoothing)
        ADX, trend_direction = self.calculate_adx(DI_period, ADX_period)
        chandelier_long = self.chandelier_exit_long(CH_period, ATR_multiplier)

        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, 
                vertical_spacing=0.05, subplot_titles=('OHCL', 'MACD (start & direction trend)', 'RSI / MFI (entry & exit signals)', 'ADX (strength & direction trend) / BBW', 'Volume'), 
                row_heights=[3.0, 1.2 ,0.9, 0.9, 0.4])
        
        fig.add_candlestick(
            x=self.data.index,
            open=self.data['Open'],
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            name='OHLC', row=1, col=1,
            increasing_line_color='green',  
            decreasing_line_color='red',   
            increasing_fillcolor='#508f51',
            decreasing_fillcolor='lightcoral')


        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['point_position'],
            mode="markers",
            marker={'size':4, 'color':"black"},
            name="pivot point"), row=1, col=1)
        
        

        xmin = np.array([self.mins_x[self.triangle_i][-1], len(self.data)+forecast_days])
        xmax = np.array([self.maxs_x[self.triangle_i][-1], len(self.data)+forecast_days])
        #xminDate = [df.iloc[int(xmin[0])].Date, df.iloc[-1].Date + timedelta(days=forecast_days)]
        #xmaxDate = [df.iloc[int(xmax[0])].Date, df.iloc[-1].Date + timedelta(days=forecast_days)]

        # Triangles
        fig.add_trace(go.Scatter(x=xmax, y=self.slope_max*xmax + self.intercep_max, mode='lines', name='resistance', line=dict(color="#4a7859")),row=1, col=1)
        fig.add_trace(go.Scatter(x=xmin, y=self.slope_min*xmin + self.intercep_min, mode='lines', name='support', line=dict(color="#ff6699")),row=1, col=1)

        # MA 50 days
        fig.add_trace(go.Scatter(x=self.data.index, y=ma_50_days, mode='lines', name='MA 50 days', line=dict(color="rgba(184, 165, 90, 0.7)")), row=1, col=1)

        # MA 20 days High and Low
        fig.add_trace(go.Scatter(x=self.data.index, y=ma_highs_20_days, mode='lines', name='MA Highs 20 days', line=dict(color="rgba(0, 102, 51, 0.5)")), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.data.index, y=ma_lows_20_days, mode='lines', name='MA Lows 20 days', line=dict(color="rgba(153, 0, 0, 0.5)")), row=1, col=1)

        # Bollinger band
        fig.add_trace(go.Scatter(x=self.data.index, y=ma_bollinger, mode='lines', name='MA 30 days', line=dict(color="rgba(89, 37, 153, 0.5)")), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.data.index, y=BB_upper, mode='lines', name='Upper Bband', line=dict(color="rgba(106, 74, 145, 0.5)", dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.data.index, y=BB_lower, mode='lines', name='Lower Bband', line=dict(color="rgba(106, 74, 145, 0.5)", dash='dot')), row=1, col=1)

        # Chandelier long
        fig.add_trace(go.Scatter(x=self.data.index, y=chandelier_long, mode='lines', name='Chandelier long', line=dict(color="rgba(207, 95, 85, 0.7)", dash='dot')), row=1, col=1)

        # MACD
        bar_colors = ['green' if val > 0 else 'red' for val in MACD_histogram.tolist()]
        fig.add_trace(go.Scatter(x=[0, len(self.data)], y=[0, 0], mode='lines', showlegend=False, line=dict(color="#a0a3b0", width=1)),row=2, col=1)
        fig.add_trace(go.Bar(x=self.data.index, y=MACD_histogram, name='Delta' ,showlegend=False, marker=dict(color=bar_colors)),row=2, col=1)
        fig.add_trace(go.Scatter(x=self.data.index, y=MACD, mode='lines', name='MACD', line=dict(color="#1D2850")),row=2, col=1)
        fig.add_trace(go.Scatter(x=self.data.index, y=MACD_signal, mode='lines', name='Signal line', line=dict(color="#ffa500")),row=2, col=1)

        # RSI and MFI
        fig.add_trace(go.Scatter(x=[0, len(self.data)], y=[25, 25], mode='lines', showlegend=False, line=dict(color="#a0a3b0", dash='dot')),row=3, col=1)
        fig.add_trace(go.Scatter(x=[0, len(self.data)], y=[75, 75], mode='lines', showlegend=False, line=dict(color="#a0a3b0", dash='dot')),row=3, col=1)
        fig.add_trace(go.Scatter(x=self.data.index, y=RSI_short, mode='lines', name='RSI 5 days', line=dict(color='rgba(235, 96, 133, 0.5)')),row=3, col=1)
        fig.add_trace(go.Scatter(x=self.data.index, y=RSI_long, mode='lines', name='RSI 50 days', line=dict(color='rgba(25, 107, 11, 0.5)')),row=3, col=1)
        fig.add_trace(go.Scatter(x=self.data.index, y=RSI_main, mode='lines', name='RSI', line=dict(color="#6d86ed")),row=3, col=1)
        fig.add_trace(go.Scatter(x=self.data.index, y=MFI, mode='lines', name='MFI', line=dict(color="#efd5ad")),row=3, col=1)

        # ADX
        line_colors_ADX = ['green' if val == 1 else 'red' for val in trend_direction.tolist()]
        fig.add_trace(go.Scatter(x=[0, len(self.data)], y=[ADX_tresh, ADX_tresh], mode='lines', showlegend=False, line=dict(color="#a0a3b0", dash='dot')),row=4, col=1)
        fig.add_trace(go.Scatter(x=self.data.index, y=BBW, mode='lines', name='BBW', line=dict(color='rgba(106, 74, 145, 0.6)')),row=4, col=1)
        fig.add_trace(go.Scatter(x=self.data.index, y=ADX, name='ADX', line=dict(color='gray'), marker=dict(color=line_colors_ADX), mode='markers+lines'),row=4, col=1)
        
        
        # Volumne
        fig.add_trace(go.Bar(x=self.data.index, y=self.data.Volume, name='Volume', marker=dict(color="#6d86ed")),row=5, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False, title=self.ticker, height=1000, width=1400,
                        xaxis=dict(tickvals=self.data.index, 
                                    ticktext=[date.strftime('%Y-%m-%d') for date in self.data.Date]),
                        paper_bgcolor='white',  # Background of the entire plot area
                        plot_bgcolor='white', # Background of the plotting area (grid area)
                        title_font=dict(color='black'),
                        legend=dict(font=dict(color='black'))) 
        fig.update_yaxes(range=[0, 100], tickvals=[0, 20, 40, 60, 80, 100], row=3, col=1)
        fig.update_yaxes(range=[0, 100], tickvals=[0, 20, 40, 60, 80, 100], row=4, col=1)
        fig.update_annotations(font=dict(color='black'))
        fig.update_xaxes(showgrid=True, showline=True, linecolor='lightgrey', gridcolor='lightgrey', tickfont=dict(color="black"))
        fig.update_yaxes(showgrid=True, showline=True, linecolor='lightgrey', gridcolor='lightgrey', tickfont=dict(color="black"))
  
        #fig.show()
        return fig