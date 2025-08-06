import streamlit as st
st.set_page_config(layout="wide")
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.stats import norm
import pandas_datareader as pdr
import math

def garman_klass(data, window=252, trading_periods=252, clean=True):

    close = stock.Close.squeeze()
    high = stock.High.squeeze()
    low = stock.Low.squeeze()
    open = stock.Open.squeeze()
    
    log_hl = (high / low).apply(np.log)
    log_co = (close / open).apply(np.log)

    rs = 0.5 * log_hl ** 2 - (2 * math.log(2) - 1) * log_co ** 2

    def f(v):
        return (trading_periods * v.mean()) ** 0.5

    result = rs.rolling(window=window, center=False).apply(func=f)

    if clean:
        return result.dropna()
    else:
        return result

# Define the function to calculate Sharpe Ratio
def sharpe_ratio(return_series, N, rf):
    mean = return_series.mean() * N - rf
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma

# Define the function to calculate Sortino Ratio
def sortino_ratio(return_series, N, rf):
    mean = return_series.mean() * N - rf
    std_neg = return_series[return_series < 0].std() * np.sqrt(N)
    return mean / std_neg

# Define the function to calculate Maximum Drawdown
def max_drawdown(return_series):
    comp_ret = (return_series + 1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret / peak) - 1
    return dd.min()

# Define the function to calculate CVaR
def calculate_cvar(returns, confidence_level=0.95):
    
    cvar = returns[returns <= returns.quantile(1-confidence_level)].mean()

    return cvar

def trend_indicator(data, window = 60, trading_periods = 252):
    
    price = data['Close'].squeeze()
    high = data['High'].squeeze()
    low = data['Low'].squeeze()
    volume = data['Volume'].squeeze()

    true_range = garman_klass(data, window=window, trading_periods=trading_periods, clean=True)
    true_range = true_range.squeeze()

    basic_upper_band = ((high + low) / 2 )*(1 + true_range)
    basic_lower_band = ((high + low) / 2)*(1 - true_range)

        # Convert bands to Series we can modify
    final_upper_band = basic_upper_band.copy()
    final_lower_band = basic_lower_band.copy()

    # Initialize uptrend Series
    uptrend = pd.Series(index=data.index, dtype=bool)
    uptrend.iloc[0] = True  # Starting assumption: trend is up

    # Supertrend calculation loop
    for current in range(1, len(data)):

        previous = current - 1

        if price.iloc[current] > final_upper_band.iloc[previous]:
            uptrend.iloc[current] = True
        elif price.iloc[current] < final_lower_band.iloc[previous]:
            uptrend.iloc[current] = False
        else:
            uptrend.iloc[current] = uptrend.iloc[previous]

            if uptrend.iloc[current] and final_lower_band.iloc[current] < final_lower_band.iloc[previous]:
                final_lower_band.iloc[current] = final_lower_band.iloc[previous]

            if not uptrend.iloc[current] and final_upper_band.iloc[current] > final_upper_band.iloc[previous]:
                final_upper_band.iloc[current] = final_upper_band.iloc[previous]

    supertrend = pd.Series(index=data.index)
    supertrend[uptrend] = final_lower_band[uptrend]
    supertrend[~uptrend] = final_upper_band[~uptrend]

    return supertrend

def trading_signal(stock):

    price = stock.Close.squeeze()

    trade = trend_indicator(stock, window = 20, trading_periods = 60)
    trend = trend_indicator(stock, window = 20, trading_periods = 252)

    signal_array = np.select(
            [(price > trade) & (price > trend),
            (price < trade) & (price > trend),
            (price < trade) & (price < trend),
            (price > trade) & (price < trend) 
            ],
            ['Strong Buy', 'Buy', 'Strong Sell', 'Sell'],
            default='Neutral'
        )

    signals = pd.Series(signal_array, index=price.index, name='signal')

    return signals

# Title of the app
st.title("Risk Management Trading App")

# Subheader with links
st.markdown("""
**App created by [Gonzalo Abduca](https://www.linkedin.com/in/gonzaloabduca)**  
[Watch my other apps](https://gonzaloabduca.github.io/GonzaloPortfolio.github.io/) | [Contact me on LinkedIn](https://www.linkedin.com/in/gonzaloabduca)
""")

# Sidebar for ticker symbol input
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Ticker Symbol", value='AAPL', max_chars=10)

# Fetch company information
company_stock = yf.Ticker(ticker)
company_info = yf.Ticker(ticker).info
info = yf.Ticker(ticker).get_info()
company_name = company_info.get("shortName", "Unknown Company")

col2 = st.columns(1)[0]
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)
col7, col8 = st.columns(2)
col9, col10 = st.columns(2)
col11, col12 = st.columns(2)

# Date range buttons
time_ranges = {
    "1 Month": timedelta(days=30),
    "3 Months": timedelta(days=90),
    "6 Months": timedelta(days=180),
    "1 Year": timedelta(days=365),
    "2 Years": timedelta(days=730),
    "5 Years": timedelta(days=1825)
}

# Default time range
selected_range = "1 Year"

# Sidebar radio buttons for selecting time range
selected_range = st.sidebar.radio("Select Time Range", list(time_ranges.keys()), index=3)

# Calculate the start date based on the selected time range
end_date = datetime.now()
start_date = end_date - time_ranges[selected_range]

stock = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
stock2 = yf.download(ticker, start='1990-01-01', auto_adjust=True)

close = stock.Close.squeeze()
close2 = stock2.Close.squeeze()
open = stock.Open.squeeze()
high = stock.High.squeeze()
low = stock.Low.squeeze()

end = datetime.now()
start = end - timedelta(days=365*10)

cagr_end = datetime.now()
cagr_start = cagr_end - timedelta(days=365*10)

# Fetch and calculate signals
signal_data = trading_signal(stock2).iloc[-1]

#############################################################################
##################### CANDLESTICK CHART #####################################
#############################################################################

# with col1:
#     # Create candlestick chart
#     fig = go.Figure(data=[go.Candlestick(x=stock.index,
#                                         open= open,
#                                         high= high,
#                                         low= low,
#                                         close= close)])

#     close_price = close.iloc[-1]

#     # Update layout for better visualization
#     fig.update_layout(title=f'Candlestick Chart for {company_name} ({ticker}) - {selected_range} - Last Price: $ {close_price:.2f}',
#                     xaxis_title='Date',
#                     yaxis_title='Price',
#                     xaxis_rangeslider_visible=False,
#                     yaxis_type='log')

#     # Display the chart in the Streamlit app
#     st.plotly_chart(fig, use_container_width=True)



#############################################################################
##################### TARGETS CHART #########################################
#############################################################################

with col2:

    st.markdown(f"### Price Chart with Analysts Targets for {company_name} - Current Price:${company_info.get('currentPrice')}")
    targets_data= {
        "Current Price": company_info.get('currentPrice'),
        "Target High Price": company_info.get('targetHighPrice'),
        "Target Low Price": company_info.get('targetLowPrice'),
        "Target Mean Price": company_info.get('targetMeanPrice'),
        "Target Median Price": company_info.get('targetMedianPrice')
        }

    analysts_targets = pd.DataFrame([targets_data])

    price = stock.Close.squeeze()

    cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=2)
    filtered_index = price.loc[price.index >= cutoff_date].index
    filtered_value = price.loc[price.index >= cutoff_date].values

    fig10 = go.Figure()

    fig10.add_trace(go.Scatter(
        x=price.index,
        y=price.values,
        mode='lines',
        name='Stock Price',
        line=dict(width=2)
    ))

    colors = {
        "Current Price": "blue",
        "Target High Price": "green",
        "Target Low Price": "red",
        "Target Mean Price": "orange",
        "Target Median Price": "purple"
    }

    for label, value in targets_data.items():
        if value:  # solo si el valor existe
            fig10.add_trace(go.Scatter(
                x=[price.index.min(), price.index.max()],
                y=[value, value],
                mode='lines',
                name=label,
                line=dict(dash='dash', color=colors.get(label, 'gray')),
                hoverinfo='text',
                text=f'{label}: {value}'
            ))

    fig10.update_layout(
        title="Analyst Targets",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",  # opcional, se puede cambiar
        height=600
    )

    st.plotly_chart(fig10, use_container_width=True)



#############################################################################
########################### PERFORMNACE CARDS ###############################
#############################################################################


# Calculate performance for specified periods
def calculate_performance(price, period_days):
    if len(price) < period_days:
        return None
    return (price[-1] - price[-period_days]) / price[-period_days] * 100

performances = {
    "1 Month": calculate_performance(close2, 30),
    "3 Months": calculate_performance(close2, 90),
    "6 Months": calculate_performance(close2, 180),
    "1 Year": calculate_performance(close2, 365)
}

with col2:
    # Display performances
    st.markdown("### Performance")
    performance_cols = st.columns(len(performances))
    for i, (label, performance) in enumerate(performances.items()):
        if performance is not None:
            color = "green" if performance > 0 else "red"
            performance_cols[i].markdown(f"<span style='color:{color}'>{label}: {performance:.2f}%</span>", unsafe_allow_html=True)
        else:
            performance_cols[i].markdown(f"{label}: N/A")


#############################################################################
########################### TRADING SIGNALS #################################
#############################################################################

with col2:
    st.markdown(f"#### Trading Algorithm Signal for {company_name}: {signal_data}")

#############################################################################
########################### BENCHMARK METRICS ###############################
#############################################################################

# Calculate performance metrics for the ticker and benchmarks
benchmarks = ['SPY', 'QQQ']
tickers = benchmarks + [ticker]

# Download the data for benchmarks and ticker
data_benchmarks = yf.download(tickers, auto_adjust=True)['Close'].dropna()

# Calculate daily returns
returns = data_benchmarks.pct_change().dropna()

# Calculate performance metrics
metrics = ['Total Return', 'Annualized Return', 'Standard Deviation', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'CVar', 'Maximum Drawdown', 'Kurtosis', 'Skewness']
performance_df = pd.DataFrame(index=metrics, columns=tickers)

# Total Return
performance_df.loc['Total Return'] = (data_benchmarks.iloc[-1] / data_benchmarks.iloc[0] - 1) * 100

# Annualized Return
performance_df.loc['Annualized Return'] = ((1 + performance_df.loc['Total Return'] / 100) ** (255 / len(data_benchmarks)) - 1) * 100

# Standard Deviation
performance_df.loc['Standard Deviation'] = returns.std() * np.sqrt(252) * 100

# Sharpe Ratio
performance_df.loc['Sharpe Ratio'] = returns.apply(lambda x: sharpe_ratio(x, 255, 0.01))

# Sortino Ratio
performance_df.loc['Sortino Ratio'] = returns.apply(lambda x: sortino_ratio(x, 255, 0.01))

# Calmar Ratio
max_drawdowns = returns.apply(max_drawdown)
performance_df.loc['Calmar Ratio'] = returns.mean() * 255 / abs(max_drawdowns)

# CVaR
performance_df.loc['CVar'] = returns.apply(calculate_cvar) * 100

# Maximum Drawdown
performance_df.loc['Maximum Drawdown'] = max_drawdowns * 100

# Kurtosis
performance_df.loc['Kurtosis'] = returns.kurtosis()

# Skewness
performance_df.loc['Skewness'] = returns.skew()

# Format as percentages with 2 decimal places for specific metrics
percentage_metrics = ['Total Return', 'Annualized Return', 'Standard Deviation', 'CVar', 'Maximum Drawdown']
performance_df.loc[percentage_metrics] = performance_df.loc[percentage_metrics].applymap(lambda x: f"{x:.2f}%")

# Format other metrics as floats with 2 decimal places
float_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Kurtosis', 'Skewness']  # Corrected 'Kurtrosis' to 'Kurtosis'
performance_df.loc[float_metrics] = performance_df.loc[float_metrics].applymap(lambda x: f"{x:.2f}")


# Display the DataFrame in the app
with col4:
    st.markdown("### Benchmark Performance Metrics")
    st.table(performance_df)

#############################################################################
########################### PERFORMACE METRICS ##############################
#############################################################################

monthly_returns = stock.Close.squeeze().resample('M').last().pct_change().dropna()

positive_monthly_returns = monthly_returns[monthly_returns > 0]
negative_monthly_returns = monthly_returns[monthly_returns < 0]

average_positive_monthly_return = positive_monthly_returns.mean() * 100
average_negative_monthly_return = negative_monthly_returns.mean() * 100

# Calculate Percentage of Positive and Negative Months
total_months = len(monthly_returns)
positive_months = len(positive_monthly_returns)
negative_months = len(negative_monthly_returns)

percentage_positive_months = (positive_months / total_months) * 100
percentage_negative_months = (negative_months / total_months) * 100

# Calculate Risk Reward Profile
if average_negative_monthly_return != 0:
    risk_reward_ratio = abs(average_positive_monthly_return / average_negative_monthly_return)
else:
    risk_reward_ratio = np.nan  # Avoid division by zero


# Calculate the 75th and 25th percentiles
percentile_75 = monthly_returns.quantile(0.95)
percentile_25 = monthly_returns.quantile(0.05)

# Calculate the average return above the 75th percentile
average_above_75th = monthly_returns[monthly_returns > percentile_75].mean()

# Calculate the average return below the 25th percentile
average_below_25th = monthly_returns[monthly_returns < percentile_25].mean()

# Calculate the risk-reward ratio
if average_below_25th != 0:
    risk_reward_ratio_percentiles = average_above_75th / abs(average_below_25th)
else:
    risk_reward_ratio_percentiles = np.nan  # Avoid division by zero

# Calculate Mathematical Expectation
average_win = average_positive_monthly_return / 100
average_loss = abs(average_negative_monthly_return / 100)
winning_percentage = percentage_positive_months / 100
losing_percentage = percentage_negative_months / 100

expectation = (average_win * winning_percentage) - (average_loss * losing_percentage)

# Best Month and Worst Month
best_month = monthly_returns.max() * 100
worst_month = monthly_returns.min() * 100

# Define metrics
monthly_performance_metrics = ['Average Positive Monthly Return', 'Average Negative Monthly Return', 'Percentage of Positive Months', 
                               'Percentage of Negative Months', 'Best Month', 'Worst Month', 'Risk Reward Profile', 
                               'Tail Ratio', 'Mathematical Expectation (E(x))']

# Initialize the DataFrame
monthly_performance_metrics_df = pd.DataFrame(index=monthly_performance_metrics, columns=[company_name])

# Populate the DataFrame
monthly_performance_metrics_df.loc['Average Positive Monthly Return'] = average_positive_monthly_return
monthly_performance_metrics_df.loc['Average Negative Monthly Return'] = average_negative_monthly_return
monthly_performance_metrics_df.loc['Percentage of Positive Months'] = percentage_positive_months
monthly_performance_metrics_df.loc['Percentage of Negative Months'] = percentage_negative_months
monthly_performance_metrics_df.loc['Best Month'] = best_month
monthly_performance_metrics_df.loc['Worst Month'] = worst_month
monthly_performance_metrics_df.loc['Risk Reward Profile'] = risk_reward_ratio
monthly_performance_metrics_df.loc['Tail Ratio'] = risk_reward_ratio_percentiles
monthly_performance_metrics_df.loc['Mathematical Expectation (E(x))'] = expectation

# Format percentage metrics
percentage_metrics = ['Average Positive Monthly Return', 'Average Negative Monthly Return', 'Percentage of Positive Months', 
                      'Percentage of Negative Months', 'Best Month', 'Worst Month']
monthly_performance_metrics_df.loc[percentage_metrics] = monthly_performance_metrics_df.loc[percentage_metrics].applymap(lambda x: f"{x:.2f}%")

# Format other metrics as floats with 2 decimal places
monthly_float_metrics = ['Risk Reward Profile', 'Tail Ratio', 'Mathematical Expectation (E(x))']
monthly_performance_metrics_df.loc[monthly_float_metrics] = monthly_performance_metrics_df.loc[monthly_float_metrics].applymap(lambda x: f"{x:.2f}")

with col3:
    # Display the DataFrame in the app
    st.markdown("### Monthly Performance Metrics")
    st.table(monthly_performance_metrics_df)


with col6:
    st.markdown("### Volume Metrics")

    # Calculate Historical Relative Volume
    avg_vol_1m = stock['Volume'].squeeze().rolling(window=20, min_periods = 1).mean()
    avg_vol_6m = stock['Volume'].squeeze().rolling(window=126, min_periods = 1).mean()
    relative_volume = (avg_vol_1m / avg_vol_6m).bfill()
    current_rvol = relative_volume.iloc[-1]
    # Create relative volume chart
    fig2 = go.Figure(data=[go.Scatter(x=relative_volume.index, y=relative_volume, mode='lines')])

    # Update layout for better visualization
    fig2.update_layout(title='Historical Relative Volume',
                    xaxis_title='Date',
                    yaxis_title='Relative Volume')
    st.plotly_chart(fig2)

    # Calculate volume change for specified periods
    def calculate_volume_change(data, period_days):

        volume = data.Volume.squeeze()

        if len(data) < period_days:
            return None
        return (volume[-1] - volume[-period_days]) / volume[-period_days] * 100

    volume_changes = {
        "1 Month": calculate_volume_change(stock2, 30),
        "3 Months": calculate_volume_change(stock2, 90),
        "6 Months": calculate_volume_change(stock2, 180),
        "1 Year": calculate_volume_change(stock2, 365)
    }

    # Display volume changes
    st.markdown(f"**Current Relative Volume**: {current_rvol:.2f}")
    volume_change_cols = st.columns(len(volume_changes))
    for i, (label, change) in enumerate(volume_changes.items()):
        if change is not None:
            color = "green" if change > 0 else "red"
            volume_change_cols[i].markdown(f"<span style='color:{color}'>{label}: {change:.2f}%</span>", unsafe_allow_html=True)
        else:
            volume_change_cols[i].markdown(f"{label}: N/A")

volatility = garman_klass(stock2, window=20, trading_periods=252, clean=True)

# Create 30-day annualized volatility chart
fig3 = go.Figure(data=[go.Scatter(x=volatility.index, y=volatility, mode='lines')])

# Update layout for better visualization
fig3.update_layout(title='1 Month Annualized Volatility',
                xaxis_title='Date',
                yaxis_title='Annualized Volatility')

with col5:
    # Display the 30-day annualized volatility chart in the Streamlit app
    st.plotly_chart(fig3)

    def calculate_volatility_change(data, period_days):
        if len(data) < period_days:
            return None
        return (data.iloc[-1] - data.iloc[-period_days]) / data.iloc[-period_days] * 100

    volatility_changes = {
        "1 Month": calculate_volatility_change(volatility, 30),
        "3 Months": calculate_volatility_change(volatility, 90),
        "6 Months": calculate_volatility_change(volatility, 180),
        "1 Year": calculate_volatility_change(volatility, 365)
    }

    current_volatility = volatility.squeeze().iloc[-1] * 100

    # Display current volatility and changes
    st.markdown("### 1-Month Annualized Volatility")
    st.markdown(f"**Current 30-Day Annualized Volatility**: {current_volatility:.2f}%")

    volatility_change_cols = st.columns(len(volatility_changes))
    for i, (label, change) in enumerate(volatility_changes.items()):
        if change is not None:
            color = "green" if change > 0 else "red"
            volatility_change_cols[i].markdown(f"<span style='color:{color}'>{label}: {change:.2f}%</span>", unsafe_allow_html=True)
        else:
            volatility_change_cols[i].markdown(f"{label}: N/A")


#############################################################################
######################## INSTITUTIONAL HOLDERS ##############################
#############################################################################

grades = yf.Ticker(ticker).get_upgrades_downgrades()

grades = grades.loc[grades.index > datetime.now() - timedelta(days=365)]

grades['Action'] = grades['Action'].replace({
    'main': 'Maintain',
    'reit': 'Reiterated',
    'up': 'Upgrade',
    'down': 'Downgrade',
    'init': 'Initiated Coverage'
})

final_grades = grades['Action'].value_counts().to_frame('Count')

color_map = {
    'Upgrade': '#00B86B',            # Dollar green
    'Downgrade': '#FF4C4C',          # Red
    'Maintain': '#1E90FF',           # Light blue
    'Reiterated': '#104E8B',         # Dark blue
    'Initiated Coverage': '#FFA500'  # Orange
}

colors = final_grades.index.map(color_map)

# Create horizontal bar chart
fig_grades = go.Figure(go.Bar(
    x=final_grades['Count'],
    y=final_grades.index,
    orientation='h',
    marker_color=colors
))

# Update layout
fig_grades.update_layout(
    title=f'Top Institutional Analyst Upgrades/ Downgrades for {company_name}',
    yaxis=dict(categoryorder='total ascending'),
    template='plotly_dark',
    xaxis_title='Number of Analysts',
    yaxis_title=''
)


institutional_holders = yf.Ticker(ticker).get_institutional_holders()
# Sort by pctHeld descending
institutional_holders = institutional_holders.sort_values(by='pctHeld', ascending=True)

# Create horizontal bar chart
fig_holders = go.Figure(go.Bar(
    x=institutional_holders['pctHeld']*100,
    y=institutional_holders['Holder'],
    orientation='h',
    marker_color = '#85BB65'
))

# Update layout
fig_holders.update_layout(
    title=f'Top Institutional Holders of {company_name}',
    yaxis=dict(categoryorder='total ascending'),
    template='plotly_dark',
    xaxis_title='Percentage Held',
    yaxis_title=''
)

with col9:
    st.plotly_chart(fig_holders)
with col7:
    st.plotly_chart(fig_grades)

#############################################################################
######################## POSITION SIZING CALCULATOR #########################
#############################################################################

def montecarlo_simulation(data, vol_spike_factor=0):

    close = data['Close'].squeeze()
    returns = close.pct_change().dropna()
    daily_vol = returns.std()
    shocked_daily_vol = daily_vol * (1 + vol_spike_factor)
    annual_vol = shocked_daily_vol * np.sqrt(252)
    current_price = close.iloc[-1]

    T_days = 252
    n_simulations = 10000
    n_steps = T_days
    dt = 1 / 252

    np.random.seed(42)
    paths = np.zeros((n_steps, n_simulations))
    paths[0] = current_price

    for t in range(1, n_steps):
        rand = np.random.standard_normal(n_simulations)
        drift = (-0.5 * annual_vol**2) * dt
        diffusion = annual_vol * np.sqrt(dt) * rand
        paths[t] = paths[t - 1] * np.exp(drift + diffusion)

    final_prices = paths

    return final_prices

montecarlo = montecarlo_simulation(stock)


# Add a radio button to select Long or Short position
position_type = st.sidebar.radio("Select Position Type", options=["Long", "Short"])

# Inputs for capital and risk percentage
capital = st.sidebar.number_input("Total Capital ($)", value=10000)
risk_percentage = st.sidebar.number_input("Percentage of Capital at Risk (%)", value=1.0)

# Fetch the last adjusted close price from data2
last_close_price = close.iloc[-1]

# Adjust the CVaR and Stop Loss calculations based on the position type
if position_type == "Long":
    # Calculate CVaR at 80% confidence level for Long positions (negative tail)
    stop_loss = ((montecarlo[20][(np.percentile(montecarlo[20], 5)) >= montecarlo[20]]).mean()).round(2)

    risk = abs(last_close_price - stop_loss) / last_close_price

    # Position size calculation for Long
    risk_amount = (risk_percentage / 100) * capital
    position_size = (risk_amount / abs(last_close_price - stop_loss)).round(0)

    target1 = (1 * np.std(montecarlo[60]) + np.mean(montecarlo[60])).round(2)
    reward1 = abs(last_close_price - target1) / last_close_price
    target2 = (2 * np.std(montecarlo[60]) + np.mean(montecarlo[60])).round(2)
    reward2 = abs(last_close_price - target2) / last_close_price

else:  # Short Position
    # Calculate CVaR at 80% confidence level for Long positions (negative tail)
    stop_loss = ((montecarlo[20][(np.percentile(montecarlo[20], 95)) <= montecarlo[20]]).mean()).round(2)

    risk = abs(last_close_price - stop_loss) / last_close_price

    # Position size calculation for Long
    risk_amount = (risk_percentage / 100) * capital
    position_size = (risk_amount / abs(last_close_price - stop_loss)).round(0)
    target1 = ((-1 * np.std(montecarlo[60])) + np.mean(montecarlo[60])).round(2)
    reward1 = abs(last_close_price - target1) / last_close_price
    target2 = (-2 * np.std(montecarlo[60]) + np.mean(montecarlo[60])).round(2)
    reward2 = abs(last_close_price - target2) / last_close_price

analyst_rating = info.get('averageAnalystRating')
dividend_yield = info.get('dividendYield')
current_price = info.get('currentPrice')
days_to_cover = info.get('shortRatio')
# Define the labels
position_size_metrics = [
    'Position Bias',
    'Algorithm Signal',
    'Analyst Rating',
    'Dividend Yield',
    'Days to Cover',
    'Risk Amount',
    'Position Size',
    'Capital to Deploy',
    'Current Price',
    'Stop Loss',
    'First Target 1Std',
    'Second Target 2Std'
]

position_size_df = pd.DataFrame(index=position_size_metrics, columns=[company_name])

position_size_df.loc['Position Bias'] = position_type                                                                                                                                                                
position_size_df.loc['Algorithm Signal'] = signal_data
position_size_df.loc['Analyst Rating'] = analyst_rating
position_size_df.loc['Days to Cover'] = f"{days_to_cover} days"
position_size_df.loc['Dividend Yield'] = f"{dividend_yield}%"
position_size_df.loc['Risk Amount'] = f"${round(risk_amount, 2)}"
position_size_df.loc['Position Size'] = round(position_size, 0)
position_size_df.loc['Capital to Deploy'] = f'${round((round(position_size, 0) * current_price), 2)}'
position_size_df.loc['Current Price'] = f'${round(last_close_price, 2)}'
position_size_df.loc['Stop Loss'] = f"${round(stop_loss, 2)} / {round(risk*100, 2)}%"
position_size_df.loc['First Target 1Std'] = f"${target1} / {round(reward1*100,2)}%"
position_size_df.loc['Second Target 2Std'] = f"${target2} / {round(reward2*100,2)}%"


with col11:
    # Display the CVaR and position size
    st.markdown("### Monte Carlo Position Size Calculator")
    st.table(position_size_df)



def safe_round(value, multiplier=1, decimals=2, suffix=''):
    if value is None:
        return 'N/A'
    try:
        return f"{round(value * multiplier, decimals)}{suffix}"
    except:
        return 'N/A'

# Extract metrics

ebit = company_stock.quarterly_income_stmt.loc['EBIT'].iloc[0] if 'EBIT' in company_stock.quarterly_income_stmt.index else np.nan
ev = company_stock.info.get('enterpriseValue', np.nan)
ebit_yield = ebit/ev
fcf_yield = company_stock.info.get('freeCashflow', np.nan) / company_stock.info.get('enterpriseValue', np.nan)
marketCap = (company_info.get('marketCap'))
trailingPE = company_info.get('trailingPE')
forwardPE = company_info.get('forwardPE')
revenueGrowth = company_info.get('revenueGrowth')
operatingMargins = company_info.get('operatingMargins')
returnOnEquity = company_info.get('returnOnEquity')
returnOnAssets = company_info.get('returnOnAssets')
debtToEquity = company_info.get('debtToEquity')
currentRatio = company_info.get('currentRatio')
quickRatio = company_info.get('quickRatio')
totalCashPerShare = company_info.get('totalCashPerShare')
priceToBook = company_info.get('priceToBook')
webSite = company_info.get('website')

company_name = company_info.get('shortName', 'Company')

# Build DataFrame
fundamental_metrics = [
    'Market Cap',
    'Trailing P/E',
    'Forward P/E',
    'Earnings Growth Expectations',
    'EBIT Yield',
    'FCF Yield',
    'Revenue Growth',
    'Operating Margins',
    'Return on Equity',
    'Return on Assets',
    'Debt to Equity',
    'Current Ratio',
    'Quick Ratio',
    'Total Cash Per Share',
    'Price to Book',
]

quick_metrics = pd.DataFrame(index=fundamental_metrics, columns=[company_name])

# Fill values safely
quick_metrics.loc['Market Cap'] = f"${marketCap/1e6:,.0f}M" if marketCap else 'N/A'
quick_metrics.loc['Trailing P/E'] = safe_round(trailingPE)
quick_metrics.loc['Forward P/E'] = safe_round(forwardPE)
quick_metrics.loc['Earnings Growth Expectations'] = (
    safe_round(((trailingPE - forwardPE) / trailingPE) if trailingPE and forwardPE else None, 100, 2, '%')
)
quick_metrics.loc['EBIT Yield'] = safe_round(ebit_yield, 100, 2, '%')
quick_metrics.loc['FCF Yield'] = safe_round(fcf_yield, 100, 2, '%')
quick_metrics.loc['Revenue Growth'] = safe_round(revenueGrowth, 100, 2, '%')
quick_metrics.loc['Operating Margins'] = safe_round(operatingMargins, 100, 2, '%')
quick_metrics.loc['Return on Equity'] = safe_round(returnOnEquity, 1, 2, '%')
quick_metrics.loc['Return on Assets'] = safe_round(returnOnAssets, 1, 2, '%')
quick_metrics.loc['Debt to Equity'] = safe_round(debtToEquity, 1, 2, '%')
quick_metrics.loc['Current Ratio'] = safe_round(currentRatio)
quick_metrics.loc['Quick Ratio'] = safe_round(quickRatio)
quick_metrics.loc['Total Cash Per Share'] = safe_round(totalCashPerShare)
quick_metrics.loc['Price to Book'] = safe_round(priceToBook)

with col12:
    # Display the CVaR and position size
    st.markdown("### Fundamental Analysis Snapshot")
    st.table(quick_metrics)


#############################################################################
######################## ANALYSTS EPS ESTIMATES #############################
#############################################################################



# Get earnings dates and take first 5
earnings_dates = yf.Ticker(ticker).get_earnings_dates()

# Reset index to make datetime a column
earnings_dates = earnings_dates.reset_index()

# Create new column with just the date part
earnings_dates['Earnings Date'] = earnings_dates['Earnings Date'].dt.date

# Filter out rows with missing EPS data
df = earnings_dates.dropna(subset=['EPS Estimate', 'Reported EPS'])

earnings_fig = go.Figure()

# EPS Estimate as hollow circles
earnings_fig.add_trace(go.Scatter(
    x=df['Earnings Date'],
    y=df['EPS Estimate'],
    mode='markers',
    name='EPS Estimate',
    marker=dict(symbol='circle-open', size=20, color='light blue'),
))

# Reported EPS as crosses
earnings_fig.add_trace(go.Scatter(
    x=df['Earnings Date'],
    y=df['Reported EPS'],
    mode='markers',
    name='Reported EPS',
    marker=dict(symbol='x', size=12, color='green'),
))

earnings_fig.update_layout(
    title=f'{company_name} EPS Estimates vs Reported EPS',
    xaxis_title='Earnings Date',
    yaxis_title='EPS',
    template='plotly_dark',
    plot_bgcolor='#1e1e1e',
    paper_bgcolor='#1e1e1e',
    font=dict(color='white')
)

with col8:
    st.plotly_chart(earnings_fig)


#############################################################################
######################## MAJOR HOLDERS ######################################
#############################################################################



major_holders = yf.Ticker(ticker).get_major_holders()

major_holders = major_holders.reset_index()
# Set column names
major_holders.columns = ['Breakdown', 'Value']

# Fix typo and define renaming map
rename_map = {
    'insidersPercentHeld': 'Held by Insiders',
    'institutionsPercentHeld': 'Held by Institutions',
    'institutionsCount': 'Number of Institutions'
}

# Filter and rename
major_holders = major_holders[major_holders['Breakdown'].isin(rename_map.keys())].copy()
major_holders['Breakdown'] = major_holders['Breakdown'].map(rename_map)

# Filter only relevant ownership types
pie_df = major_holders[major_holders['Breakdown'].isin(['Held by Insiders', 'Held by Institutions'])].copy()

# Prepare labels and values
labels = pie_df['Breakdown'].tolist()
values = pie_df['Value'].tolist()

# Add "Other" if total < 1
remaining = 1 - sum(values)
if remaining > 0:
    labels.append("Other")
    values.append(remaining)

fig_pie = go.Figure(data=[go.Pie(
    labels=labels,
    values=values,
    hole=0.4,  # Donut style
    marker=dict(colors=['#009CDF', '#00BFA5', '#FFC107']),
    textinfo='label+percent',
    insidetextorientation='radial'
)])

fig_pie.update_layout(
    title= f'Ownership Breakdown of {company_name}',
    template='plotly_dark',
    height=500,
    margin=dict(t=60, b=40, l=0, r=0),
    paper_bgcolor='#1e1e1e',
    font=dict(color='white')
)

with col10:
    st.plotly_chart(fig_pie)
