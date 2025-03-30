import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.data_processor import group_data_by_period
from utils.visualization import create_trend_chart, create_heatmap

# Page configuration
st.set_page_config(
    page_title="Sales History - InsightCommerce Pro",
    page_icon="📈",
    layout="wide"
)

# Initialize session state if not already done
if 'data' not in st.session_state:
    st.session_state.data = None

# Page header with branded styling
st.markdown('<h1 style="color:#0C3A6D;">Historical Sales Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.1rem; color: #666; margin-bottom: 2rem;">Analyze sales trends, patterns, and growth metrics over time</p>', unsafe_allow_html=True)

# Check if data is loaded
if st.session_state.data is None:
    st.warning("No data loaded. Please go to the Data Import page or load sample data from the Dashboard.")
    st.stop()

# Get filtered data from session state
if 'filtered_data' in st.session_state:
    df = st.session_state.filtered_data
else:
    df = st.session_state.data

# Time period selection
st.subheader("Time Period Analysis")
period = st.radio(
    "Select time period granularity:",
    ["Day", "Week", "Month"],
    horizontal=True
)

# Convert period to lowercase for the function
period_lower = period.lower()

# Group data by selected period
grouped_data = group_data_by_period(df, period=period_lower)

# Create time series chart with metrics
st.subheader(f"Sales Trends by {period}")

# Select metrics to display
metrics_options = ["Revenue", "Orders", "Profit"]
if 'profit' not in df.columns:
    metrics_options.remove("Profit")

selected_metrics = st.multiselect(
    "Select metrics to display:",
    metrics_options,
    default=["Revenue"]
)

# Convert selected metrics to lowercase
selected_metrics_lower = [metric.lower() for metric in selected_metrics]

# Check if there are any metrics selected
if not selected_metrics:
    st.warning("Please select at least one metric to display.")
else:
    # Create trend chart
    fig = create_trend_chart(
        grouped_data, 
        metrics=selected_metrics_lower,
        title=f"{', '.join(selected_metrics)} Trends by {period}"
    )
    st.plotly_chart(fig, use_container_width=True)

# Year-over-Year comparison
st.subheader("Year-over-Year Comparison")

# Check if there's enough data for YoY comparison
df['date'] = pd.to_datetime(df['date'])
min_year = df['date'].dt.year.min()
max_year = df['date'].dt.year.max()

if max_year - min_year < 1:
    st.info("Not enough data for year-over-year comparison. Need at least two years of data.")
else:
    # Select years to compare
    all_years = sorted(df['date'].dt.year.unique())
    selected_years = st.multiselect(
        "Select years to compare:",
        all_years,
        default=list(all_years[-2:])  # Default to last two years
    )
    
    if len(selected_years) < 2:
        st.warning("Please select at least two years to compare.")
    else:
        # Select metric for comparison
        yoy_metric = st.selectbox(
            "Select metric for comparison:",
            metrics_options,
            index=0
        )
        
        # Lower case metric name
        yoy_metric_lower = yoy_metric.lower()
        
        # Prepare data for comparison
        yoy_data = []
        for year in selected_years:
            year_data = df[df['date'].dt.year == year].copy()
            
            # Reset the dates to be in the same year for comparison
            year_data['month_day'] = year_data['date'].dt.strftime('%m-%d')
            
            # Handle differently based on metric
            agg_dict = {'date': 'first'}
            
            if yoy_metric_lower == 'orders':
                # For orders, use transaction_id if available
                if 'transaction_id' in year_data.columns:
                    # First count unique transactions per day
                    temp_df = year_data.groupby('month_day')['transaction_id'].nunique().reset_index()
                    temp_df.rename(columns={'transaction_id': 'orders'}, inplace=True)
                    # Then merge back with the date info
                    date_df = year_data.groupby('month_day')['date'].first().reset_index()
                    year_data = pd.merge(temp_df, date_df, on='month_day')
                else:
                    # Fallback to counting rows as "orders" if no transaction_id
                    year_data = year_data.groupby('month_day').size().reset_index(name='orders')
                    # Get the date information
                    date_df = year_data.groupby('month_day')['date'].first().reset_index()
                    year_data = pd.merge(year_data, date_df, on='month_day')
            else:
                # For revenue, profit, etc. just sum
                agg_dict[yoy_metric_lower] = 'sum'
                year_data = year_data.groupby('month_day').agg(agg_dict).reset_index()
            
            year_data['year'] = year
            year_data['day_of_year'] = year_data['date'].dt.dayofyear
            
            yoy_data.append(year_data)
        
        # Combine all years
        yoy_combined = pd.concat(yoy_data)
        
        # Create YoY comparison chart
        fig = px.line(
            yoy_combined,
            x='day_of_year',
            y=yoy_metric_lower,
            color='year',
            labels={
                'day_of_year': 'Day of Year',
                yoy_metric_lower: yoy_metric,
                'year': 'Year'
            },
            title=f'Year-over-Year {yoy_metric} Comparison',
        )
        
        # Format x-axis to show month names
        month_positions = [datetime(2020, m, 15).timetuple().tm_yday for m in range(1, 13)]
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig.update_xaxes(
            tickvals=month_positions,
            ticktext=month_labels,
            tickangle=0
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Day of week analysis
st.subheader("Day of Week Analysis")

# Calculate average metrics by day of week
df['day_of_week'] = df['date'].dt.dayofweek
df['day_name'] = df['date'].dt.day_name()

# Prepare aggregation dictionary for day of week
dow_agg = {'revenue': 'mean'}

# For orders, check transaction_id first
if 'transaction_id' in df.columns:
    # Group by day and count unique transactions
    dow_trans = df.groupby(['day_of_week', 'day_name'])['transaction_id'].nunique().reset_index()
    dow_trans.rename(columns={'transaction_id': 'orders'}, inplace=True)
    
    # Group other metrics separately
    dow_data = df.groupby(['day_of_week', 'day_name']).agg({
        'revenue': 'mean'
    }).reset_index()
    
    # Merge the counts with other metrics
    dow_data = pd.merge(dow_data, dow_trans, on=['day_of_week', 'day_name'])
else:
    # Without transaction_id, use traditional approach but with safety checks
    if 'orders' in df.columns:
        dow_agg['orders'] = 'mean'
    else:
        # Count rows if no orders column exists
        dow_data = df.groupby(['day_of_week', 'day_name']).agg({
            'revenue': 'mean'
        }).reset_index()
        # Add orders as row count - this creates a proxy for orders
        dow_count = df.groupby(['day_of_week', 'day_name']).size().reset_index(name='orders')
        dow_data = pd.merge(dow_data, dow_count, on=['day_of_week', 'day_name'])
        # Skip the normal aggregation
        dow_agg = None

# Only do this aggregation if we have set an aggregation dictionary
if dow_agg:
    dow_data = df.groupby(['day_of_week', 'day_name']).agg(dow_agg).reset_index()

# Add profit if available
if 'profit' in df.columns:
    dow_profit = df.groupby(['day_of_week', 'day_name'])['profit'].mean().reset_index()
    dow_data = pd.merge(dow_data, dow_profit, on=['day_of_week', 'day_name'])

# Order by day of week
dow_data = dow_data.sort_values('day_of_week')

# Get available metrics in dow_data
dow_available_metrics = list(dow_data.columns)
# Filter out non-metric columns
dow_available_metrics = [m for m in dow_available_metrics if m not in ['day_of_week', 'day_name']]

# Select metric for day of week analysis - only show metrics actually in the data
dow_metric = st.selectbox(
    "Select metric for day of week analysis:",
    dow_available_metrics,
    index=0 if 'revenue' in dow_available_metrics else 0
)

# Lower case metric name
dow_metric_lower = dow_metric.lower()

# Check if selected metric is in the data
if dow_metric_lower not in dow_data.columns:
    st.error(f"The metric '{dow_metric}' is not available in the data. Please select another metric.")
    # Default to using revenue as fallback
    dow_metric = "Revenue"
    dow_metric_lower = "revenue"

# Create day of week bar chart
fig = px.bar(
    dow_data,
    x='day_name',
    y=dow_metric_lower,
    title=f'Average {dow_metric} by Day of Week',
    labels={
        'day_name': 'Day of Week',
        dow_metric_lower: f'Average {dow_metric}'
    },
    color=dow_metric_lower,
    color_continuous_scale='Viridis'
)

st.plotly_chart(fig, use_container_width=True)

# Month of year analysis
st.subheader("Month of Year Analysis")

# Calculate average metrics by month
df['month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.strftime('%b')

# Prepare aggregation dictionary for month
month_agg = {'revenue': 'mean'}

# For orders, check transaction_id first
if 'transaction_id' in df.columns:
    # Group by month and count unique transactions
    month_trans = df.groupby(['month', 'month_name'])['transaction_id'].nunique().reset_index()
    month_trans.rename(columns={'transaction_id': 'orders'}, inplace=True)
    
    # Group other metrics separately
    month_data = df.groupby(['month', 'month_name']).agg({
        'revenue': 'mean'
    }).reset_index()
    
    # Merge the counts with other metrics
    month_data = pd.merge(month_data, month_trans, on=['month', 'month_name'])
else:
    # Without transaction_id, use traditional approach but with safety checks
    if 'orders' in df.columns:
        month_agg['orders'] = 'mean'
    else:
        # Count rows if no orders column exists
        month_data = df.groupby(['month', 'month_name']).agg({
            'revenue': 'mean'
        }).reset_index()
        # Add orders as row count - this creates a proxy for orders
        month_count = df.groupby(['month', 'month_name']).size().reset_index(name='orders')
        month_data = pd.merge(month_data, month_count, on=['month', 'month_name'])
        # Skip the normal aggregation
        month_agg = None

# Only do this aggregation if we have set an aggregation dictionary
if month_agg:
    month_data = df.groupby(['month', 'month_name']).agg(month_agg).reset_index()

# Add profit if available
if 'profit' in df.columns:
    month_profit = df.groupby(['month', 'month_name'])['profit'].mean().reset_index()
    month_data = pd.merge(month_data, month_profit, on=['month', 'month_name'])

# Order by month
month_data = month_data.sort_values('month')

# Get available metrics in month_data
month_available_metrics = list(month_data.columns)
# Filter out non-metric columns
month_available_metrics = [m for m in month_available_metrics if m not in ['month', 'month_name']]

# Select metric for month analysis - only show metrics actually in the data
month_metric = st.selectbox(
    "Select metric for month analysis:",
    month_available_metrics,
    index=0 if 'revenue' in month_available_metrics else 0,
    key="month_metric"
)

# Lower case metric name
month_metric_lower = month_metric.lower()

# Check if selected metric is in the data
if month_metric_lower not in month_data.columns:
    st.error(f"The metric '{month_metric}' is not available in the data. Please select another metric.")
    # Default to using revenue as fallback
    month_metric = "Revenue"
    month_metric_lower = "revenue"

# Create month bar chart
fig = px.bar(
    month_data,
    x='month_name',
    y=month_metric_lower,
    title=f'Average {month_metric} by Month',
    labels={
        'month_name': 'Month',
        month_metric_lower: f'Average {month_metric}'
    },
    color=month_metric_lower,
    color_continuous_scale='Viridis'
)

st.plotly_chart(fig, use_container_width=True)

# Correlation analysis
st.subheader("Correlation Analysis")

# Select only numeric columns for correlation
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

# Create correlation heatmap
if len(numeric_columns) > 1:
    st.write("Correlation between different metrics:")
    
    # Create and display the heatmap
    corr_fig = create_heatmap(df[numeric_columns], title="Metric Correlation Heatmap")
    st.plotly_chart(corr_fig, use_container_width=True)
    
    st.markdown("""
    **Understanding the correlation heatmap:**
    - Values close to 1 indicate strong positive correlation
    - Values close to -1 indicate strong negative correlation
    - Values close to 0 indicate little to no correlation
    
    A high correlation between two metrics suggests they tend to move together.
    """)
else:
    st.info("Not enough numeric columns for correlation analysis.")

# Growth rate analysis
st.subheader("Growth Rate Analysis")

# Group data by period and calculate period-over-period growth
if len(grouped_data) > 1:
    # Calculate growth rates
    grouped_data = grouped_data.sort_values('period')
    grouped_data['previous_revenue'] = grouped_data['revenue'].shift(1)
    grouped_data['revenue_growth'] = (grouped_data['revenue'] - grouped_data['previous_revenue']) / grouped_data['previous_revenue']
    
    if 'profit' in grouped_data.columns:
        grouped_data['previous_profit'] = grouped_data['profit'].shift(1)
        grouped_data['profit_growth'] = (grouped_data['profit'] - grouped_data['previous_profit']) / grouped_data['previous_profit']
    
    # Select metric for growth analysis
    growth_metric = st.selectbox(
        "Select metric for growth analysis:",
        ["Revenue", "Profit"] if 'profit' in grouped_data.columns else ["Revenue"],
        index=0,
        key="growth_metric"
    )
    
    # Lower case metric name
    growth_metric_lower = growth_metric.lower()
    growth_column = f"{growth_metric_lower}_growth"
    
    # Remove first row (no growth rate available) and infinite values
    growth_data = grouped_data.iloc[1:].replace([np.inf, -np.inf], np.nan).dropna(subset=[growth_column])
    
    if len(growth_data) > 0:
        # Create growth rate chart
        fig = px.line(
            growth_data,
            x='period',
            y=growth_column,
            title=f'{period.capitalize()}-over-{period.capitalize()} {growth_metric} Growth Rate',
            labels={
                'period': f'{period.capitalize()}',
                growth_column: f'{growth_metric} Growth Rate'
            }
        )
        
        # Add zero line
        fig.add_shape(
            type="line",
            x0=growth_data['period'].iloc[0],
            y0=0,
            x1=growth_data['period'].iloc[-1],
            y1=0,
            line=dict(
                color="red",
                width=2,
                dash="dash",
            )
        )
        
        # Format y-axis as percentage
        fig.update_layout(yaxis_tickformat='.1%')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show average growth rate
        avg_growth = growth_data[growth_column].mean()
        st.metric(
            f"Average {growth_metric} Growth Rate",
            f"{avg_growth:.2%}",
            delta=f"{avg_growth:.2%}",
            delta_color="normal"
        )
    else:
        st.info(f"Not enough data to calculate {growth_metric.lower()} growth rates.")
else:
    st.info("Not enough data points to calculate growth rates.")

# Download historical data
st.subheader("Export Data")

# Create a download button for the grouped data
grouped_csv = grouped_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Historical Analysis Data",
    data=grouped_csv,
    file_name=f"historical_analysis_{period_lower}.csv",
    mime="text/csv"
)
