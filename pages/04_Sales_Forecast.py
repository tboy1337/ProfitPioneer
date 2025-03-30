import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.data_processor import group_data_by_period
from utils.forecasting import generate_forecast, create_forecast_chart

# Page configuration
st.set_page_config(
    page_title="Sales Forecast - E-Commerce Analytics",
    page_icon="🔮",
    layout="wide"
)

# Initialize session state if not already done
if 'data' not in st.session_state:
    st.session_state.data = None

# Page header
st.title("Sales Forecast Analysis")
st.markdown("Predict future sales and plan your business strategy")

# Check if data is loaded
if st.session_state.data is None:
    st.warning("No data loaded. Please go to the Data Import page or load sample data from the Dashboard.")
    st.stop()

# Get filtered data from session state
if 'filtered_data' in st.session_state:
    df = st.session_state.filtered_data
else:
    df = st.session_state.data

# Check if there's enough data for forecasting
df['date'] = pd.to_datetime(df['date'])
date_range = (df['date'].max() - df['date'].min()).days

if date_range < 14:
    st.error("Not enough data for forecasting. We need at least 14 days of historical data.")
    st.stop()

# Forecast configuration
st.header("Forecast Configuration")

col1, col2 = st.columns(2)

with col1:
    forecast_periods = st.slider(
        "Forecast Horizon (Days)",
        min_value=7,
        max_value=90,
        value=30,
        step=1,
        help="Number of days to forecast into the future"
    )

with col2:
    # Determine available metrics for forecasting
    available_metrics = ["Revenue"]  # Revenue is always available
    
    # Check for orders in transaction_id or orders column
    if "transaction_id" in df.columns:
        # We'll need to count transactions if selected
        available_metrics.append("Orders")
    elif "orders" in df.columns:
        # Orders column already exists
        available_metrics.append("Orders")
        
    # Check if profit is available
    if "profit" in df.columns:
        available_metrics.append("Profit")
    
    # Show available metrics
    forecast_metric = st.selectbox(
        "Metric to Forecast",
        options=available_metrics,
        index=0,
        help="Choose which metric to forecast"
    )
    
    # Convert to lowercase for column name
    forecast_metric_lower = forecast_metric.lower()

# Model selection
forecast_model = st.selectbox(
    "Forecasting Model",
    options=["Linear", "Polynomial", "Random Forest"],
    index=0,
    help="Choose the forecasting model type"
)

# Map model selection to backend model type
model_type_map = {
    "Linear": "linear",
    "Polynomial": "polynomial",
    "Random Forest": "random_forest"
}
model_type = model_type_map[forecast_model]

# Generate forecast on button click
generate_button = st.button("Generate Forecast", type="primary")

if generate_button:
    with st.spinner(f"Generating {forecast_metric} forecast for the next {forecast_periods} days..."):
        try:
            # Ensure data is sorted by date
            df = df.sort_values('date')
            
            # Special handling for orders if selected metric is orders but column doesn't exist
            if forecast_metric_lower == 'orders' and 'orders' not in df.columns and 'transaction_id' in df.columns:
                # Count unique transactions per day
                daily_trans = df.groupby('date')['transaction_id'].nunique().reset_index()
                daily_trans.rename(columns={'transaction_id': 'orders'}, inplace=True)
                daily_data = daily_trans
            
            # Default handling for existing columns
            elif forecast_metric_lower in df.columns:
                # Group data by day if not already daily
                if len(df) > len(df['date'].unique()):
                    daily_data = df.groupby('date').agg({
                        forecast_metric_lower: 'sum'
                    }).reset_index()
                else:
                    daily_data = df[['date', forecast_metric_lower]]
            else:
                # If we get here, raise an error with clear message
                raise ValueError(f"Cannot generate forecast: '{forecast_metric}' data is not available in the dataset.")
            
            # Generate forecast
            forecast_df, metrics = generate_forecast(
                daily_data,
                periods=forecast_periods,
                model_type=model_type,
                target_col=forecast_metric_lower
            )
            
            # Store forecast in session state
            st.session_state.forecast_df = forecast_df
            st.session_state.forecast_metrics = metrics
            st.session_state.forecast_generated = True
        
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            st.session_state.forecast_generated = False

# Display forecast if available
if 'forecast_generated' in st.session_state and st.session_state.forecast_generated:
    forecast_df = st.session_state.forecast_df
    metrics = st.session_state.forecast_metrics
    
    # Forecast results
    st.header("Forecast Results")
    
    try:
        # Create and display the forecast chart
        # Check if the target column matches what's in the forecast_df
        available_metrics = [col for col in ['revenue', 'orders', 'profit'] if col in forecast_df.columns]
        
        # If the selected metric isn't in the forecast data, use the one that is
        # This prevents errors when switching metrics without regenerating the forecast
        display_metric = forecast_metric_lower
        if forecast_metric_lower not in available_metrics and len(available_metrics) > 0:
            display_metric = available_metrics[0]
            st.info(f"Showing previous forecast for {display_metric.title()}. Click 'Generate Forecast' to update with {forecast_metric}.")
        
        fig = create_forecast_chart(
            forecast_df,
            target_col=display_metric,
            title=f'{display_metric.title()} Forecast for Next {forecast_periods} Days'
        )
    except Exception as e:
        st.error(f"Error displaying forecast: {str(e)}. Please generate a new forecast.")
        st.session_state.forecast_generated = False
        st.stop()
    st.plotly_chart(fig, use_container_width=True)
    
    # Model performance metrics
    st.subheader("Forecast Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Absolute Error (MAE)", f"£{metrics['mae']:.2f}" if forecast_metric_lower == 'revenue' else f"{metrics['mae']:.2f}")
    
    with col2:
        st.metric("Root Mean Square Error (RMSE)", f"£{metrics['rmse']:.2f}" if forecast_metric_lower == 'revenue' else f"{metrics['rmse']:.2f}")
    
    with col3:
        st.metric("R² Score", f"{metrics['r2']:.4f}")
    
    # Forecast summary
    st.subheader("Forecast Summary")
    
    # Extract actual and forecast data
    actual_data = forecast_df[forecast_df['type'] == 'Actual']
    forecast_data = forecast_df[forecast_df['type'] == 'Forecast']
    
    try:
        # Use the same display metric as the chart for consistency
        available_metrics = [col for col in ['revenue', 'orders', 'profit'] if col in forecast_df.columns]
        display_metric = forecast_metric_lower
        
        if forecast_metric_lower not in available_metrics and len(available_metrics) > 0:
            display_metric = available_metrics[0]
            # No need for additional info message as it's already shown above
        
        # Calculate key statistics
        actual_mean = actual_data[display_metric].mean()
        forecast_mean = forecast_data[display_metric].mean()
        forecast_total = forecast_data[display_metric].sum()
        
        # Calculate growth rate
        growth_rate = (forecast_mean - actual_mean) / actual_mean if actual_mean > 0 else 0
        
        # Format differently based on whether it's revenue or not
        is_revenue = display_metric == 'revenue'
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                f"Average Daily {display_metric.title()} (Historical)", 
                f"£{actual_mean:.2f}" if is_revenue else f"{actual_mean:.2f}"
            )
        
        with col2:
            st.metric(
                f"Average Daily {display_metric.title()} (Forecast)", 
                f"£{forecast_mean:.2f}" if is_revenue else f"{forecast_mean:.2f}",
                delta=f"{growth_rate:.2%}"
            )
        
        with col3:
            st.metric(
                f"Total {display_metric.title()} (Next {forecast_periods} Days)", 
                f"£{forecast_total:.2f}" if is_revenue else f"{forecast_total:.2f}"
            )
    except Exception as e:
        st.warning(f"Could not calculate forecast summary: {str(e)}. Please generate a new forecast.")
        st.session_state.forecast_generated = False
    
    # Forecast data table
    st.subheader("Detailed Forecast Data")
    
    try:
        # Use the same display metric as above for consistency
        available_metrics = [col for col in ['revenue', 'orders', 'profit'] if col in forecast_df.columns]
        display_metric = forecast_metric_lower
        
        if forecast_metric_lower not in available_metrics and len(available_metrics) > 0:
            display_metric = available_metrics[0]
        
        # Format the forecast data for display
        display_forecast = forecast_data.copy()
        display_forecast['date'] = display_forecast['date'].dt.strftime('%Y-%m-%d')
        
        # Prepare the column names for display
        display_metric_title = display_metric.title()
        
        # Rename columns
        column_rename = {
            'date': 'Date',
            'type': 'Type'
        }
        column_rename[display_metric] = display_metric_title
        display_forecast = display_forecast.rename(columns=column_rename)
        
        # Sort by date
        display_forecast = display_forecast.sort_values('Date')
        
        # Display the table
        st.dataframe(
            display_forecast[['Date', display_metric_title]],
            use_container_width=True,
            column_config={
                'Date': 'Date',
                display_metric_title: st.column_config.NumberColumn(
                    display_metric_title,
                    format="£%.2f" if display_metric == 'revenue' else "%.2f"
                )
            }
        )
        
        # Download forecast data
        st.download_button(
            label="Download Forecast Data",
            data=display_forecast.to_csv(index=False).encode('utf-8'),
            file_name=f"{display_metric}_forecast.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.warning(f"Could not display forecast details: {str(e)}. Please generate a new forecast.")
        st.session_state.forecast_generated = False
    
    # Interpretation and recommendations
    st.header("Forecast Interpretation")
    
    try:
        # Use the same display metric as above for consistency
        available_metrics = [col for col in ['revenue', 'orders', 'profit'] if col in forecast_df.columns]
        display_metric = forecast_metric_lower
        
        if forecast_metric_lower not in available_metrics and len(available_metrics) > 0:
            display_metric = available_metrics[0]
        
        # Calculate short-term trend (next 7 days)
        short_term = forecast_data.iloc[:min(7, len(forecast_data))]
        short_term_trend = np.polyfit(range(len(short_term)), short_term[display_metric], 1)[0]
        
        # Calculate medium-term trend (entire forecast period)
        medium_term_trend = np.polyfit(range(len(forecast_data)), forecast_data[display_metric], 1)[0]
        
        # Display trend interpretation
        st.subheader("Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Short-Term Trend (7 Days)", 
                "Increasing ↗️" if short_term_trend > 0 else "Decreasing ↘️" if short_term_trend < 0 else "Stable →",
                delta=f"{short_term_trend:.2f} per day"
            )
        
        with col2:
            st.metric(
                "Medium-Term Trend", 
                "Increasing ↗️" if medium_term_trend > 0 else "Decreasing ↘️" if medium_term_trend < 0 else "Stable →",
                delta=f"{medium_term_trend:.2f} per day"
            )
        
        # Calculate effective growth rate for recommendations
        actual_mean = actual_data[display_metric].mean()
        forecast_mean = forecast_data[display_metric].mean()
        growth_rate = (forecast_mean - actual_mean) / actual_mean if actual_mean > 0 else 0
        
        # Recommendations based on forecast
        st.subheader("Recommendations")
        
        if growth_rate > 0.1:
            st.success("""
            ### Strong Growth Expected
            
            **Recommended Actions:**
            - Ensure inventory levels can support increased demand
            - Consider scaling up operations and marketing efforts
            - Plan for additional resources to maintain customer satisfaction
            - Leverage the momentum to introduce new products
            """)
        elif growth_rate > 0:
            st.info("""
            ### Moderate Growth Expected
            
            **Recommended Actions:**
            - Maintain current inventory levels with slight increases
            - Focus on customer retention strategies
            - Identify top-performing products and optimize their marketing
            - Look for opportunities to increase average order value
            """)
        elif growth_rate > -0.1:
            st.warning("""
            ### Stable Performance Expected
            
            **Recommended Actions:**
            - Focus on efficiency and cost optimization
            - Invest in customer loyalty and retention programs
            - Look for opportunities to diversify revenue streams
            - Consider promotional activities to stimulate growth
            """)
        else:
            st.error("""
            ### Decline Expected
            
            **Recommended Actions:**
            - Identify root causes of the projected decline
            - Develop targeted promotions to boost sales
            - Consider adjusting inventory to prevent overstocking
            - Focus on high-margin products and loyal customers
            - Explore new marketing channels or product offerings
            """)
    except Exception as e:
        st.warning(f"Could not generate trend analysis: {str(e)}. Please generate a new forecast.")
    
    # Seasonality detection
    if date_range > 60:  # Only if we have enough historical data
        st.subheader("Seasonality Analysis")
        
        try:
            # Special handling for orders if orders metric doesn't exist
            if forecast_metric_lower == 'orders' and 'orders' not in df.columns and 'transaction_id' in df.columns:
                # Count unique transaction IDs for each day of the week
                daily_data = df.groupby(df['date'].dt.dayofweek)['transaction_id'].nunique().reset_index()
                daily_data.rename(columns={'transaction_id': 'orders'}, inplace=True)
            
            # Handle forecasting profit if it exists
            elif forecast_metric_lower == 'profit' and 'profit' in df.columns:
                daily_data = df.groupby(df['date'].dt.dayofweek)['profit'].mean().reset_index()
            
            # Default handling for revenue which should always be present
            elif forecast_metric_lower == 'revenue':
                daily_data = df.groupby(df['date'].dt.dayofweek)['revenue'].mean().reset_index()
                
            # If we can't handle the metric in a standard way, try the generic approach with dynamic column
            elif forecast_metric_lower in df.columns:
                daily_data = df.groupby(df['date'].dt.dayofweek).agg({
                    forecast_metric_lower: 'mean'
                }).reset_index()
            else:
                # If we really can't get the data, show a warning and create an empty placeholder
                st.warning(f"Cannot perform seasonality analysis for {forecast_metric}. Data not available.")
                # Skip the rest of the seasonality section
                raise ValueError(f"Column {forecast_metric_lower} not available for seasonality analysis")
        except Exception as e:
            st.warning(f"Seasonality analysis error: {str(e)}")
            daily_data = None
            
        # Only proceed if we have valid data
        if daily_data is not None:
            daily_data['day'] = daily_data['date'].map({
                0: 'Monday',
                1: 'Tuesday',
                2: 'Wednesday',
                3: 'Thursday',
                4: 'Friday',
                5: 'Saturday',
                6: 'Sunday'
            })
            
            # Create day of week pattern chart
            fig = px.bar(
                daily_data,
                x='day',
                y=forecast_metric_lower,
                title=f'Day of Week Pattern for {forecast_metric}',
                labels={
                    'day': 'Day of Week',
                    forecast_metric_lower: f'Average {forecast_metric}'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Find best and worst days
            best_day = daily_data.loc[daily_data[forecast_metric_lower].idxmax()]
            worst_day = daily_data.loc[daily_data[forecast_metric_lower].idxmin()]
            
            st.markdown(f"""
            **Day of Week Insights:**
            - Best performing day: **{best_day['day']}** (Average {forecast_metric}: {'£' if forecast_metric_lower == 'revenue' else ''}{best_day[forecast_metric_lower]:.2f})
            - Worst performing day: **{worst_day['day']}** (Average {forecast_metric}: {'£' if forecast_metric_lower == 'revenue' else ''}{worst_day[forecast_metric_lower]:.2f})
            
            **Recommendations:**
            - Consider scheduling promotions or marketing efforts for lower-performing days
            - Ensure adequate staffing and inventory for high-performing days
            - Analyze customer behavior patterns on different days of the week
            """)
else:
    st.info("""
    Configure your forecast settings above and click "Generate Forecast" to see predictions for future sales performance.
    
    **Tips for accurate forecasting:**
    - Use at least 30 days of historical data for better accuracy
    - Choose the model type that best fits your data pattern
    - Start with shorter forecast horizons for better reliability
    - Consider using Revenue as the primary forecast metric
    """)

# What is forecasting
with st.expander("About Sales Forecasting"):
    st.markdown("""
    ## Understanding Sales Forecasting
    
    Sales forecasting uses historical data and statistical models to predict future sales performance. This helps businesses make informed decisions about inventory, staffing, marketing, and financial planning.
    
    ### Forecasting Methods Used in This Dashboard
    
    1. **Linear Forecasting**: Uses a straight-line approach to predict future values based on past performance. Works well for steady, consistent growth or decline patterns.
    
    2. **Polynomial Forecasting**: Can capture non-linear trends and is more flexible than linear forecasting. Better for data with curved patterns or seasonal fluctuations.
    
    3. **Random Forest Forecasting**: A machine learning approach that can capture complex patterns and relationships in your data. Often provides more accurate forecasts for complicated business data.
    
    ### How to Use Forecast Results
    
    - **Inventory Planning**: Adjust stock levels based on predicted demand
    - **Staffing**: Plan workforce needs according to expected business volume
    - **Cash Flow Management**: Prepare for expected revenue changes
    - **Marketing Strategy**: Time campaigns to coincide with forecast peaks and valleys
    - **Business Goal Setting**: Establish realistic targets based on predicted performance
    
    ### Forecast Accuracy
    
    The model performance metrics help you understand how reliable the forecast is:
    
    - **MAE (Mean Absolute Error)**: The average magnitude of errors in the forecast
    - **RMSE (Root Mean Square Error)**: Similar to MAE but gives more weight to large errors
    - **R² Score**: How well the model fits the data (1.0 is perfect, 0.0 is no relationship)
    
    Remember that all forecasts have uncertainty. Longer forecast horizons generally have less accuracy than shorter ones.
    """)
