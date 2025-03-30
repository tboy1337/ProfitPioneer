import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from datetime import timedelta

def prepare_time_features(df, target_col='revenue'):
    """
    Prepare time-based features for forecasting
    
    Args:
        df: pandas DataFrame with time series data
        target_col: Target column to forecast
        
    Returns:
        X: Feature matrix
        y: Target values
        dates: Date values
    """
    # Check if target_col is 'orders' and needs to be created from transaction_id
    if target_col == 'orders' and 'orders' not in df.columns and 'transaction_id' in df.columns:
        # Group by date and count unique transactions
        grouped = df.groupby('date')['transaction_id'].nunique().reset_index()
        grouped.rename(columns={'transaction_id': 'orders'}, inplace=True)
        
        # Merge with original data for other features
        temp_df = df.drop_duplicates('date')
        temp_df = temp_df[['date']]  # Keep only date column
        temp_df = pd.merge(temp_df, grouped, on='date', how='left')
        
        # For any remaining features in the original df, group and merge
        for col in df.columns:
            if col not in ['date', 'transaction_id', 'orders'] and col not in temp_df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    col_agg = df.groupby('date')[col].mean().reset_index()
                    temp_df = pd.merge(temp_df, col_agg, on='date', how='left')
        
        # Use the new aggregated dataframe
        df = temp_df
    
    if 'date' not in df.columns or target_col not in df.columns:
        raise ValueError(f"DataFrame must have 'date' and '{target_col}' columns")
    
    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Extract date features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Add time index
    df['time_index'] = np.arange(len(df))
    
    # Create trend and seasonality features
    features = [
        'time_index', 'day_of_week', 'day_of_month', 
        'month', 'quarter', 'is_weekend'
    ]
    
    # Create lag features if enough data
    if len(df) > 7:
        df['lag_1'] = df[target_col].shift(1)
        df['lag_7'] = df[target_col].shift(7)
        features.extend(['lag_1', 'lag_7'])
    
    # Drop NaN values
    df = df.dropna()
    
    # Prepare X and y
    X = df[features]
    y = df[target_col]
    dates = df['date']
    
    return X, y, dates

def train_linear_forecast_model(X, y):
    """
    Train a linear regression forecast model
    
    Args:
        X: Feature matrix
        y: Target values
        
    Returns:
        Trained model
    """
    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)
    
    return model

def train_polynomial_forecast_model(X, y, degree=2):
    """
    Train a polynomial regression forecast model
    
    Args:
        X: Feature matrix
        y: Target values
        degree: Polynomial degree
        
    Returns:
        Trained model and polynomial features transformer
    """
    # First, handle potential extreme values in the input data
    # We'll scale the data to prevent overflow issues
    from sklearn.preprocessing import StandardScaler
    
    # Create scaler to normalize the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use a safer degree=1 if there's not enough data
    if X.shape[0] < 10:  # Small datasets can lead to overfitting with high polynomial degrees
        actual_degree = 1
    else:
        actual_degree = min(degree, 2)  # Limit to degree 2 to prevent overflow
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=actual_degree, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)
    
    # Create and train the model with regularization to prevent overfitting
    # Use Ridge regression instead of LinearRegression for better stability
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=0.5)  # Add regularization to prevent extreme coefficient values
    model.fit(X_poly, y)
    
    # Package the scaler with the model for later use
    return (model, poly, scaler)

def train_random_forest_forecast_model(X, y):
    """
    Train a random forest regression forecast model
    
    Args:
        X: Feature matrix
        y: Target values
        
    Returns:
        Trained model
    """
    # Create and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def generate_forecast(df, periods=30, model_type='linear', target_col='revenue'):
    """
    Generate a forecast for future periods
    
    Args:
        df: pandas DataFrame with time series data
        periods: Number of periods to forecast
        model_type: Type of forecasting model ('linear', 'polynomial', 'random_forest')
        target_col: Target column to forecast
        
    Returns:
        DataFrame with original data and forecast,
        Model performance metrics
    """
    # Prepare data
    X, y, dates = prepare_time_features(df, target_col)
    
    # Train model
    if model_type == 'linear':
        model = train_linear_forecast_model(X, y)
        y_pred = model.predict(X)
        poly = None
        scaler = None
    elif model_type == 'polynomial':
        try:
            model, poly, scaler = train_polynomial_forecast_model(X, y)
            X_scaled = scaler.transform(X)
            X_poly = poly.transform(X_scaled)
            y_pred = model.predict(X_poly)
        except Exception as e:
            # Fallback to linear model if polynomial fails
            print(f"Polynomial model failed: {str(e)}. Falling back to linear model.")
            model_type = 'linear'
            model = train_linear_forecast_model(X, y)
            y_pred = model.predict(X)
            poly = None
            scaler = None
    elif model_type == 'random_forest':
        model = train_random_forest_forecast_model(X, y)
        y_pred = model.predict(X)
        poly = None
        scaler = None
    else:
        raise ValueError("model_type must be 'linear', 'polynomial', or 'random_forest'")
    
    # Calculate performance metrics
    metrics = {
        'mae': mean_absolute_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'r2': r2_score(y, y_pred)
    }
    
    # Prepare forecast dates
    last_date = dates.iloc[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
    
    # Prepare forecast features
    forecast_df = pd.DataFrame({'date': forecast_dates})
    forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek
    forecast_df['day_of_month'] = forecast_df['date'].dt.day
    forecast_df['month'] = forecast_df['date'].dt.month
    forecast_df['quarter'] = forecast_df['date'].dt.quarter
    forecast_df['year'] = forecast_df['date'].dt.year
    forecast_df['is_weekend'] = forecast_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Add time index continuing from training data
    last_index = X['time_index'].iloc[-1]
    forecast_df['time_index'] = np.arange(last_index + 1, last_index + 1 + periods)
    
    # Add lag features if they were used in training
    if 'lag_1' in X.columns:
        # For the first forecast point, use the last actual value
        last_actual = y.iloc[-1]
        forecast_values = []
        
        for i in range(periods):
            if i == 0:
                lag_1 = last_actual
                lag_7 = y.iloc[-7] if len(y) > 7 else last_actual
            else:
                lag_1 = forecast_values[-1]
                lag_7 = y.iloc[-7+i] if i < 7 and len(y) > 7-i else forecast_values[-7] if i >= 7 else last_actual
            
            forecast_df.loc[i, 'lag_1'] = lag_1
            forecast_df.loc[i, 'lag_7'] = lag_7
            
            # Get feature columns in the same order as training
            X_forecast = forecast_df.loc[i:i, X.columns]
            
            # Make prediction
            if model_type == 'polynomial':
                try:
                    # Apply the same scaling as during training
                    X_scaled = scaler.transform(X_forecast)
                    # Handle potential issues with polynomial features
                    X_poly = poly.transform(X_scaled)
                    
                    # Check for infinity or NaN values
                    if np.isnan(X_poly).any() or np.isinf(X_poly).any():
                        raise ValueError("Polynomial features contain NaN or infinity values")
                    
                    prediction = model.predict(X_poly)[0]
                    
                    # Sanity check on prediction value
                    if np.isnan(prediction) or np.isinf(prediction) or prediction > y.mean() * 10:
                        raise ValueError("Polynomial prediction produced an extreme value")
                except Exception as e:
                    # Fallback to a simple average of recent values if prediction fails
                    recent_values = y.tail(7).mean()
                    prediction = recent_values
                    print(f"Polynomial prediction failed: {str(e)}. Using fallback value of {recent_values}.")
            else:
                prediction = model.predict(X_forecast)[0]
            
            # Ensure prediction is positive and not extreme
            prediction = max(0, prediction)  # Enforce non-negative values
            # Limit to reasonable range (e.g., max 5x the mean value in training)
            max_reasonable = y.mean() * 5
            prediction = min(prediction, max_reasonable)
            
            forecast_values.append(prediction)
    else:
        # Get feature columns in the same order as training
        X_forecast = forecast_df[X.columns]
        
        # Generate forecast
        if model_type == 'polynomial':
            try:
                # Apply the same scaling as during training
                X_scaled = scaler.transform(X_forecast)
                X_poly = poly.transform(X_scaled)
                forecast_values = model.predict(X_poly)
                
                # Apply reasonable limits to prevent extreme values
                forecast_values = np.maximum(0, forecast_values)  # No negative values
                max_reasonable = y.mean() * 5
                forecast_values = np.minimum(forecast_values, max_reasonable)
            except Exception as e:
                # Fallback to a more stable model if polynomial fails
                print(f"Polynomial prediction failed: {str(e)}. Using fallback linear model.")
                from sklearn.linear_model import LinearRegression
                fallback_model = LinearRegression()
                fallback_model.fit(X, y)
                forecast_values = fallback_model.predict(X_forecast)
        else:
            forecast_values = model.predict(X_forecast)
            
            # Apply reasonable limits to prevent extreme values
            forecast_values = np.maximum(0, forecast_values)  # No negative values
            max_reasonable = y.mean() * 5
            forecast_values = np.minimum(forecast_values, max_reasonable)
    
    # Add forecast to the DataFrame
    forecast_df[target_col] = forecast_values
    
    # Combine original data and forecast
    original_df = df[['date', target_col]].copy()
    original_df['type'] = 'Actual'
    
    forecast_df = forecast_df[['date', target_col]]
    forecast_df['type'] = 'Forecast'
    
    combined_df = pd.concat([original_df, forecast_df], ignore_index=True)
    combined_df = combined_df.sort_values('date')
    
    return combined_df, metrics

def create_forecast_chart(combined_df, target_col='revenue', title='Revenue Forecast'):
    """
    Create a forecast chart with actual and predicted values
    
    Args:
        combined_df: DataFrame with actual and forecast values
        target_col: Target column being forecasted
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Ensure the target column exists in the DataFrame
    if target_col not in combined_df.columns:
        # If using a new metric that doesn't match the previously forecasted metric
        # Replace with the existing forecast column (which should be either 'revenue', 'orders', or 'profit')
        existing_cols = [col for col in ['revenue', 'orders', 'profit'] if col in combined_df.columns]
        if existing_cols:
            target_col = existing_cols[0]
        else:
            # Fallback to a safe default if no valid columns found
            raise ValueError(f"Column {target_col} not found in forecast data")
    
    # Split the data
    actual_data = combined_df[combined_df['type'] == 'Actual']
    forecast_data = combined_df[combined_df['type'] == 'Forecast']
    
    # Create the figure
    fig = go.Figure()
    
    # Add actual data
    fig.add_trace(go.Scatter(
        x=actual_data['date'],
        y=actual_data[target_col],
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue', width=2),
        marker=dict(size=5),
    ))
    
    # Add forecast data
    fig.add_trace(go.Scatter(
        x=forecast_data['date'],
        y=forecast_data[target_col],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=5),
    ))
    
    # Add shaded area for forecast uncertainty
    forecast_std = forecast_data[target_col].std()
    fig.add_trace(go.Scatter(
        x=forecast_data['date'],
        y=forecast_data[target_col] + forecast_std * 1.96,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='none',
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_data['date'],
        y=forecast_data[target_col] - forecast_std * 1.96,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.2)',
        name='95% Confidence Interval',
        hoverinfo='none',
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=target_col.title(),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return fig
