import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data(file_path):
    """
    Load data from a CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas DataFrame with the loaded data
    """
    try:
        # Attempt to read the file
        df = pd.read_csv(file_path)
        
        # Ensure the date column is in the correct format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Basic validation of required columns
        required_columns = ['date', 'revenue']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
        return df
        
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def filter_data_by_date(df, start_date, end_date):
    """
    Filter data by date range
    
    Args:
        df: pandas DataFrame with data
        start_date: Start date for filtering
        end_date: End date for filtering
        
    Returns:
        Filtered pandas DataFrame
    """
    # Convert string dates to datetime if they're not already
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()
        
    # Convert df['date'] to datetime for comparison
    df['date_temp'] = pd.to_datetime(df['date'])
    
    # Filter by date range
    filtered_df = df[(df['date_temp'].dt.date >= start_date) & 
                     (df['date_temp'].dt.date <= end_date)]
    
    # Remove temporary column
    filtered_df = filtered_df.drop('date_temp', axis=1)
    
    return filtered_df

def calculate_kpis(df):
    """
    Calculate key performance indicators from the data
    
    Args:
        df: pandas DataFrame with e-commerce data
        
    Returns:
        Dictionary of KPIs
    """
    # Initialize results dictionary
    kpis = {}
    
    # Basic KPIs
    kpis['total_revenue'] = df['revenue'].sum()
    
    # Calculate orders based on transaction_id if available
    if 'transaction_id' in df.columns:
        kpis['total_orders'] = df['transaction_id'].nunique()
    elif 'orders' in df.columns:
        kpis['total_orders'] = df['orders'].sum()
    else:
        kpis['total_orders'] = len(df)
    
    kpis['avg_order_value'] = kpis['total_revenue'] / kpis['total_orders'] if kpis['total_orders'] > 0 else 0
    
    # Profit metrics
    if 'profit' in df.columns:
        kpis['total_profit'] = df['profit'].sum()
        kpis['profit_margin'] = kpis['total_profit'] / kpis['total_revenue'] if kpis['total_revenue'] > 0 else 0
    else:
        kpis['total_profit'] = 0
        kpis['profit_margin'] = 0
    
    # Customer metrics
    if 'customer_id' in df.columns:
        kpis['unique_customers'] = df['customer_id'].nunique()
    elif 'customers' in df.columns:
        kpis['unique_customers'] = df['customers'].sum()
    else:
        kpis['unique_customers'] = 0
    
    # Conversion rate
    if 'conversion_rate' in df.columns:
        kpis['avg_conversion_rate'] = df['conversion_rate'].mean()
    else:
        kpis['avg_conversion_rate'] = 0
    
    # Customer acquisition cost
    if 'customer_acquisition_cost' in df.columns:
        kpis['avg_cac'] = df['customer_acquisition_cost'].mean()
    else:
        kpis['avg_cac'] = 0
    
    # Calculate growth metrics by comparing the most recent half of the data with the older half
    df['date'] = pd.to_datetime(df['date'])
    sorted_df = df.sort_values('date')
    midpoint = len(sorted_df) // 2
    
    older_half = sorted_df.iloc[:midpoint]
    newer_half = sorted_df.iloc[midpoint:]
    
    # Calculate growth rates
    kpis['revenue_growth'] = calculate_growth_rate(newer_half['revenue'].sum(), older_half['revenue'].sum())
    
    # Calculate order growth using transaction_id if available
    if 'transaction_id' in df.columns:
        new_orders = newer_half['transaction_id'].nunique()
        old_orders = older_half['transaction_id'].nunique()
        kpis['order_growth'] = calculate_growth_rate(new_orders, old_orders)
    elif 'orders' in newer_half.columns:
        kpis['order_growth'] = calculate_growth_rate(
            newer_half['orders'].sum(),
            older_half['orders'].sum()
        )
    else:
        # Use record count as a fallback
        kpis['order_growth'] = calculate_growth_rate(len(newer_half), len(older_half))
    
    # Calculate other growth rates
    kpis['profit_growth'] = calculate_growth_rate(
        newer_half['profit'].sum() if 'profit' in newer_half.columns else 0,
        older_half['profit'].sum() if 'profit' in older_half.columns else 0
    )
    
    kpis['profit_margin_growth'] = calculate_growth_rate(
        (newer_half['profit'].sum() / newer_half['revenue'].sum()) if 'profit' in newer_half.columns and newer_half['revenue'].sum() > 0 else 0,
        (older_half['profit'].sum() / older_half['revenue'].sum()) if 'profit' in older_half.columns and older_half['revenue'].sum() > 0 else 0
    )
    
    # AOV growth calculation
    # Calculate new AOV based on available data
    if 'transaction_id' in newer_half.columns:
        # Use unique transaction count for new AOV
        new_order_count = newer_half['transaction_id'].nunique()
        new_aov = newer_half['revenue'].sum() / new_order_count if new_order_count > 0 else 0
    elif 'orders' in newer_half.columns and newer_half['orders'].sum() > 0:
        # Use orders column if available
        new_aov = newer_half['revenue'].sum() / newer_half['orders'].sum()
    else:
        # Fallback to record count
        new_aov = newer_half['revenue'].sum() / len(newer_half) if len(newer_half) > 0 else 0
        
    # Calculate old AOV based on available data
    if 'transaction_id' in older_half.columns:
        # Use unique transaction count for old AOV
        old_order_count = older_half['transaction_id'].nunique()
        old_aov = older_half['revenue'].sum() / old_order_count if old_order_count > 0 else 0
    elif 'orders' in older_half.columns and older_half['orders'].sum() > 0:
        # Use orders column if available
        old_aov = older_half['revenue'].sum() / older_half['orders'].sum()
    else:
        # Fallback to record count
        old_aov = older_half['revenue'].sum() / len(older_half) if len(older_half) > 0 else 0
    
    # Calculate AOV growth rate
    kpis['aov_growth'] = calculate_growth_rate(new_aov, old_aov)
    
    # Conversion rate growth
    if 'conversion_rate' in df.columns:
        kpis['conversion_rate_growth'] = calculate_growth_rate(newer_half['conversion_rate'].mean(), older_half['conversion_rate'].mean())
    else:
        kpis['conversion_rate_growth'] = 0
    
    # CAC growth
    if 'customer_acquisition_cost' in df.columns:
        kpis['cac_growth'] = calculate_growth_rate(newer_half['customer_acquisition_cost'].mean(), older_half['customer_acquisition_cost'].mean())
    else:
        kpis['cac_growth'] = 0
    
    # Customer growth
    if 'customer_id' in df.columns:
        kpis['customer_growth'] = calculate_growth_rate(newer_half['customer_id'].nunique(), older_half['customer_id'].nunique())
    elif 'customers' in df.columns:
        kpis['customer_growth'] = calculate_growth_rate(newer_half['customers'].sum(), older_half['customers'].sum())
    else:
        kpis['customer_growth'] = 0
    
    return kpis

def calculate_growth_rate(new_value, old_value):
    """
    Calculate percentage growth rate between two values
    
    Args:
        new_value: The newer (current) value
        old_value: The older (previous) value
        
    Returns:
        Growth rate as a decimal (0.05 = 5% growth)
    """
    if old_value == 0:
        return 0  # Avoid division by zero
    
    return (new_value - old_value) / old_value

def group_data_by_period(df, period='day'):
    """
    Group data by time period
    
    Args:
        df: pandas DataFrame with data
        period: Time period to group by ('day', 'week', 'month')
        
    Returns:
        Grouped pandas DataFrame
    """
    # Convert date to datetime if it's not already
    df['date'] = pd.to_datetime(df['date'])
    
    # Define grouping based on period
    if period == 'day':
        df['period'] = df['date'].dt.strftime('%Y-%m-%d')
    elif period == 'week':
        df['period'] = df['date'].dt.strftime('%Y-%U')
    elif period == 'month':
        df['period'] = df['date'].dt.strftime('%Y-%m')
    else:
        df['period'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Prepare aggregation dictionary
    agg_dict = {'revenue': 'sum'}
    
    # Add transaction/order aggregation if available
    if 'transaction_id' in df.columns:
        agg_dict['transaction_id'] = 'nunique'
    elif 'orders' in df.columns:
        agg_dict['orders'] = 'sum'
        
    # Add customer aggregation if available
    if 'customer_id' in df.columns:
        agg_dict['customer_id'] = 'nunique'
    elif 'customers' in df.columns:
        agg_dict['customers'] = 'sum'
        
    # Add profit if available
    if 'profit' in df.columns:
        agg_dict['profit'] = 'sum'
    
    # Group by period and calculate metrics
    grouped_df = df.groupby('period').agg(agg_dict).reset_index()
    
    # Rename columns if needed for consistency
    if 'transaction_id' in grouped_df.columns:
        grouped_df.rename(columns={'transaction_id': 'orders'}, inplace=True)
    if 'customer_id' in grouped_df.columns:
        grouped_df.rename(columns={'customer_id': 'customers'}, inplace=True)
    
    # Calculate additional metrics
    if 'orders' in grouped_df.columns and grouped_df['orders'].sum() > 0:
        grouped_df['aov'] = grouped_df['revenue'] / grouped_df['orders']
    else:
        # If no orders or orders is all zeros, set AOV to zero to avoid division errors
        grouped_df['aov'] = 0
    
    if 'profit' in grouped_df.columns and grouped_df['revenue'].sum() > 0:
        grouped_df['profit_margin'] = grouped_df['profit'] / grouped_df['revenue']
    elif 'profit' in grouped_df.columns:
        grouped_df['profit_margin'] = 0
    
    return grouped_df

def calculate_product_metrics(df):
    """
    Calculate metrics for product analysis
    
    Args:
        df: pandas DataFrame with e-commerce data
        
    Returns:
        DataFrame with product metrics
    """
    if 'product_id' not in df.columns and 'product_name' not in df.columns:
        return pd.DataFrame()
    
    # Determine grouping column
    group_col = 'product_name' if 'product_name' in df.columns else 'product_id'
    
    # First, ensure we have transaction counts for each product
    # Count transactions per product
    if 'transaction_id' in df.columns:
        # Calculate transactions per product
        product_transactions = df.groupby(group_col)['transaction_id'].nunique().reset_index()
        product_transactions.rename(columns={'transaction_id': 'orders'}, inplace=True)
    else:
        # Fallback if no transaction_id column
        product_transactions = pd.DataFrame({
            group_col: df[group_col].unique(),
            'orders': 1  # Default to 1 if no transaction data
        })
    
    # Group by product for other metrics
    product_metrics = df.groupby(group_col).agg({
        'revenue': 'sum',
        'profit': 'sum' if 'profit' in df.columns else None,
    }).reset_index()
    
    # Merge with transaction counts
    product_metrics = pd.merge(product_metrics, product_transactions, on=group_col)
    
    # Calculate additional metrics
    if 'orders' in product_metrics.columns and product_metrics['orders'].sum() > 0:
        product_metrics['aov'] = product_metrics['revenue'] / product_metrics['orders'].replace(0, np.nan)
    else:
        product_metrics['aov'] = 0
        
    if 'profit' in product_metrics.columns:
        product_metrics['profit_margin'] = product_metrics['profit'] / product_metrics['revenue'].replace(0, np.nan)
        product_metrics['profit_per_order'] = product_metrics['profit'] / product_metrics['orders'].replace(0, np.nan)
    
    # Replace NaN with 0
    product_metrics = product_metrics.fillna(0)
    
    # Calculate contribution percentages
    total_revenue = product_metrics['revenue'].sum()
    product_metrics['revenue_contribution'] = product_metrics['revenue'] / total_revenue if total_revenue > 0 else 0
    
    # Sort by revenue (descending)
    product_metrics = product_metrics.sort_values('revenue', ascending=False)
    
    return product_metrics

def calculate_customer_metrics(df):
    """
    Calculate metrics for customer segmentation analysis
    
    Args:
        df: pandas DataFrame with e-commerce data
        
    Returns:
        DataFrame with customer metrics
    """
    if 'customer_segment' not in df.columns:
        return pd.DataFrame()
    
    # Initialize customer segment list
    segment_list = df['customer_segment'].unique()
    
    # Create base DataFrame with segments
    customer_metrics = pd.DataFrame({'customer_segment': segment_list})
    
    # Calculate revenue per segment
    if 'revenue' in df.columns:
        revenue_per_segment = df.groupby('customer_segment')['revenue'].sum().reset_index()
        customer_metrics = pd.merge(customer_metrics, revenue_per_segment, on='customer_segment', how='left')
    else:
        # Default value if revenue not available
        customer_metrics['revenue'] = 0
    
    # Calculate profit per segment if available
    if 'profit' in df.columns:
        profit_per_segment = df.groupby('customer_segment')['profit'].sum().reset_index()
        customer_metrics = pd.merge(customer_metrics, profit_per_segment, on='customer_segment', how='left')
    
    # Calculate customers per segment if customer_id available
    if 'customer_id' in df.columns:
        segment_customers = df.groupby('customer_segment')['customer_id'].nunique().reset_index()
        segment_customers.rename(columns={'customer_id': 'customers'}, inplace=True)
        customer_metrics = pd.merge(customer_metrics, segment_customers, on='customer_segment', how='left')
    else:
        # Use count of rows as estimate of customers
        customer_metrics['customers'] = df.groupby('customer_segment').size().reset_index(name='customers')['customers']
    
    # Calculate transactions per segment
    if 'transaction_id' in df.columns:
        segment_transactions = df.groupby('customer_segment')['transaction_id'].nunique().reset_index()
        segment_transactions.rename(columns={'transaction_id': 'orders'}, inplace=True)
        customer_metrics = pd.merge(customer_metrics, segment_transactions, on='customer_segment', how='left')
    else:
        # Fallback if no transaction_id column - use row count as proxy for orders
        orders_count = df.groupby('customer_segment').size().reset_index(name='orders')
        customer_metrics = pd.merge(customer_metrics, orders_count, on='customer_segment', how='left')
    
    # Ensure no missing values
    customer_metrics = customer_metrics.fillna(0)
    
    # Calculate additional metrics
    # Handle potential division by zero using replace
    if 'orders' in customer_metrics.columns and customer_metrics['orders'].sum() > 0:
        customer_metrics['aov'] = customer_metrics['revenue'] / customer_metrics['orders'].replace(0, np.nan)
    else:
        customer_metrics['aov'] = 0
        
    if 'customers' in customer_metrics.columns and customer_metrics['customers'].sum() > 0:
        customer_metrics['revenue_per_customer'] = customer_metrics['revenue'] / customer_metrics['customers'].replace(0, np.nan)
        customer_metrics['orders_per_customer'] = customer_metrics['orders'] / customer_metrics['customers'].replace(0, np.nan)
    else:
        customer_metrics['revenue_per_customer'] = 0
        customer_metrics['orders_per_customer'] = 0
    
    if 'profit' in customer_metrics.columns and not customer_metrics['profit'].isna().all():
        customer_metrics['profit_margin'] = customer_metrics['profit'] / customer_metrics['revenue'].replace(0, np.nan)
        if 'customers' in customer_metrics.columns and customer_metrics['customers'].sum() > 0:
            customer_metrics['profit_per_customer'] = customer_metrics['profit'] / customer_metrics['customers'].replace(0, np.nan)
        else:
            customer_metrics['profit_per_customer'] = 0
            
    # Replace NaN with 0
    customer_metrics = customer_metrics.fillna(0)
    
    # Sort by revenue (descending)
    customer_metrics = customer_metrics.sort_values('revenue', ascending=False)
    
    return customer_metrics

def calculate_rfm_metrics(df, customer_id_col='customer_id', date_col='date', revenue_col='revenue'):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for customer segmentation
    
    Args:
        df: pandas DataFrame with e-commerce data
        customer_id_col: Column name for customer ID
        date_col: Column name for transaction date
        revenue_col: Column name for revenue/monetary value
        
    Returns:
        DataFrame with RFM metrics and segments
    """
    if customer_id_col not in df.columns or date_col not in df.columns or revenue_col not in df.columns:
        return pd.DataFrame()
    
    # Ensure date is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Get the most recent date
    most_recent_date = df[date_col].max()
    
    # Calculate RFM metrics 
    freq_column = 'transaction_id' if 'transaction_id' in df.columns else df.index.name if df.index.name is not None else 'index'
    
    rfm = df.groupby(customer_id_col).agg({
        date_col: lambda x: (most_recent_date - x.max()).days,  # Recency
        freq_column: 'nunique' if 'transaction_id' in df.columns else 'count',  # Frequency
        revenue_col: 'sum'  # Monetary
    }).reset_index()
    
    # Rename columns
    rfm.columns = [customer_id_col, 'recency', 'frequency', 'monetary']
    
    # Create RFM scores (1-5) using quintiles
    # For recency, lower is better, so we reverse the labels
    rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    # Convert scores to strings for combination
    rfm['r_score'] = rfm['r_score'].astype(str)
    rfm['f_score'] = rfm['f_score'].astype(str)
    rfm['m_score'] = rfm['m_score'].astype(str)
    
    # Combined score (R, F, and M combined)
    rfm['rfm_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
    
    # Define customer segments based on RFM score
    segment_map = {
        '555': 'Champions',
        '554': 'Champions',
        '544': 'Champions',
        '545': 'Champions',
        '454': 'Loyal Customers',
        '455': 'Loyal Customers',
        '445': 'Loyal Customers',
        '444': 'Loyal Customers',
        '435': 'Potential Loyalists',
        '434': 'Potential Loyalists',
        '344': 'Potential Loyalists',
        '343': 'Potential Loyalists',
        '334': 'Potential Loyalists',
        '335': 'Potential Loyalists',
        '325': 'New Customers',
        '324': 'New Customers',
        '315': 'New Customers',
        '314': 'New Customers',
        '215': 'Promising',
        '214': 'Promising',
        '213': 'Promising',
        '115': 'Recent Customers',
        '114': 'Recent Customers',
        '113': 'Recent Customers',
        '134': 'Need Attention',
        '133': 'Need Attention',
        '124': 'Need Attention',
        '123': 'Need Attention',
        '145': 'Need Attention',
        '144': 'Need Attention',
        '243': 'At Risk',
        '234': 'At Risk',
        '233': 'At Risk',
        '225': 'At Risk',
        '224': 'At Risk',
        '223': 'At Risk',
        '155': 'Can\'t Lose Them',
        '154': 'Can\'t Lose Them',
        '153': 'Can\'t Lose Them',
        '152': 'Can\'t Lose Them',
        '151': 'Can\'t Lose Them',
        '245': 'Hibernating',
        '244': 'Hibernating',
        '235': 'Hibernating',
        '234': 'Hibernating',
        '232': 'Hibernating',
        '231': 'Hibernating',
        '125': 'About to Sleep',
        '124': 'About to Sleep',
        '123': 'About to Sleep',
        '122': 'About to Sleep',
        '121': 'About to Sleep',
        '111': 'Lost',
        '112': 'Lost',
        '121': 'Lost',
        '131': 'Lost',
        '141': 'Lost',
        '151': 'Lost'
    }
    
    # Map segments (with a default for any combinations not in our map)
    rfm['segment'] = rfm['rfm_score'].map(segment_map)
    rfm['segment'] = rfm['segment'].fillna('Other')
    
    # Calculate segment value (for prioritization)
    segment_value_map = {
        'Champions': 5,
        'Loyal Customers': 4.5,
        'Potential Loyalists': 4,
        'New Customers': 3.5,
        'Promising': 3,
        'Need Attention': 2.5,
        'At Risk': 2,
        'Can\'t Lose Them': 1.5,
        'Hibernating': 1,
        'About to Sleep': 0.5,
        'Lost': 0,
        'Other': 1
    }
    
    rfm['segment_value'] = rfm['segment'].map(segment_value_map)
    
    # Add RFM combined score (for easier sorting)
    rfm['rfm_combined'] = rfm['r_score'].astype(int) + rfm['f_score'].astype(int) + rfm['m_score'].astype(int)
    
    return rfm

def analyze_market_basket(df, min_support=0.01, min_confidence=0.2, min_lift=1.0):
    """
    Perform market basket analysis to identify product associations
    
    Args:
        df: pandas DataFrame with e-commerce data
        min_support: Minimum support threshold for frequent itemsets
        min_confidence: Minimum confidence threshold for rules
        min_lift: Minimum lift threshold for rules
        
    Returns:
        DataFrame with association rules
    """
    if 'product_name' not in df.columns:
        return pd.DataFrame()
    
    # Create a transaction dataset
    # If we have transaction or order IDs
    id_column = None
    if 'transaction_id' in df.columns:
        id_column = 'transaction_id'
    elif 'order_id' in df.columns:
        id_column = 'order_id'
    
    # Simple approach - count co-occurrences
    if id_column is None:
        # Create a simplified approach if we don't have transaction IDs
        product_list = df['product_name'].unique()
        
        # Create a co-occurrence matrix
        co_occurrence = pd.DataFrame(0, index=product_list, columns=product_list)
        
        # Fill the co-occurrence matrix
        for product in product_list:
            for other_product in product_list:
                if product != other_product:
                    # Count how often these products appear in the same time period
                    co_occurrence.loc[product, other_product] = (
                        (df['product_name'] == product) & 
                        (df['product_name'] == other_product)
                    ).sum()
        
        # Create a list of product pairs
        product_pairs = []
        
        for product in product_list:
            # Get products that co-occur with this product
            co_products = co_occurrence.loc[product].sort_values(ascending=False).index[:5]
            
            for co_product in co_products:
                if product != co_product and co_occurrence.loc[product, co_product] > 0:
                    product_pairs.append({
                        'product_a': product,
                        'product_b': co_product,
                        'co_occurrences': co_occurrence.loc[product, co_product],
                        'lift': co_occurrence.loc[product, co_product] / (df['product_name'] == co_product).sum()
                    })
        
        # Convert to DataFrame
        result = pd.DataFrame(product_pairs)
        
        # Sort by co-occurrences
        if not result.empty:
            result = result.sort_values('co_occurrences', ascending=False)
            
            # Add a relevance score
            result['relevance_score'] = result['co_occurrences'] * result['lift']
            result = result.sort_values('relevance_score', ascending=False)
        
        return result
    
    # More accurate approach using transaction IDs
    else:
        # Group by transaction ID to get baskets
        baskets = df.groupby([id_column, 'product_name'])['revenue'].sum().unstack().fillna(0)
        baskets = (baskets > 0).astype(int)
        
        # Simple association rule mining
        product_list = baskets.columns
        
        # Calculate support for each product
        supports = {}
        for product in product_list:
            supports[product] = baskets[product].sum() / len(baskets)
        
        # Find products that meet minimum support
        frequent_products = [product for product, support in supports.items() 
                            if support >= min_support]
        
        # Calculate associations
        associations = []
        
        for product_a in frequent_products:
            for product_b in frequent_products:
                if product_a != product_b:
                    # Calculate support for the pair (A and B)
                    support_a_and_b = ((baskets[product_a] == 1) & (baskets[product_b] == 1)).sum() / len(baskets)
                    
                    # Calculate confidence (P(B|A))
                    confidence = support_a_and_b / supports[product_a] if supports[product_a] > 0 else 0
                    
                    # Calculate lift
                    lift = confidence / supports[product_b] if supports[product_b] > 0 else 0
                    
                    if (support_a_and_b >= min_support and 
                        confidence >= min_confidence and 
                        lift >= min_lift):
                        
                        # Calculate co-occurrences for consistency with the simplified approach
                        co_occurrences = ((baskets[product_a] == 1) & (baskets[product_b] == 1)).sum()
                        
                        associations.append({
                            'product_a': product_a,  # Using product_a for consistency
                            'product_b': product_b,  # Using product_b for consistency
                            'co_occurrences': co_occurrences,
                            'support': support_a_and_b,
                            'confidence': confidence,
                            'lift': lift,
                            'relevance_score': lift * co_occurrences  # Adding relevance score
                        })
        
        # Convert to DataFrame
        result = pd.DataFrame(associations)
        
        # Sort by lift and add relevance score
        if not result.empty:
            result = result.sort_values('relevance_score', ascending=False)
        
        return result
