import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from utils.data_processor import (
    load_data, calculate_kpis, filter_data_by_date, group_data_by_period, 
    calculate_product_metrics, calculate_customer_metrics
)
from utils.visualization import (
    create_kpi_card, create_trend_chart, create_product_comparison_chart,
    create_customer_segment_chart, create_pareto_chart
)
from utils.forecasting import generate_forecast, create_forecast_chart

# Page configuration
st.set_page_config(
    page_title="InsightCommerce Pro",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for branding
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0C3A6D;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        font-style: italic;
        margin-top: -0.5rem;
    }
    .accent-text {
        color: #E67F14;
        font-weight: bold;
    }
    div.block-container {
        padding-top: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f9f9f9;
    }
    h1, h2, h3 {
        color: #0C3A6D;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #0C3A6D;
        font-size: 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E6F2FF;
        color: #0C3A6D;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App title with branding
st.markdown('<p class="main-header">Insight<span class="accent-text">Commerce</span> Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your intelligent analytics platform for e-commerce excellence</p>', unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

# Sidebar for data loading and filtering
with st.sidebar:
    # Add logo/branding to sidebar
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
        <div style="font-size: 1.2rem; font-weight: 700; color: #0C3A6D; margin-bottom: 5px;">
            Insight<span style="color: #E67F14;">Commerce</span> Pro
        </div>
        <div style="font-size: 0.8rem; color: #666; font-style: italic;">Business intelligence dashboard</div>
        <div style="margin-top: 10px; font-size: 0.7rem; color: #999;">v1.0.0 | Analytics Suite</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Data Controls")
    
    if st.session_state.data is None:
        st.info("No data loaded. Please upload data in the Data Import page or use the sample data.")
        if st.button("Load Sample Data"):
            # Generate some sample data for demonstration
            today = datetime.now()
            dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(90)]
            
            # Create sample data with realistic e-commerce metrics
            # Generate customer IDs (30 unique customers)
            customer_ids = [f'CUST-{str(i).zfill(3)}' for i in range(1, 31)]
            
            # Generate transaction IDs (60 unique transactions)
            transaction_ids = [f'TXN-{str(i).zfill(4)}' for i in range(1, 61)]
            
            # Create a more complex sample dataset with transaction IDs to enable market basket analysis
            # We'll create multiple rows per transaction to simulate multiple products in a single order
            sample_rows = []
            for i in range(150):  # Generate more records to create multiple products per transaction
                transaction_id = np.random.choice(transaction_ids)
                customer_id = np.random.choice(customer_ids)
                date = np.random.choice(dates)
                
                # Randomly select product
                product_id = np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'])
                
                # Map product IDs to names
                product_map = {
                    'P001': 'T-Shirt',
                    'P002': 'Jeans',
                    'P003': 'Sneakers',
                    'P004': 'Hoodie',
                    'P005': 'Hat'
                }
                
                # Map product IDs to categories
                category_map = {
                    'P001': 'Clothing',
                    'P002': 'Clothing',
                    'P003': 'Footwear',
                    'P004': 'Clothing',
                    'P005': 'Accessories'
                }
                
                # Generate metrics
                revenue = np.random.normal(500, 100)
                cost = np.random.normal(300, 50)
                
                # Add row to dataset
                sample_rows.append({
                    'transaction_id': transaction_id,
                    'date': date,
                    'customer_id': customer_id,
                    'product_id': product_id,
                    'product_name': product_map[product_id],
                    'category': category_map[product_id],
                    'revenue': revenue,
                    'cost': cost,
                    'customer_segment': np.random.choice(['New', 'Returning', 'Loyal']),
                    'customer_region': np.random.choice(['North', 'South', 'East', 'West', 'Central']),
                    'customer_acquisition_cost': np.random.normal(20, 5)
                })
            
            # Create DataFrame from rows
            sample_data = pd.DataFrame(sample_rows)
            
            # Aggregate some metrics by transaction for overall metrics
            transaction_totals = sample_data.groupby('transaction_id').agg({
                'revenue': 'sum',
                'customer_id': 'first'
            }).reset_index()
            
            # Summarize by date for time-based metrics
            sample_data_by_date = sample_data.groupby('date').agg({
                'transaction_id': 'nunique',
                'customer_id': 'nunique',
                'revenue': 'sum',
                'cost': 'sum'
            }).reset_index()
            
            # Rename columns for consistency with existing code
            sample_data_by_date.rename(columns={
                'transaction_id': 'orders',
                'customer_id': 'customers'
            }, inplace=True)
            
            # Calculate key metrics
            sample_data['profit'] = sample_data['revenue'] - sample_data['cost']
            
            # Add transaction-level metrics
            transaction_metrics = sample_data.groupby('transaction_id').agg({
                'revenue': 'sum',
                'profit': 'sum'
            }).reset_index()
            
            # Add date-level metrics
            date_metrics = sample_data.groupby('date').agg({
                'transaction_id': 'nunique',
                'customer_id': 'nunique'
            }).reset_index()
            date_metrics.rename(columns={'transaction_id': 'orders', 'customer_id': 'customers'}, inplace=True)
            
            # Generate conversion rate (this would normally come from actual web analytics)
            date_metrics['conversion_rate'] = np.random.uniform(0.01, 0.05, len(date_metrics))
            
            st.session_state.data = sample_data
            st.success("Sample data loaded successfully!")
    
    else:
        # Date range filter
        st.subheader("Date Range")
        min_date = pd.to_datetime(st.session_state.data['date']).min().date()
        max_date = pd.to_datetime(st.session_state.data['date']).max().date()
        
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Apply filters
        if start_date > end_date:
            st.error("Error: End date must be after start date")
            filtered_data = st.session_state.data
        else:
            filtered_data = filter_data_by_date(st.session_state.data, start_date, end_date)
            
        # Additional filters if data exists
        if 'product_name' in st.session_state.data.columns:
            st.subheader("Product Filter")
            all_products = ['All'] + sorted(st.session_state.data['product_name'].unique().tolist())
            selected_product = st.selectbox("Select Product", all_products)
            
            if selected_product != 'All':
                filtered_data = filtered_data[filtered_data['product_name'] == selected_product]
                
        if 'category' in st.session_state.data.columns:
            st.subheader("Category Filter")
            all_categories = ['All'] + sorted(st.session_state.data['category'].unique().tolist())
            selected_category = st.selectbox("Select Category", all_categories)
            
            if selected_category != 'All':
                filtered_data = filtered_data[filtered_data['category'] == selected_category]
                
        if 'customer_segment' in st.session_state.data.columns:
            st.subheader("Customer Segment")
            all_segments = ['All'] + sorted(st.session_state.data['customer_segment'].unique().tolist())
            selected_segment = st.selectbox("Select Segment", all_segments)
            
            if selected_segment != 'All':
                filtered_data = filtered_data[filtered_data['customer_segment'] == selected_segment]
                
        if 'customer_region' in st.session_state.data.columns:
            st.subheader("Region")
            all_regions = ['All'] + sorted(st.session_state.data['customer_region'].unique().tolist())
            selected_region = st.selectbox("Select Region", all_regions)
            
            if selected_region != 'All':
                filtered_data = filtered_data[filtered_data['customer_region'] == selected_region]
                
        # Store filtered data in session state for use in all pages
        st.session_state.filtered_data = filtered_data

# Main dashboard content
if st.session_state.data is not None:
    # Get filtered data
    if 'filtered_data' in st.session_state:
        df = st.session_state.filtered_data
    else:
        df = st.session_state.data
    
    # Calculate KPIs
    kpis = calculate_kpis(df)
    
    # Create tabs for different dashboard sections
    tab1, tab2, tab3 = st.tabs(["Performance Overview", "Profit Analysis", "Strategic Insights"])
    
    with tab1:
        # Display KPIs in a grid with enhanced styling
        st.header("Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_kpi_card("Total Revenue", f"£{kpis['total_revenue']:,.2f}", kpis['revenue_growth'])
            
        with col2:
            create_kpi_card("Total Orders", f"{kpis['total_orders']:,}", kpis['order_growth'])
            
        with col3:
            create_kpi_card("Conversion Rate", f"{kpis['avg_conversion_rate']:.2%}", kpis['conversion_rate_growth'])
            
        with col4:
            create_kpi_card("Average Order Value", f"£{kpis['avg_order_value']:,.2f}", kpis['aov_growth'])
            
        # Second row of KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_kpi_card("Total Profit", f"£{kpis['total_profit']:,.2f}", kpis['profit_growth'])
            
        with col2:
            create_kpi_card("Profit Margin", f"{kpis['profit_margin']:.2%}", kpis['profit_margin_growth'])
            
        with col3:
            create_kpi_card("Customer Acquisition Cost", f"£{kpis['avg_cac']:,.2f}", kpis['cac_growth'] * -1)  # Inverse growth for CAC
            
        with col4:
            create_kpi_card("Unique Customers", f"{kpis['unique_customers']:,}", kpis['customer_growth'])
        
        # Revenue trend chart with enhanced options
        st.header("Revenue & Profit Trends")
        
        # Add time period selector for trend analysis
        trend_period = st.radio(
            "Time aggregation:",
            ["Daily", "Weekly", "Monthly"],
            horizontal=True
        )
        
        period_map = {"Daily": "day", "Weekly": "week", "Monthly": "month"}
        period_lower = period_map[trend_period]
        
        # Group by selected period
        trend_data = group_data_by_period(df, period=period_lower)
        
        # Create the enhanced trend chart
        fig = create_trend_chart(
            trend_data, 
            metrics=['revenue', 'profit', 'orders'],
            title=f"{trend_period} Revenue, Profit and Orders Trends"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Profit Optimization Analysis")
        
        # Calculate profit metrics
        if 'profit' in df.columns and 'revenue' in df.columns:
            # Create profit metrics
            profit_metrics = {
                "Total Profit": f"£{kpis['total_profit']:,.2f}",
                "Overall Profit Margin": f"{kpis['profit_margin']:.2%}",
                "Profit per Order": f"£{kpis['total_profit']/kpis['total_orders']:,.2f}" if kpis['total_orders'] > 0 else "N/A",
                "Profit per Customer": f"£{kpis['total_profit']/kpis['unique_customers']:,.2f}" if kpis['unique_customers'] > 0 else "N/A",
                "Return on Ad Spend": f"{kpis['total_profit']/(df['customer_acquisition_cost'].sum()):,.2f}x" if 'customer_acquisition_cost' in df.columns else "N/A"
            }
            
            # Create columns for profit metrics
            col1, col2, col3 = st.columns(3)
            
            keys = list(profit_metrics.keys())
            for i, key in enumerate(keys):
                with [col1, col2, col3][i % 3]:
                    st.metric(key, profit_metrics[key])
            
            # Profit Driver Analysis
            st.subheader("Profit Drivers")
            
            if 'product_name' in df.columns:
                # Group by product and calculate profitability metrics
                agg_dict = {
                    'revenue': 'sum',
                    'profit': 'sum'
                }
                
                # Handle orders calculation based on available data
                if 'transaction_id' in df.columns:
                    # Count unique transactions per product
                    product_orders = df.groupby('product_name')['transaction_id'].nunique().reset_index()
                    product_orders.rename(columns={'transaction_id': 'orders'}, inplace=True)
                    
                    # Group by product for other metrics
                    product_profit = df.groupby('product_name').agg(agg_dict).reset_index()
                    
                    # Merge with orders counts
                    product_profit = pd.merge(product_profit, product_orders, on='product_name', how='left')
                elif 'orders' in df.columns:
                    # If orders column exists, use it directly
                    agg_dict['orders'] = 'sum'
                    product_profit = df.groupby('product_name').agg(agg_dict).reset_index()
                else:
                    # Count rows as proxy for orders if neither exists
                    product_profit = df.groupby('product_name').agg(agg_dict).reset_index()
                    # Add a count of rows as orders
                    orders_count = df.groupby('product_name').size().reset_index(name='orders')
                    product_profit = pd.merge(product_profit, orders_count, on='product_name', how='left')
                
                product_profit['profit_margin'] = product_profit['profit'] / product_profit['revenue']
                product_profit['profit_contribution'] = product_profit['profit'] / product_profit['profit'].sum()
                product_profit = product_profit.sort_values('profit', ascending=False)
                
                # Create a more advanced chart
                fig = px.bar(
                    product_profit.head(10),
                    x='product_name',
                    y='profit',
                    color='profit_margin',
                    text='profit_margin',
                    color_continuous_scale='RdYlGn',
                    title='Top 10 Products by Profit',
                    labels={
                        'product_name': 'Product',
                        'profit': 'Profit (£)',
                        'profit_margin': 'Profit Margin'
                    }
                )
                
                fig.update_traces(
                    texttemplate='%{text:.1%}',
                    textposition='outside'
                )
                
                fig.update_layout(
                    xaxis_title='Product',
                    yaxis_title='Profit (£)',
                    coloraxis_colorbar_title='Profit Margin'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Profit bottlenecks - Lowest margin products
                st.subheader("Profit Improvement Opportunities")
                
                low_margin = product_profit.sort_values('profit_margin', ascending=True).head(5)
                high_revenue_low_margin = product_profit[product_profit['revenue'] > product_profit['revenue'].median()].sort_values('profit_margin', ascending=True).head(5)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Lowest Margin Products")
                    st.dataframe(
                        low_margin[['product_name', 'profit_margin', 'revenue', 'profit']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            'product_name': 'Product',
                            'profit_margin': st.column_config.NumberColumn('Profit Margin', format="%.2f%%", width="medium"),
                            'revenue': st.column_config.NumberColumn('Revenue', format="£%.2f", width="medium"),
                            'profit': st.column_config.NumberColumn('Profit', format="£%.2f", width="medium")
                        }
                    )
                    
                    # Opportunity calculation for lowest margin products
                    avg_margin = product_profit['profit_margin'].median()
                    potential_profit_increase = sum([(avg_margin - row['profit_margin']) * row['revenue'] for _, row in low_margin.iterrows()])
                    
                    st.metric(
                        "Potential Profit Increase",
                        f"£{potential_profit_increase:,.2f}",
                        help="Estimated profit increase if these products achieved average profit margin"
                    )
                
                with col2:
                    st.markdown("#### High Revenue, Low Margin Products")
                    st.dataframe(
                        high_revenue_low_margin[['product_name', 'profit_margin', 'revenue', 'profit']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            'product_name': 'Product',
                            'profit_margin': st.column_config.NumberColumn('Profit Margin', format="%.2f%%", width="medium"),
                            'revenue': st.column_config.NumberColumn('Revenue', format="£%.2f", width="medium"),
                            'profit': st.column_config.NumberColumn('Profit', format="£%.2f", width="medium")
                        }
                    )
                    
                    # Opportunity calculation for high revenue, low margin products
                    high_rev_potential = sum([(avg_margin - row['profit_margin']) * row['revenue'] for _, row in high_revenue_low_margin.iterrows()])
                    
                    st.metric(
                        "High-Value Opportunity",
                        f"£{high_rev_potential:,.2f}",
                        help="Estimated profit increase if these high-revenue products achieved average profit margin"
                    )
                
                # Profit optimization recommendations
                with st.expander("Profit Optimization Recommendations", expanded=True):
                    st.markdown("""
                    ### Recommendations to Maximize Profitability
                    
                    #### For Low Margin Products:
                    1. **Review Pricing Strategy**: Consider increasing prices by 5-10% for products with margins below average
                    2. **Negotiate with Suppliers**: Seek volume discounts or alternative suppliers for high-cost inputs
                    3. **Streamline Operations**: Identify and eliminate inefficiencies in production or fulfillment
                    4. **Bundle with High-Margin Products**: Create attractive packages that include high-margin items
                    
                    #### For High Revenue, Low Margin Products:
                    1. **Premium Versions**: Develop premium versions with additional features/benefits at higher margins
                    2. **Focus on Add-Ons**: Promote complementary products with higher margins
                    3. **Review Discounting Practices**: Ensure discounts on these products are measured and strategic
                    4. **Customer Segmentation**: Adjust pricing based on customer segments that are less price-sensitive
                    
                    #### Overall Profit Enhancement Strategies:
                    1. **Optimize Marketing Spend**: Focus on channels with highest return on ad spend
                    2. **Enhance Customer Lifetime Value**: Develop loyalty programs to reduce acquisition costs
                    3. **Inventory Management**: Reduce costs associated with overstocking or stockouts
                    4. **Dynamic Pricing**: Implement price adjustments based on demand, seasonality, and competition
                    """)
    
    with tab3:
        st.header("Strategic Business Insights")
        
        # Revenue & profit forecast teaser
        st.subheader("Revenue & Profit Forecast Preview")
        
        try:
            # Generate a quick forecast for next 30 days
            if len(df) > 14:  # Only if we have enough data
                daily_data = df.groupby('date').agg({
                    'revenue': 'sum'
                }).reset_index()
                
                forecast_df, _ = generate_forecast(
                    daily_data,
                    periods=30,
                    model_type='linear',
                    target_col='revenue'
                )
                
                # Create the forecast chart
                fig = create_forecast_chart(
                    forecast_df,
                    target_col='revenue',
                    title='30-Day Revenue Forecast Preview'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate total forecasted revenue
                forecast_only = forecast_df[forecast_df['type'] == 'Forecast']
                total_forecast = forecast_only['revenue'].sum()
                
                st.info(f"Projected Revenue (Next 30 Days): £{total_forecast:,.2f} - **View detailed forecast analysis in the Sales Forecast page**")
            else:
                st.info("Insufficient data for forecasting. Please upload more historical data for forecast projections.")
        except Exception as e:
            st.warning(f"Couldn't generate forecast preview: {str(e)}. Try the full forecast on the Sales Forecast page.")
        
        # Key insights section 
        st.subheader("Key Business Insights")
        
        # Generate some dynamic insights based on the data
        insights = []
        
        # Revenue trend insight
        if kpis['revenue_growth'] > 0.1:
            insights.append("**Strong Revenue Growth**: Revenue is growing rapidly at {:.1%}. Focus on scaling operations to meet increased demand.".format(kpis['revenue_growth']))
        elif kpis['revenue_growth'] > 0:
            insights.append("**Steady Revenue Growth**: Revenue shows positive growth at {:.1%}. Continue current strategies while looking for growth opportunities.".format(kpis['revenue_growth']))
        else:
            insights.append("**Revenue Decline**: Revenue is declining at {:.1%}. Urgent review of pricing, marketing and product strategies needed.".format(kpis['revenue_growth']))
        
        # Profit margin insight
        if kpis['profit_margin'] < 0.2:
            insights.append("**Low Profit Margin**: Overall profit margin of {:.1%} is relatively low. Focus on cost reduction and pricing strategies.".format(kpis['profit_margin']))
        elif kpis['profit_margin'] > 0.4:
            insights.append("**Excellent Profit Margin**: Strong profit margin of {:.1%}. Consider reinvesting in growth while maintaining operational efficiency.".format(kpis['profit_margin']))
        
        # Customer insights
        if 'customer_segment' in df.columns:
            segment_data = df.groupby('customer_segment').agg({'revenue': 'sum'}).reset_index()
            top_segment = segment_data.loc[segment_data['revenue'].idxmax(), 'customer_segment']
            segment_revenue_pct = segment_data.loc[segment_data['revenue'].idxmax(), 'revenue'] / segment_data['revenue'].sum()
            
            if segment_revenue_pct > 0.6:
                insights.append(f"**Customer Concentration Risk**: The '{top_segment}' segment accounts for {segment_revenue_pct:.1%} of revenue. Consider diversification strategies.")
            else:
                insights.append(f"**Balanced Customer Segments**: The '{top_segment}' segment is your strongest at {segment_revenue_pct:.1%} of revenue. Continue targeted marketing to all segments.")
        
        # Product insights
        if 'product_name' in df.columns:
            product_data = df.groupby('product_name').agg({'revenue': 'sum', 'profit': 'sum'}).reset_index()
            product_data['profit_margin'] = product_data['profit'] / product_data['revenue']
            
            top_product = product_data.loc[product_data['revenue'].idxmax(), 'product_name']
            top_product_pct = product_data.loc[product_data['revenue'].idxmax(), 'revenue'] / product_data['revenue'].sum()
            
            if top_product_pct > 0.3:
                insights.append(f"**Product Dependency**: '{top_product}' generates {top_product_pct:.1%} of revenue. Consider developing complementary products.")
            
            # Find most profitable product
            most_profitable = product_data.loc[product_data['profit_margin'].idxmax()]
            insights.append(f"**Most Profitable Product**: '{most_profitable['product_name']}' has your highest profit margin at {most_profitable['profit_margin']:.1%}. Consider expansion opportunities.")
        
        # Display insights
        for i, insight in enumerate(insights):
            st.markdown(f"{i+1}. {insight}")
        
        # Strategy recommendations
        st.subheader("Strategic Recommendations")
        
        with st.expander("View Strategic Recommendations", expanded=True):
            st.markdown("""
            ### Short-Term Actions (Next 30 Days)
            
            1. **Price Optimization**:
               - Review prices of lowest-margin products
               - Test price elasticity with small increases on top-selling items
               - Implement seasonal promotions on high-margin products
            
            2. **Marketing Focus**:
               - Shift ad spend toward highest-converting channels
               - Create targeted campaigns for most profitable customer segments
               - Develop retargeting campaigns for abandoned carts
            
            ### Medium-Term Strategies (1-3 Months)
            
            1. **Product Portfolio Optimization**:
               - Increase inventory of high-margin, high-demand products
               - Consider phasing out consistently underperforming products
               - Develop product bundles that combine high and low margin items
            
            2. **Customer Experience Enhancements**:
               - Implement post-purchase follow-up sequences
               - Develop loyalty program for repeat customers
               - Streamline checkout process to improve conversion rates
            
            ### Long-Term Initiatives (3-12 Months)
            
            1. **Business Development**:
               - Explore new markets or customer segments
               - Evaluate channel expansion opportunities
               - Consider strategic partnerships to reduce costs or expand reach
            
            2. **Operational Efficiency**:
               - Review and optimize supply chain
               - Identify automation opportunities
               - Develop better inventory forecasting models
            """)
    
    # Display product and customer segments on main dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        # Product performance overview
        st.header("Top Products")
        
        if 'product_name' in df.columns and 'revenue' in df.columns:
            # Build the aggregation dictionary
            agg_dict = {
                'revenue': 'sum',
                'profit': 'sum' if 'profit' in df.columns else None
            }
            agg_dict = {k: v for k, v in agg_dict.items() if v is not None}
            
            # Handle orders calculation based on available data
            if 'transaction_id' in df.columns:
                # Count unique transactions per product
                product_orders = df.groupby('product_name')['transaction_id'].nunique().reset_index()
                product_orders.rename(columns={'transaction_id': 'orders'}, inplace=True)
                
                # Group by product for other metrics
                product_data = df.groupby('product_name').agg(agg_dict).reset_index()
                
                # Merge with orders counts
                product_data = pd.merge(product_data, product_orders, on='product_name', how='left')
            elif 'orders' in df.columns:
                # If orders column exists, use it directly
                agg_dict['orders'] = 'sum'
                product_data = df.groupby('product_name').agg(agg_dict).reset_index()
            else:
                # Count rows as proxy for orders if neither exists
                product_data = df.groupby('product_name').agg(agg_dict).reset_index()
                # Add a count of rows as orders
                orders_count = df.groupby('product_name').size().reset_index(name='orders')
                product_data = pd.merge(product_data, orders_count, on='product_name', how='left')
            
            product_data['profit_margin'] = product_data['profit'] / product_data['revenue']
            product_data = product_data.sort_values('revenue', ascending=False)
            
            # Create the bar chart
            fig = px.bar(
                product_data.head(5),
                x='product_name',
                y='revenue',
                color='profit_margin',
                color_continuous_scale='RdYlGn',
                text_auto='.2s',
                title='Top 5 Products by Revenue',
                labels={
                    'product_name': 'Product',
                    'revenue': 'Revenue (£)',
                    'profit_margin': 'Profit Margin'
                }
            )
            
            fig.update_layout(
                xaxis_title='Product',
                yaxis_title='Revenue (£)',
                coloraxis_colorbar_title='Profit Margin',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer segments
        if 'customer_segment' in df.columns:
            st.header("Customer Segments")
            
            # Build the base aggregation dictionary
            agg_dict = {
                'revenue': 'sum'
            }
            
            # Initialize variables for separate calculations
            segment_orders = None
            segment_customers = None
            
            # Handle orders calculation
            if 'transaction_id' in df.columns:
                # Count unique transactions per segment
                segment_orders = df.groupby('customer_segment')['transaction_id'].nunique().reset_index()
                segment_orders.rename(columns={'transaction_id': 'orders'}, inplace=True)
            elif 'orders' in df.columns:
                agg_dict['orders'] = 'sum'
            
            # Handle customers calculation
            if 'customer_id' in df.columns:
                segment_customers = df.groupby('customer_segment')['customer_id'].nunique().reset_index()
                segment_customers.rename(columns={'customer_id': 'customers'}, inplace=True)
            elif 'customers' in df.columns:
                agg_dict['customers'] = 'sum'
            
            # Group by customer segment for base metrics
            segment_data = df.groupby('customer_segment').agg(agg_dict).reset_index()
            
            # Merge in orders if calculated separately
            if segment_orders is not None:
                segment_data = pd.merge(segment_data, segment_orders, on='customer_segment', how='left')
            
            # Merge in customers if calculated separately
            if segment_customers is not None:
                segment_data = pd.merge(segment_data, segment_customers, on='customer_segment', how='left')
            
            # If neither customers nor customer_id exists, estimate customers based on segments
            if 'customers' not in segment_data.columns:
                segment_data['customers'] = df.groupby('customer_segment').size().reset_index(name='customers')['customers']
            
            # If orders isn't calculated yet, use segment rows as proxy
            if 'orders' not in segment_data.columns:
                segment_data['orders'] = df.groupby('customer_segment').size().reset_index(name='orders')['orders']
            
            # Calculate average revenue per customer
            segment_data['avg_revenue_per_customer'] = segment_data['revenue'] / segment_data['customers']
            
            # Create the chart
            fig = px.pie(
                segment_data,
                values='revenue',
                names='customer_segment',
                title='Revenue by Customer Segment',
                hover_data=['orders', 'customers', 'avg_revenue_per_customer'],
                labels={
                    'avg_revenue_per_customer': 'Avg. Revenue per Customer (£)'
                },
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer with instructions
    st.info(
        "Navigate to the different pages using the sidebar to explore detailed analysis, "
        "forecasting, and data import options. Use the date and other filters to narrow down your analysis."
    )
    
    # Instructions if no data is loaded yet
else:
    # Main welcome content with branding
    st.markdown('<h2 style="color:#0C3A6D;">Welcome to Insight<span style="color:#E67F14;">Commerce</span> Pro</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .welcome-text {
        background-color: #F0F5FA;
        border-left: 5px solid #0C3A6D;
        padding: 20px;
        border-radius: 5px;
    }
    .feature-item {
        margin-bottom: 10px;
    }
    .feature-icon {
        color: #E67F14;
        font-weight: bold;
        margin-right: 5px;
    }
    </style>
    
    <div class="welcome-text">
    <p>Your intelligent analytics platform designed to transform complex e-commerce data into actionable business insights.</p>
    
    <p><strong>To get started:</strong></p>
    <ol>
        <li>Go to the <strong>Data Import</strong> page to upload your e-commerce data (CSV format)</li>
        <li>Or use the <strong>Load Sample Data</strong> button in the sidebar to explore dashboard features</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Features with icons
    st.markdown("### Platform Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-item">
            <span class="feature-icon">📊</span> <strong>Dashboard:</strong> Real-time KPI overview
        </div>
        <div class="feature-item">
            <span class="feature-icon">📈</span> <strong>Sales History:</strong> In-depth sales analysis
        </div>
        <div class="feature-item">
            <span class="feature-icon">🛍️</span> <strong>Product Analysis:</strong> Catalog performance metrics
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-item">
            <span class="feature-icon">👥</span> <strong>Customer Segmentation:</strong> Audience insights
        </div>
        <div class="feature-item">
            <span class="feature-icon">🔮</span> <strong>Sales Forecast:</strong> AI-powered predictions
        </div>
        <div class="feature-item">
            <span class="feature-icon">📁</span> <strong>Data Import:</strong> Seamless data management
        </div>
        """, unsafe_allow_html=True)
    
    # Add a branded visual
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ca.svg", width=100)
    
    # Add a footer
    st.markdown("""
    <div style="margin-top: 50px; padding: 20px; text-align: center; border-top: 1px solid #eee; font-size: 0.8rem;">
        <span style="color: #0C3A6D; font-weight: bold;">Insight<span style="color: #E67F14;">Commerce</span> Pro</span> | &copy; 2025 | Version 1.0.0
    </div>
    """, unsafe_allow_html=True)
