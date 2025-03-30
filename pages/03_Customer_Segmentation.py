import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processor import calculate_customer_metrics, calculate_rfm_metrics
from utils.visualization import create_customer_segment_chart, create_scatter_plot, create_heatmap

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation - E-Commerce Analytics",
    page_icon="👥",
    layout="wide"
)

# Initialize session state if not already done
if 'data' not in st.session_state:
    st.session_state.data = None

# Page header
st.title("Customer Segmentation Analysis")
st.markdown("Understand your customer base and identify high-value segments")

# Check if data is loaded
if st.session_state.data is None:
    st.warning("No data loaded. Please go to the Data Import page or load sample data from the Dashboard.")
    st.stop()

# Get filtered data from session state
if 'filtered_data' in st.session_state:
    df = st.session_state.filtered_data
else:
    df = st.session_state.data

# Check if customer segmentation data is available
has_segment_data = 'customer_segment' in df.columns
has_region_data = 'customer_region' in df.columns
has_customer_id = 'customer_id' in df.columns

if not (has_segment_data or has_region_data):
    st.warning("Limited customer data available. For full segmentation analysis, your data should include 'customer_segment' and/or 'customer_region' columns.")

# Customer overview
st.header("Customer Overview")

# Calculate total customers
if has_customer_id:
    total_customers = df['customer_id'].nunique()
elif 'customers' in df.columns:
    total_customers = df['customers'].sum()
else:
    total_customers = "N/A"

# Calculate key metrics
total_revenue = df['revenue'].sum()
total_orders = df['orders'].sum() if 'orders' in df.columns else 0

if total_customers != "N/A" and total_customers > 0:
    revenue_per_customer = total_revenue / total_customers
    orders_per_customer = total_orders / total_customers
else:
    revenue_per_customer = "N/A"
    orders_per_customer = "N/A"

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", f"{total_customers:,}" if total_customers != "N/A" else "N/A")
    
with col2:
    st.metric("Total Revenue", f"£{total_revenue:,.2f}")
    
with col3:
    st.metric("Revenue per Customer", f"£{revenue_per_customer:,.2f}" if revenue_per_customer != "N/A" else "N/A")
    
with col4:
    st.metric("Orders per Customer", f"{orders_per_customer:.2f}" if orders_per_customer != "N/A" else "N/A")

# Segment analysis
if has_segment_data:
    st.header("Customer Segment Analysis")
    
    # Calculate segment metrics
    segment_metrics = calculate_customer_metrics(df)
    
    # Segment distribution
    st.subheader("Segment Distribution")
    
    # Create segment pie chart for revenue
    fig = create_customer_segment_chart(
        segment_metrics,
        values='revenue',
        names='customer_segment',
        title='Revenue by Customer Segment'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment comparison
    st.subheader("Segment Comparison")
    
    # Select metrics to compare
    compare_metrics = ["Revenue per Customer", "Orders per Customer"]
    if 'profit' in segment_metrics.columns:
        compare_metrics.append("Profit per Customer")
        compare_metrics.append("Profit Margin")
    
    selected_metrics = st.multiselect(
        "Select metrics to compare across segments:",
        compare_metrics,
        default=["Revenue per Customer"]
    )
    
    if not selected_metrics:
        st.info("Please select at least one metric to compare.")
    else:
        # Create a figure for each selected metric
        for metric in selected_metrics:
            # Map selected metric to column name
            if metric == "Revenue per Customer":
                column = "revenue_per_customer"
                format_str = "£{:,.2f}"
            elif metric == "Orders per Customer":
                column = "orders_per_customer"
                format_str = "{:.2f}"
            elif metric == "Profit per Customer":
                column = "profit_per_customer"
                format_str = "£{:,.2f}"
            elif metric == "Profit Margin":
                column = "profit_margin"
                format_str = "{:.2%}"
            
            # Create bar chart
            fig = px.bar(
                segment_metrics,
                x='customer_segment',
                y=column,
                color=column,
                title=f'{metric} by Segment',
                labels={
                    'customer_segment': 'Customer Segment',
                    column: metric
                },
                text_auto=True,
                color_continuous_scale='Viridis'
            )
            
            # Format the values
            fig.update_traces(
                texttemplate=format_str.format(0),
                textposition='outside'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # RFM Analysis (if available)
    st.header("Customer Behavior Analysis")
    
    # Use our RFM function if customer_id is available
    if has_customer_id and 'date' in df.columns and 'revenue' in df.columns:
        st.subheader("RFM (Recency, Frequency, Monetary) Analysis")
        
        st.markdown("""
        RFM analysis segments customers based on:
        - **Recency**: How recently a customer made a purchase
        - **Frequency**: How often they purchase
        - **Monetary**: How much they spend
        """)
        
        # Calculate RFM metrics using our utility function
        rfm = calculate_rfm_metrics(df)
        
        # Get tabs for different RFM visualizations
        rfm_tab1, rfm_tab2, rfm_tab3, rfm_tab4 = st.tabs([
            "3D Visualization", 
            "Segment Distribution", 
            "Segment Metrics", 
            "Segment Recommendations"
        ])
        
        with rfm_tab1:
            # Show 3D scatter plot of RFM
            fig = px.scatter_3d(
                rfm,
                x='recency',
                y='frequency',
                z='monetary',
                color='segment',
                opacity=0.7,
                title='3D RFM Analysis',
                labels={
                    'recency': 'Recency (days)',
                    'frequency': 'Frequency (orders)',
                    'monetary': 'Monetary (£)'
                },
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            # Add hover information
            fig.update_traces(
                hovertemplate="<b>%{customdata[0]}</b><br>Recency: %{x} days<br>Frequency: %{y} orders<br>Monetary: £%{z:.2f}<extra></extra>",
                customdata=[[s] for s in rfm['segment']]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add RFM Score distribution heatmap
            st.subheader("RFM Score Distribution")
            
            # Create a DataFrame for r_score and f_score combinations
            r_f_counts = rfm.groupby(['r_score', 'f_score']).size().reset_index(name='count')
            
            # Convert to matrix format for heatmap
            r_f_matrix = r_f_counts.pivot(index='r_score', columns='f_score', values='count').fillna(0)
            
            # Create heatmap
            fig = px.imshow(
                r_f_matrix,
                labels=dict(x="Frequency Score", y="Recency Score", color="Customer Count"),
                x=sorted(rfm['f_score'].unique()),
                y=sorted(rfm['r_score'].unique(), reverse=True),  # Reverse to show 5 (highest) at the top
                color_continuous_scale="YlGnBu",
                aspect="auto",
                title="Distribution of Customers by Recency and Frequency Scores"
            )
            
            fig.update_layout(
                xaxis_title="Frequency Score (1-5)",
                yaxis_title="Recency Score (1-5)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with rfm_tab2:
            # Segment distribution
            segment_counts = rfm['segment'].value_counts().reset_index()
            segment_counts.columns = ['segment', 'count']
            
            fig = px.pie(
                segment_counts,
                values='count',
                names='segment',
                title='Customer Segment Distribution',
                color_discrete_sequence=px.colors.qualitative.Bold,
                hole=0.4
            )
            
            # Improve layout
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a bar chart showing segment value distribution
            fig = px.bar(
                rfm.groupby('segment')['segment_value'].first().sort_values(ascending=False).reset_index(),
                x='segment',
                y='segment_value',
                color='segment_value',
                title='Customer Segment Value (Higher is Better)',
                labels={
                    'segment': 'Segment',
                    'segment_value': 'Segment Value'
                },
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with rfm_tab3:
            # Segment metrics table
            segment_metrics = rfm.groupby('segment').agg({
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': 'mean',
                'customer_id': 'count',
                'segment_value': 'first',
                'rfm_combined': 'mean'
            }).reset_index()
            
            # Sort by segment value (descending)
            segment_metrics = segment_metrics.sort_values('segment_value', ascending=False)
            
            # Rename columns for display
            segment_metrics.columns = ['Segment', 'Avg. Recency (days)', 'Avg. Frequency (orders)', 
                                    'Avg. Monetary (£)', 'Count', 'Segment Value', 'Avg. RFM Score']
            
            st.dataframe(
                segment_metrics,
                use_container_width=True,
                column_config={
                    'Segment': 'Segment',
                    'Avg. Recency (days)': st.column_config.NumberColumn(
                        'Avg. Recency (days)',
                        format="%.1f"
                    ),
                    'Avg. Frequency (orders)': st.column_config.NumberColumn(
                        'Avg. Frequency (orders)',
                        format="%.2f"
                    ),
                    'Avg. Monetary (£)': st.column_config.NumberColumn(
                        'Avg. Monetary (£)',
                        format="£%.2f"
                    ),
                    'Count': st.column_config.NumberColumn(
                        'Count',
                        format="%d"
                    ),
                    'Segment Value': st.column_config.NumberColumn(
                        'Segment Value (5=Best)',
                        format="%.1f"
                    ),
                    'Avg. RFM Score': st.column_config.NumberColumn(
                        'Avg. RFM Score',
                        format="%.1f"
                    )
                }
            )
            
            # Create a parallel coordinates plot to compare segments
            fig = px.parallel_coordinates(
                segment_metrics, 
                dimensions=['Avg. Recency (days)', 'Avg. Frequency (orders)', 
                            'Avg. Monetary (£)', 'Count', 'Segment Value'],
                color='Segment Value',
                color_continuous_scale=px.colors.sequential.Viridis,
                title='Parallel Coordinates Plot of Customer Segments'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with rfm_tab4:
            # Segment characteristics and recommendations
            st.markdown("""
            ## Customer Segment Characteristics and Recommendations
            
            ### High-Value Segments
            
            **Champions (555, 554, 544, 545):**
            - Bought recently, buy often, and spend the most
            - Highly engaged and loyal customers
            - **Recommendations:** 
                - Reward these customers with loyalty bonuses and exclusive offers
                - Seek their feedback on new products and services
                - Make them feel valued with personalized communications
                - Ask for reviews and testimonials
                - Develop referral programs to leverage their network
            
            **Loyal Customers (454, 455, 445, 444):**
            - Buy regularly and recently
            - Strong repeat purchase pattern
            - **Recommendations:** 
                - Upsell higher-value products or premium versions
                - Invite to exclusive loyalty programs with additional benefits
                - Cross-sell complementary products based on purchase history
                - Provide early access to new products and sales
            
            **Potential Loyalists (435, 434, 344, etc.):**
            - Recent customers with average frequency
            - Show promising engagement patterns
            - **Recommendations:** 
                - Offer membership or loyalty program to encourage repeat purchases
                - Provide personalized product recommendations
                - Send targeted content showcasing product benefits
                - Create special "second purchase" offers
            
            ### Growth Opportunity Segments
            
            **New Customers (325, 324, etc.):**
            - Bought more recently, but not frequently
            - Still in the evaluation phase
            - **Recommendations:** 
                - Provide excellent onboarding experience
                - Send educational content about products
                - Offer support and usage guidance
                - Request feedback on first purchase experience
                - Send timely reminders for repurchase
            
            **Promising (215, 214, etc.):**
            - Recent first-time buyers with good initial spend
            - **Recommendations:**
                - Focus on first 30-day experience
                - Welcome series emails with product tips
                - Early re-engagement offers if no repeat purchase
            
            **Need Attention (134, 133, etc.):**
            - Active but decreasing engagement
            - **Recommendations:**
                - Survey to understand changing needs
                - Re-engagement campaigns with special incentives
                - Promote new product features or improvements
            
            ### At-Risk Segments
            
            **At Risk (243, 234, etc.):**
            - Above average history but haven't purchased recently
            - Previously good customers showing declining engagement
            - **Recommendations:** 
                - Reactivation campaign with personalized message
                - Special "we miss you" offers
                - Product recommendations based on past purchases
                - Request feedback on why they stopped purchasing
            
            **Can't Lose Them (155, 154, etc.):**
            - Made big purchases but haven't returned recently
            - High-value customers at risk of churning
            - **Recommendations:** 
                - High-priority win-back campaigns
                - New product announcements aligned with past preferences
                - Personalized offers with strong incentives
                - Direct outreach from account managers (for B2B)
            
            **Hibernating (245, 244, etc.):**
            - Last purchase was some time ago
            - **Recommendations:**
                - Reactivation campaigns
                - "We've improved" messaging
                - Surveys to understand what went wrong
            
            **About to Sleep (125, 124, etc.):**
            - Recent decline in activity
            - **Recommendations:**
                - Re-engagement emails before they fully lapse
                - Targeted offers based on previous purchases
                - Remind of benefits and value proposition
            
            **Lost (111, 112, etc.):**
            - Lowest recency, frequency, and monetary scores
            - Long-inactive customers
            - **Recommendations:** 
                - Low-cost reactivation campaigns
                - Re-engage only with high-margin offers
                - Consider excluding from regular campaigns
                - Last-attempt win-back with significant offer
            """)
        
        # Customer Lifetime Value Analysis
        st.subheader("Customer Lifetime Value (CLV) Analysis")
        
        st.markdown("""
        Customer Lifetime Value (CLV) estimates the total revenue a business can expect from a customer 
        throughout their relationship. This helps prioritize marketing and retention efforts.
        """)
        
        # Create CLV calculation
        if 'date' in df.columns and has_customer_id:
            # Calculate parameters for CLV
            # First, convert dates to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date
            df = df.sort_values('date')
            
            # Calculate time between purchases (in days)
            customer_purchase_dates = df.groupby('customer_id')['date'].apply(list).reset_index()
            
            # Calculate average purchase frequency (average time between orders)
            purchase_intervals = []
            
            for _, row in customer_purchase_dates.iterrows():
                dates = row['date']
                if len(dates) > 1:
                    # Calculate intervals between consecutive purchases
                    intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                    purchase_intervals.extend(intervals)
            
            if purchase_intervals:
                avg_purchase_interval = np.mean(purchase_intervals)
                purchase_frequency = 365 / avg_purchase_interval  # Annual purchase frequency
            else:
                purchase_frequency = 1  # Default to 1 purchase per year if we can't calculate
            
            # Calculate average order value
            avg_order_value = df.groupby(['customer_id', 'date'])['revenue'].sum().mean()
            
            # Calculate customer average lifespan (in years)
            # For this demo, we'll estimate based on the RFM data
            # In a real system, this would be calculated from historical data
            
            # Use the average of the most loyal segments as a proxy for customer lifespan
            loyal_segments = ['Champions', 'Loyal Customers']
            loyal_customers = rfm[rfm['segment'].isin(loyal_segments)]
            
            if not loyal_customers.empty:
                # Estimate retention rate from the loyal segment frequency
                avg_loyal_frequency = loyal_customers['frequency'].mean()
                estimated_retention_rate = min(0.9, max(0.5, (avg_loyal_frequency - 1) / avg_loyal_frequency))
            else:
                # Default if we don't have loyal customers yet
                estimated_retention_rate = 0.7  # 70% retention rate
            
            # Calculate average customer lifespan: 1 / (1 - retention_rate)
            avg_customer_lifespan = 1 / (1 - estimated_retention_rate)
            
            # Calculate profit margin if we have profit data
            if 'profit' in df.columns:
                profit_margin = df['profit'].sum() / df['revenue'].sum()
            else:
                # Default profit margin if not available
                profit_margin = 0.3  # 30% profit margin
            
            # Calculate simple CLV
            simple_clv = avg_order_value * purchase_frequency * avg_customer_lifespan
            
            # Calculate profit-adjusted CLV
            profit_adjusted_clv = simple_clv * profit_margin
            
            # Calculate discount rate for present value (typically 10%)
            discount_rate = 0.1
            
            # Calculate discounted CLV
            discounted_clv = profit_adjusted_clv * (1 - discount_rate)
            
            # Display CLV metrics
            clv_col1, clv_col2, clv_col3 = st.columns(3)
            
            with clv_col1:
                st.metric("Average Order Value", f"£{avg_order_value:.2f}")
                st.metric("Purchase Frequency (per year)", f"{purchase_frequency:.2f}")
            
            with clv_col2:
                st.metric("Est. Customer Lifespan (years)", f"{avg_customer_lifespan:.2f}")
                st.metric("Est. Retention Rate", f"{estimated_retention_rate:.0%}")
            
            with clv_col3:
                st.metric("Average Customer Lifetime Value", f"£{simple_clv:.2f}")
                st.metric("Profit-Adjusted CLV", f"£{profit_adjusted_clv:.2f}")
            
            # Calculate CLV by segment
            segment_clv = rfm.copy()
            
            # Adjust CLV calculation by segment value
            segment_clv['retention_factor'] = segment_clv['segment_value'] / 5  # Normalize to 0-1 range
            segment_clv['segment_retention_rate'] = estimated_retention_rate * (0.5 + 0.5 * segment_clv['retention_factor'])
            segment_clv['segment_lifespan'] = 1 / (1 - segment_clv['segment_retention_rate'])
            segment_clv['segment_clv'] = avg_order_value * purchase_frequency * segment_clv['segment_lifespan'] * profit_margin * (1 - discount_rate)
            
            # Group by segment
            segment_clv_summary = segment_clv.groupby('segment').agg({
                'segment_clv': 'mean',
                'segment_retention_rate': 'mean',
                'segment_lifespan': 'mean',
                'customer_id': 'count'
            }).reset_index()
            
            segment_clv_summary = segment_clv_summary.sort_values('segment_clv', ascending=False)
            
            # Rename columns
            segment_clv_summary.columns = ['Segment', 'Average CLV', 'Est. Retention Rate', 'Est. Lifespan (years)', 'Customer Count']
            
            # Show CLV by segment
            st.subheader("Customer Lifetime Value by Segment")
            
            fig = px.bar(
                segment_clv_summary,
                x='Segment',
                y='Average CLV',
                color='Est. Retention Rate',
                text_auto=True,
                title='Estimated Customer Lifetime Value by Segment',
                labels={
                    'Segment': 'Customer Segment',
                    'Average CLV': 'Average Lifetime Value (£)',
                    'Est. Retention Rate': 'Estimated Retention Rate'
                },
                color_continuous_scale='Viridis'
            )
            
            # Format text labels
            fig.update_traces(
                texttemplate='£%{y:.2f}',
                textposition='outside'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display CLV segment data
            st.dataframe(
                segment_clv_summary,
                use_container_width=True,
                column_config={
                    'Segment': st.column_config.TextColumn('Segment'),
                    'Average CLV': st.column_config.NumberColumn(
                        'Average Lifetime Value',
                        format="£%.2f"
                    ),
                    'Est. Retention Rate': st.column_config.NumberColumn(
                        'Est. Retention Rate',
                        format="%.0f%%",
                        help="Estimated customer retention rate for this segment"
                    ),
                    'Est. Lifespan (years)': st.column_config.NumberColumn(
                        'Est. Lifespan (years)',
                        format="%.1f"
                    ),
                    'Customer Count': st.column_config.NumberColumn(
                        'Customer Count',
                        format="%d"
                    )
                }
            )
            
            # CLV insights
            st.markdown("""
            ### Key CLV Insights and Recommendations
            
            1. **Focus retention efforts on high-CLV segments**: The segments with the highest CLV represent your most valuable customers. Allocate more resources to retaining these customers.
            
            2. **Target acquisition of similar customers**: Use the characteristics of high-CLV segments to target similar prospects in acquisition campaigns.
            
            3. **Improve retention rates**: Even small improvements in retention rate can dramatically increase CLV. Implement loyalty programs, excellent customer service, and personalized communications.
            
            4. **Increase purchase frequency**: Create reasons for customers to purchase more often through cross-selling, subscriptions, or timely communications.
            
            5. **Increase average order value**: Use techniques like bundling, upselling, and minimum thresholds for free shipping or discounts.
            
            6. **Segment-specific strategies**:
               - For high-CLV segments: Invest in premium experiences and retention
               - For mid-CLV segments: Focus on increasing purchase frequency and basket size
               - For low-CLV segments: Evaluate acquisition costs against expected returns
            """)
        else:
            st.info("CLV analysis requires customer_id, date, and revenue data.")
    
    else:
        st.info("Detailed RFM analysis requires customer_id, date, and revenue data.")

# Regional analysis
if has_region_data:
    st.header("Regional Analysis")
    
    # Group by region
    # Create base aggregation for revenue
    agg_dict = {'revenue': 'sum'}
    
    # Add orders aggregation
    if 'transaction_id' in df.columns:
        # For transaction-based data, use transaction_id count
        region_trans = df.groupby('customer_region')['transaction_id'].nunique().reset_index()
        region_trans.rename(columns={'transaction_id': 'orders'}, inplace=True)
        
        # Create base metrics first
        region_metrics = df.groupby('customer_region').agg({
            'revenue': 'sum'
        }).reset_index()
        
        # Then merge with transaction counts
        region_metrics = pd.merge(region_metrics, region_trans, on='customer_region')
    elif 'orders' in df.columns:
        # For traditional data structure
        agg_dict['orders'] = 'sum'
        region_metrics = df.groupby('customer_region').agg(agg_dict).reset_index()
    else:
        # Fallback to counting rows as proxy for orders
        region_metrics = df.groupby('customer_region').agg({
            'revenue': 'sum'
        }).reset_index()
        # Count rows per region as proxy for orders
        region_count = df.groupby('customer_region').size().reset_index(name='orders')
        region_metrics = pd.merge(region_metrics, region_count, on='customer_region')
    
    # Add customer counts
    if has_customer_id:
        # Count unique customers if customer_id exists
        region_customers = df.groupby('customer_region')['customer_id'].nunique().reset_index()
        region_customers.rename(columns={'customer_id': 'customers'}, inplace=True)
        region_metrics = pd.merge(region_metrics, region_customers, on='customer_region')
    elif 'customers' in df.columns:
        # Sum customer counts if column exists
        region_customers = df.groupby('customer_region')['customers'].sum().reset_index()
        region_metrics = pd.merge(region_metrics, region_customers, on='customer_region')
    else:
        # Add customers column with same value as orders (proxy)
        region_metrics['customers'] = region_metrics['orders']
    
    if 'profit' in df.columns:
        region_profit = df.groupby('customer_region')['profit'].sum().reset_index()
        region_metrics = pd.merge(region_metrics, region_profit, on='customer_region')
        region_metrics['profit_margin'] = region_metrics['profit'] / region_metrics['revenue']
    
    # Calculate per customer metrics
    region_metrics['revenue_per_customer'] = region_metrics['revenue'] / region_metrics['customers']
    region_metrics['orders_per_customer'] = region_metrics['orders'] / region_metrics['customers']
    
    if 'profit' in region_metrics.columns:
        region_metrics['profit_per_customer'] = region_metrics['profit'] / region_metrics['customers']
    
    # Sort by revenue
    region_metrics = region_metrics.sort_values('revenue', ascending=False)
    
    # Region revenue chart
    st.subheader("Revenue by Region")
    
    fig = px.bar(
        region_metrics,
        x='customer_region',
        y='revenue',
        color='revenue_per_customer',
        title='Revenue by Region',
        labels={
            'customer_region': 'Region',
            'revenue': 'Revenue (£)',
            'revenue_per_customer': 'Revenue per Customer (£)'
        },
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Region comparison
    st.subheader("Region Comparison")
    
    # Select metrics to compare
    region_compare_metrics = ["Revenue per Customer", "Orders per Customer"]
    if 'profit' in region_metrics.columns:
        region_compare_metrics.append("Profit per Customer")
        region_compare_metrics.append("Profit Margin")
    
    selected_region_metrics = st.multiselect(
        "Select metrics to compare across regions:",
        region_compare_metrics,
        default=["Revenue per Customer"],
        key="region_metrics"
    )
    
    if not selected_region_metrics:
        st.info("Please select at least one metric to compare.")
    else:
        # Create a figure for each selected metric
        for metric in selected_region_metrics:
            # Map selected metric to column name
            if metric == "Revenue per Customer":
                column = "revenue_per_customer"
                format_str = "£{:,.2f}"
            elif metric == "Orders per Customer":
                column = "orders_per_customer"
                format_str = "{:.2f}"
            elif metric == "Profit per Customer":
                column = "profit_per_customer"
                format_str = "£{:,.2f}"
            elif metric == "Profit Margin":
                column = "profit_margin"
                format_str = "{:.2%}"
            
            # Create bar chart
            fig = px.bar(
                region_metrics,
                x='customer_region',
                y=column,
                color=column,
                title=f'{metric} by Region',
                labels={
                    'customer_region': 'Region',
                    column: metric
                },
                text_auto=True,
                color_continuous_scale='Viridis'
            )
            
            # Format the values
            fig.update_traces(
                texttemplate=format_str.format(0),
                textposition='outside'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Customer acquisition cost analysis
if 'customer_acquisition_cost' in df.columns:
    st.header("Customer Acquisition Cost (CAC) Analysis")
    
    # Calculate average CAC
    avg_cac = df['customer_acquisition_cost'].mean()
    
    # Calculate customer lifetime value if possible
    if has_customer_id:
        customer_revenue = df.groupby('customer_id')['revenue'].sum().mean()
        if 'profit' in df.columns:
            customer_profit = df.groupby('customer_id')['profit'].sum().mean()
            cltv = customer_profit
            cltv_metric = "Customer Lifetime Profit"
        else:
            cltv = customer_revenue
            cltv_metric = "Customer Lifetime Revenue"
    else:
        if total_customers != "N/A" and total_customers > 0:
            cltv = total_revenue / total_customers
            cltv_metric = "Average Customer Revenue"
        else:
            cltv = "N/A"
            cltv_metric = "Average Customer Revenue"
    
    # Calculate CAC to LTV ratio
    if cltv != "N/A" and avg_cac > 0:
        cltv_cac_ratio = cltv / avg_cac
    else:
        cltv_cac_ratio = "N/A"
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average CAC", f"£{avg_cac:.2f}")
    
    with col2:
        st.metric(cltv_metric, f"£{cltv:.2f}" if cltv != "N/A" else "N/A")
    
    with col3:
        st.metric("LTV:CAC Ratio", f"{cltv_cac_ratio:.2f}x" if cltv_cac_ratio != "N/A" else "N/A")
    
    # CAC over time if dates are available
    if 'date' in df.columns:
        st.subheader("CAC Trend Over Time")
        
        # Group by date
        df['date'] = pd.to_datetime(df['date'])
        cac_trend = df.groupby(df['date'].dt.strftime('%Y-%m')).agg({
            'customer_acquisition_cost': 'mean'
        }).reset_index()
        
        cac_trend.columns = ['month', 'avg_cac']
        cac_trend = cac_trend.sort_values('month')
        
        # Create trend chart
        fig = px.line(
            cac_trend,
            x='month',
            y='avg_cac',
            title='Average Customer Acquisition Cost Over Time',
            labels={
                'month': 'Month',
                'avg_cac': 'Average CAC (£)'
            },
            markers=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # CAC by segment if segment data is available
    if has_segment_data:
        st.subheader("CAC by Customer Segment")
        
        # Group by segment
        segment_cac = df.groupby('customer_segment')['customer_acquisition_cost'].mean().reset_index()
        segment_cac.columns = ['customer_segment', 'avg_cac']
        segment_cac = segment_cac.sort_values('avg_cac')
        
        # Create bar chart
        fig = px.bar(
            segment_cac,
            x='customer_segment',
            y='avg_cac',
            title='Average CAC by Customer Segment',
            labels={
                'customer_segment': 'Customer Segment',
                'avg_cac': 'Average CAC (£)'
            },
            text_auto=True,
            color='avg_cac',
            color_continuous_scale='RdYlGn_r'  # Reversed scale (lower CAC is better)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # CAC by region if region data is available
    if has_region_data:
        st.subheader("CAC by Region")
        
        # Group by region
        region_cac = df.groupby('customer_region')['customer_acquisition_cost'].mean().reset_index()
        region_cac.columns = ['customer_region', 'avg_cac']
        region_cac = region_cac.sort_values('avg_cac')
        
        # Create bar chart
        fig = px.bar(
            region_cac,
            x='customer_region',
            y='avg_cac',
            title='Average CAC by Region',
            labels={
                'customer_region': 'Region',
                'avg_cac': 'Average CAC (£)'
            },
            text_auto=True,
            color='avg_cac',
            color_continuous_scale='RdYlGn_r'  # Reversed scale (lower CAC is better)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # CAC to LTV Analysis
    st.subheader("CAC to LTV Analysis")
    
    st.markdown("""
    **Interpreting the CAC to LTV Ratio:**
    - **Less than 1:1** - Losing money on customers; need to reduce CAC or increase LTV
    - **1:1 to 3:1** - Sustainable but could be improved
    - **3:1 or higher** - Healthy ratio; good return on customer acquisition investment
    
    **Strategies to improve the ratio:**
    1. **Reduce CAC:**
       - Optimize marketing channels
       - Improve conversion rates
       - Enhance targeting to reach high-value prospects
    
    2. **Increase LTV:**
       - Improve customer retention
       - Implement upselling and cross-selling
       - Create loyalty programs
       - Enhance customer experience
    """)

# Customer retention analysis
if has_customer_id and 'date' in df.columns:
    st.header("Customer Retention Analysis")
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate cohorts
    df['cohort_month'] = df.groupby('customer_id')['date'].transform('min').dt.strftime('%Y-%m')
    
    # Create cohort analysis
    date_range = sorted(df['date'].dt.strftime('%Y-%m').unique())
    cohort_range = sorted(df['cohort_month'].unique())
    
    # Create a retention matrix
    retention_matrix = pd.DataFrame(index=cohort_range, columns=range(len(date_range)))
    
    # Fill the retention matrix
    for i, cohort in enumerate(cohort_range):
        # Get all customers in cohort
        cohort_customers = df[df['cohort_month'] == cohort]['customer_id'].unique()
        
        # Calculate retention for each month
        for j, month in enumerate(date_range):
            # Skip if cohort month is after current month
            if cohort > month:
                retention_matrix.loc[cohort, j] = np.nan
                continue
                
            # Calculate month number since cohort
            month_number = date_range.index(month) - date_range.index(cohort)
            
            # Get active customers in this month
            active_customers = df[df['date'].dt.strftime('%Y-%m') == month]['customer_id'].unique()
            
            # Calculate retention
            retention = np.intersect1d(cohort_customers, active_customers).size / cohort_customers.size
            
            # Set retention value
            retention_matrix.loc[cohort, month_number] = retention
    
    # Convert to percentage
    retention_matrix = retention_matrix * 100
    
    # Display retention heatmap
    st.subheader("Cohort Retention Heatmap")
    
    # Melt the matrix for plotting
    retention_melted = retention_matrix.reset_index().melt(id_vars=['index'], var_name='month_number', value_name='retention')
    retention_melted.columns = ['Cohort', 'Month Number', 'Retention']
    
    # Create heatmap
    fig = px.imshow(
        retention_matrix,
        labels=dict(x="Months Since First Purchase", y="Cohort Month", color="Retention %"),
        x=list(range(len(date_range))),
        y=cohort_range,
        color_continuous_scale="YlGnBu",
        aspect="auto",
        title="Customer Retention by Cohort"
    )
    
    fig.update_layout(
        xaxis_title="Months Since First Purchase",
        yaxis_title="Cohort Month"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate average retention by month
    avg_retention = retention_matrix.mean(axis=0).reset_index()
    avg_retention.columns = ['Month Number', 'Avg Retention']
    
    # Create line chart
    fig = px.line(
        avg_retention,
        x='Month Number',
        y='Avg Retention',
        title='Average Retention by Month',
        labels={
            'Month Number': 'Months Since First Purchase',
            'Avg Retention': 'Average Retention %'
        },
        markers=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Interpreting the Retention Curve:**
    - **Month 0** is always 100% as this is when customers first purchased
    - **The drop from Month 0 to Month 1** shows initial churn - how many customers don't come back after first purchase
    - **The curve flattening** indicates loyal customers who continue to purchase regularly
    
    **Strategies to improve retention:**
    1. Focus on the first few months to improve early retention
    2. Implement post-purchase email campaigns and follow-ups
    3. Create loyalty programs to reward repeat purchases
    4. Gather feedback from churned customers to identify improvement areas
    """)

# Export customer analysis
st.header("Export Data")

# Create downloadable data
if has_segment_data:
    segment_csv = segment_metrics.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Segment Analysis Data",
        data=segment_csv,
        file_name="customer_segment_analysis.csv",
        mime="text/csv"
    )

if has_region_data:
    region_csv = region_metrics.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Regional Analysis Data",
        data=region_csv,
        file_name="customer_region_analysis.csv",
        mime="text/csv"
    )
