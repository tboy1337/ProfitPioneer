import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processor import calculate_product_metrics, analyze_market_basket
from utils.visualization import create_product_comparison_chart, create_pareto_chart, create_heatmap

# Page configuration
st.set_page_config(
    page_title="Product Analysis - E-Commerce Analytics",
    page_icon="📦",
    layout="wide"
)

# Initialize session state if not already done
if 'data' not in st.session_state:
    st.session_state.data = None

# Page header
st.title("Product Performance Analysis")
st.markdown("Analyze your product catalog performance, identify top performers, and optimize your product strategy")

# Check if data is loaded
if st.session_state.data is None:
    st.warning("No data loaded. Please go to the Data Import page or load sample data from the Dashboard.")
    st.stop()

# Check if product data is available
if 'product_id' not in st.session_state.data.columns and 'product_name' not in st.session_state.data.columns:
    st.error("Product data not available. The dataset must contain 'product_id' or 'product_name' columns.")
    st.stop()

# Get filtered data from session state
if 'filtered_data' in st.session_state:
    df = st.session_state.filtered_data
else:
    df = st.session_state.data

# Calculate product metrics
product_metrics = calculate_product_metrics(df)

# Product overview
st.header("Product Overview")

# Key metrics summary
total_products = len(product_metrics)
total_revenue = product_metrics['revenue'].sum()
avg_revenue_per_product = total_revenue / total_products if total_products > 0 else 0

if 'profit' in product_metrics.columns:
    total_profit = product_metrics['profit'].sum()
    avg_profit_margin = total_profit / total_revenue if total_revenue > 0 else 0
else:
    total_profit = 0
    avg_profit_margin = 0

# Display metrics in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Products", f"{total_products:,}")
    
with col2:
    st.metric("Total Revenue", f"£{total_revenue:,.2f}")
    
with col3:
    st.metric("Avg. Revenue per Product", f"£{avg_revenue_per_product:,.2f}")
    
with col4:
    st.metric("Avg. Profit Margin", f"{avg_profit_margin:.2%}")

# Top products analysis
st.header("Top Products Analysis")

# Select metric for analysis
metric_options = ["Revenue", "Orders"]
if 'profit' in product_metrics.columns:
    metric_options.append("Profit")
    metric_options.append("Profit Margin")

selected_metric = st.selectbox(
    "Select metric for product ranking:",
    metric_options,
    index=0
)

# Convert to lowercase for column names
selected_metric_lower = selected_metric.lower()
if selected_metric == "Profit Margin":
    selected_metric_lower = "profit_margin"

# Number of top products to show
top_n = st.slider("Number of products to show:", 5, min(50, len(product_metrics)), 10)

# Sort products by selected metric
sorted_products = product_metrics.sort_values(selected_metric_lower, ascending=False).head(top_n)

# Create the bar chart
fig = create_product_comparison_chart(
    sorted_products,
    x='product_name' if 'product_name' in sorted_products.columns else 'product_id',
    y=selected_metric_lower,
    color='profit_margin' if 'profit_margin' in sorted_products.columns else None,
    title=f'Top {top_n} Products by {selected_metric}'
)

st.plotly_chart(fig, use_container_width=True)

# Pareto analysis (80/20 rule)
st.header("Pareto Analysis (80/20 Rule)")
st.markdown("""
The Pareto principle suggests that approximately 80% of effects come from 20% of causes.
In e-commerce, this often means that a small percentage of products generate most of the revenue.
""")

# Select metric for Pareto analysis
pareto_metric = st.selectbox(
    "Select metric for Pareto analysis:",
    ["Revenue", "Profit"] if 'profit' in product_metrics.columns else ["Revenue"],
    index=0,
    key="pareto_metric"
)

# Convert to lowercase for column names
pareto_metric_lower = pareto_metric.lower()

# Create Pareto chart
product_id_col = 'product_name' if 'product_name' in product_metrics.columns else 'product_id'
pareto_fig = create_pareto_chart(
    product_metrics,
    category_col=product_id_col,
    value_col=pareto_metric_lower,
    title=f'Pareto Analysis: {pareto_metric} by Product'
)

st.plotly_chart(pareto_fig, use_container_width=True)

# Calculate Pareto metrics
product_metrics_sorted = product_metrics.sort_values(pareto_metric_lower, ascending=False).reset_index(drop=True)
product_metrics_sorted['cumulative'] = product_metrics_sorted[pareto_metric_lower].cumsum()
product_metrics_sorted['cumulative_percent'] = product_metrics_sorted['cumulative'] / product_metrics_sorted[pareto_metric_lower].sum() * 100

# Find how many products make up 80% of the metric
products_80_percent = product_metrics_sorted[product_metrics_sorted['cumulative_percent'] <= 80].shape[0]
if products_80_percent == 0:
    products_80_percent = 1  # At least one product

percent_of_products = (products_80_percent / len(product_metrics_sorted)) * 100

st.markdown(f"""
**Pareto Analysis Results:**
- **{products_80_percent}** products ({percent_of_products:.1f}% of your catalog) generate 80% of your {pareto_metric.lower()}
- This confirms the Pareto principle (80/20 rule) in your product performance
""")

# Product quadrant analysis
if 'profit_margin' in product_metrics.columns and 'revenue' in product_metrics.columns:
    st.header("Product Portfolio Analysis")
    st.markdown("""
    This quadrant analysis plots products based on revenue and profit margin to help identify:
    - **Stars**: High revenue, high profit margin
    - **Cash Cows**: High revenue, lower profit margin
    - **Question Marks**: Low revenue, high profit margin
    - **Dogs**: Low revenue, low profit margin
    """)
    
    # Calculate medians for quadrant divisions
    revenue_median = product_metrics['revenue'].median()
    profit_margin_median = product_metrics['profit_margin'].median()
    
    # Create scatter plot
    fig = px.scatter(
        product_metrics,
        x='revenue',
        y='profit_margin',
        color='profit_margin',
        size='revenue',
        hover_name='product_name' if 'product_name' in product_metrics.columns else 'product_id',
        title='Product Portfolio Analysis: Revenue vs. Profit Margin',
        color_continuous_scale='RdYlGn',
        labels={
            'revenue': 'Revenue (£)',
            'profit_margin': 'Profit Margin'
        }
    )
    
    # Add quadrant lines
    fig.add_shape(
        type="line",
        x0=revenue_median,
        y0=0,
        x1=revenue_median,
        y1=max(product_metrics['profit_margin']) * 1.1,
        line=dict(
            color="rgba(0,0,0,0.5)",
            width=1,
            dash="dash",
        )
    )
    
    fig.add_shape(
        type="line",
        x0=0,
        y0=profit_margin_median,
        x1=max(product_metrics['revenue']) * 1.1,
        y1=profit_margin_median,
        line=dict(
            color="rgba(0,0,0,0.5)",
            width=1,
            dash="dash",
        )
    )
    
    # Add quadrant labels
    fig.add_annotation(
        x=revenue_median / 2,
        y=profit_margin_median / 2,
        text="Dogs",
        showarrow=False,
        font=dict(
            size=14,
            color="rgba(0,0,0,0.5)"
        )
    )
    
    fig.add_annotation(
        x=revenue_median * 1.5,
        y=profit_margin_median / 2,
        text="Cash Cows",
        showarrow=False,
        font=dict(
            size=14,
            color="rgba(0,0,0,0.5)"
        )
    )
    
    fig.add_annotation(
        x=revenue_median / 2,
        y=profit_margin_median * 1.5,
        text="Question Marks",
        showarrow=False,
        font=dict(
            size=14,
            color="rgba(0,0,0,0.5)"
        )
    )
    
    fig.add_annotation(
        x=revenue_median * 1.5,
        y=profit_margin_median * 1.5,
        text="Stars",
        showarrow=False,
        font=dict(
            size=14,
            color="rgba(0,0,0,0.5)"
        )
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Revenue (£)",
        yaxis_title="Profit Margin",
        yaxis_tickformat='.1%',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyze quadrants
    stars = product_metrics[(product_metrics['revenue'] > revenue_median) & 
                           (product_metrics['profit_margin'] > profit_margin_median)]
    
    cash_cows = product_metrics[(product_metrics['revenue'] > revenue_median) & 
                               (product_metrics['profit_margin'] <= profit_margin_median)]
    
    question_marks = product_metrics[(product_metrics['revenue'] <= revenue_median) & 
                                    (product_metrics['profit_margin'] > profit_margin_median)]
    
    dogs = product_metrics[(product_metrics['revenue'] <= revenue_median) & 
                          (product_metrics['profit_margin'] <= profit_margin_median)]
    
    # Display quadrant statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("Stars")
        st.metric("Count", f"{len(stars)}")
        st.metric("% of Revenue", f"{stars['revenue'].sum() / total_revenue:.1%}")
        st.metric("Avg. Profit Margin", f"{stars['profit_margin'].mean():.1%}")
    
    with col2:
        st.subheader("Cash Cows")
        st.metric("Count", f"{len(cash_cows)}")
        st.metric("% of Revenue", f"{cash_cows['revenue'].sum() / total_revenue:.1%}")
        st.metric("Avg. Profit Margin", f"{cash_cows['profit_margin'].mean():.1%}")
    
    with col3:
        st.subheader("Question Marks")
        st.metric("Count", f"{len(question_marks)}")
        st.metric("% of Revenue", f"{question_marks['revenue'].sum() / total_revenue:.1%}")
        st.metric("Avg. Profit Margin", f"{question_marks['profit_margin'].mean():.1%}")
    
    with col4:
        st.subheader("Dogs")
        st.metric("Count", f"{len(dogs)}")
        st.metric("% of Revenue", f"{dogs['revenue'].sum() / total_revenue:.1%}")
        st.metric("Avg. Profit Margin", f"{dogs['profit_margin'].mean():.1%}")
    
    # Recommendations based on quadrant analysis
    st.subheader("Product Strategy Recommendations")
    
    st.markdown("""
    Based on the quadrant analysis, consider the following strategies:
    
    **Stars:**
    - Maintain or increase inventory
    - Consider premium pricing
    - Invest in marketing these products
    - Protect market share
    
    **Cash Cows:**
    - Optimize costs to improve margins
    - Maintain market share
    - Use revenue to fund growth elsewhere
    - Minimize unnecessary investment
    
    **Question Marks:**
    - Evaluate potential for scaling
    - Test marketing strategies to increase volume
    - Consider bundling with popular products
    - Analyze if they can become stars
    
    **Dogs:**
    - Consider phasing out or replacing
    - Minimize inventory investment
    - Look for cost reduction opportunities
    - Evaluate if they serve a strategic purpose
    """)

# Product categories analysis
if 'category' in df.columns:
    st.header("Product Categories Analysis")
    
    # Group by category - first create base metrics
    category_metrics = df.groupby('category').agg({
        'revenue': 'sum'
    }).reset_index()
    
    # Add orders - check for transaction_id first
    if 'transaction_id' in df.columns:
        # Count unique transactions per category
        cat_trans = df.groupby('category')['transaction_id'].nunique().reset_index()
        cat_trans.rename(columns={'transaction_id': 'orders'}, inplace=True)
        # Merge with base metrics
        category_metrics = pd.merge(category_metrics, cat_trans, on='category')
    elif 'orders' in df.columns:
        # Use orders column if available
        cat_orders = df.groupby('category')['orders'].sum().reset_index()
        category_metrics = pd.merge(category_metrics, cat_orders, on='category')
    else:
        # Fallback to counting rows as proxy for orders
        cat_count = df.groupby('category').size().reset_index(name='orders')
        category_metrics = pd.merge(category_metrics, cat_count, on='category')
    
    # Add profit if available
    if 'profit' in df.columns:
        category_profit = df.groupby('category')['profit'].sum().reset_index()
        category_metrics = pd.merge(category_metrics, category_profit, on='category')
        category_metrics['profit_margin'] = category_metrics['profit'] / category_metrics['revenue']
    
    # Sorting and percentage calculation
    total_category_revenue = category_metrics['revenue'].sum()
    category_metrics['revenue_percent'] = category_metrics['revenue'] / total_category_revenue
    category_metrics = category_metrics.sort_values('revenue', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        category_metrics,
        x='category',
        y='revenue',
        color='profit_margin' if 'profit_margin' in category_metrics.columns else None,
        color_continuous_scale='RdYlGn',
        text='revenue_percent',
        title='Revenue by Product Category',
        labels={
            'category': 'Category',
            'revenue': 'Revenue (£)',
            'profit_margin': 'Profit Margin',
            'revenue_percent': 'Percentage of Total'
        }
    )
    
    # Format percentage labels
    fig.update_traces(
        texttemplate='%{text:.1%}',
        textposition='outside'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show average metrics by category
    st.subheader("Average Metrics by Category")
    
    # Calculate average metrics per product in each category
    if 'product_name' in df.columns or 'product_id' in df.columns:
        id_column = 'product_name' if 'product_name' in df.columns else 'product_id'
        
        # Calculate base metrics first
        category_avg = df.groupby('category').agg({
            id_column: 'nunique',
            'revenue': 'sum'
        }).reset_index()
        
        # Add orders - check for transaction_id first
        if 'transaction_id' in df.columns:
            # Count unique transactions per category
            cat_avg_trans = df.groupby('category')['transaction_id'].nunique().reset_index()
            cat_avg_trans.rename(columns={'transaction_id': 'orders'}, inplace=True)
            # Merge with base metrics
            category_avg = pd.merge(category_avg, cat_avg_trans, on='category')
        elif 'orders' in df.columns:
            # Use orders column if available
            cat_avg_orders = df.groupby('category')['orders'].sum().reset_index()
            category_avg = pd.merge(category_avg, cat_avg_orders, on='category')
        else:
            # Fallback to counting rows as proxy for orders
            cat_avg_count = df.groupby('category').size().reset_index(name='orders')
            category_avg = pd.merge(category_avg, cat_avg_count, on='category')
        
        if 'profit' in df.columns:
            category_avg_profit = df.groupby('category')['profit'].sum().reset_index()
            category_avg = pd.merge(category_avg, category_avg_profit, on='category')
        
        # Calculate averages
        category_avg['avg_revenue_per_product'] = category_avg['revenue'] / category_avg[id_column]
        category_avg['avg_orders_per_product'] = category_avg['orders'] / category_avg[id_column]
        
        if 'profit' in category_avg.columns:
            category_avg['avg_profit_per_product'] = category_avg['profit'] / category_avg[id_column]
            category_avg['profit_margin'] = category_avg['profit'] / category_avg['revenue']
        
        # Sort by average revenue per product
        category_avg = category_avg.sort_values('avg_revenue_per_product', ascending=False)
        
        # Display table
        st.dataframe(
            category_avg[[
                'category', 
                id_column, 
                'avg_revenue_per_product', 
                'avg_orders_per_product',
                'avg_profit_per_product' if 'profit' in category_avg.columns else None,
                'profit_margin' if 'profit' in category_avg.columns else None
            ]].dropna(axis=1),
            use_container_width=True,
            column_config={
                'category': 'Category',
                id_column: '# of Products',
                'avg_revenue_per_product': st.column_config.NumberColumn(
                    'Avg. Revenue per Product',
                    format="£%.2f"
                ),
                'avg_orders_per_product': 'Avg. Orders per Product',
                'avg_profit_per_product': st.column_config.NumberColumn(
                    'Avg. Profit per Product',
                    format="£%.2f"
                ) if 'profit' in category_avg.columns else None,
                'profit_margin': st.column_config.NumberColumn(
                    'Profit Margin',
                    format="%.1f%%"
                ) if 'profit' in category_avg.columns else None
            }
        )

# Market Basket Analysis
st.header("Market Basket Analysis")
st.markdown("""
This analysis helps identify products that are often purchased together, 
allowing you to create effective bundles, optimize recommendations, 
and implement cross-selling strategies.
""")

# Run market basket analysis
market_basket_results = analyze_market_basket(df)

if not market_basket_results.empty:
    # Display the number of product associations found
    st.info(f"Found {len(market_basket_results)} potential product associations.")
    
    # Display the top product associations
    st.subheader("Top Product Associations")
    
    # Create a table showing top product associations
    st.dataframe(
        market_basket_results.head(10),
        use_container_width=True,
        column_config={
            'product_a': 'Product A',
            'product_b': 'Product B',
            'co_occurrences': 'Co-occurrences',
            'lift': st.column_config.NumberColumn('Lift', format="%.2f"),
            'relevance_score': st.column_config.NumberColumn('Relevance Score', format="%.2f")
        }
    )
    
    # Product association recommendations
    st.subheader("Product Bundle Recommendations")
    
    # Generate specific recommendations based on the analysis
    if len(market_basket_results) >= 5:
        top_5_associations = market_basket_results.head(5)
        
        for i, row in top_5_associations.iterrows():
            product_a = row['product_a']
            product_b = row['product_b']
            lift = row.get('lift', 1.0)
            
            bundle_name = f"Bundle {i+1}: {product_a} + {product_b}"
            bundle_description = f"These products are {lift:.1f}x more likely to be purchased together"
            
            st.markdown(f"**{bundle_name}**")
            st.markdown(bundle_description)
            
            if i < len(top_5_associations) - 1:
                st.markdown("---")
    
    # Strategic recommendations
    with st.expander("Cross-Selling Strategy Recommendations", expanded=True):
        st.markdown("""
        ### How to Use These Insights
        
        1. **Create Product Bundles**
           - Bundle frequently co-purchased items with a small discount
           - Feature these bundles prominently on product pages
           - Create special bundle packaging for holiday seasons
        
        2. **Optimize Store Layout**
           - Place complementary products near each other
           - Create promotional displays featuring product combinations
           - Use signage to suggest complementary purchases
        
        3. **Enhance Marketing Campaigns**
           - Target customers who bought one product with ads for complementary items
           - Create "complete the set" email campaigns
           - Develop content showcasing how products work together
        
        4. **Improve Recommendation Engine**
           - Implement "Frequently Bought Together" widgets on product pages
           - Add "You Might Also Like" sections to the shopping cart
           - Personalize recommendations based on past purchase history
        """)
else:
    st.warning("""
    Not enough data to perform market basket analysis. This analysis requires:
    - Product purchase data
    - Multiple products per order or transaction
    - Sufficient historical data
    """)
    
    with st.expander("Tips for Collecting Better Data"):
        st.markdown("""
        ### Improving Your Data for Market Basket Analysis
        
        1. **Track Order IDs**
           - Ensure each transaction has a unique identifier
           - Record all products purchased in each transaction
        
        2. **Collect More Transaction Data**
           - The more historical data, the better the analysis
           - Aim for at least several hundred transactions
        
        3. **Standardize Product Names**
           - Use consistent naming conventions
           - Group variations of the same product
        """)

# Full product list
st.header("Complete Product List")

# Sort options for the table
sort_options = ["Revenue (High to Low)", "Revenue (Low to High)"]
if 'profit_margin' in product_metrics.columns:
    sort_options.extend(["Profit Margin (High to Low)", "Profit Margin (Low to High)"])

sort_by = st.selectbox("Sort by:", sort_options)

# Apply sorting
if sort_by == "Revenue (High to Low)":
    product_metrics = product_metrics.sort_values('revenue', ascending=False)
elif sort_by == "Revenue (Low to High)":
    product_metrics = product_metrics.sort_values('revenue', ascending=True)
elif sort_by == "Profit Margin (High to Low)":
    product_metrics = product_metrics.sort_values('profit_margin', ascending=False)
elif sort_by == "Profit Margin (Low to High)":
    product_metrics = product_metrics.sort_values('profit_margin', ascending=True)

# Display the full product metrics table
st.dataframe(
    product_metrics,
    use_container_width=True,
    column_config={
        'product_name' if 'product_name' in product_metrics.columns else 'product_id': 'Product',
        'revenue': st.column_config.NumberColumn(
            'Revenue',
            format="£%.2f"
        ),
        'orders': 'Orders',
        'aov': st.column_config.NumberColumn(
            'AOV',
            format="£%.2f"
        ),
        'profit': st.column_config.NumberColumn(
            'Profit',
            format="£%.2f"
        ) if 'profit' in product_metrics.columns else None,
        'profit_margin': st.column_config.NumberColumn(
            'Profit Margin',
            format="%.2f%%"
        ) if 'profit_margin' in product_metrics.columns else None,
        'revenue_contribution': st.column_config.NumberColumn(
            'Revenue Contribution',
            format="%.2f%%"
        )
    }
)

# Download product analysis data
st.download_button(
    label="Download Product Analysis Data",
    data=product_metrics.to_csv(index=False).encode('utf-8'),
    file_name="product_analysis.csv",
    mime="text/csv"
)
