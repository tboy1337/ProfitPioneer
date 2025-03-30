import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime
from utils.data_processor import load_data

# Page configuration
st.set_page_config(
    page_title="Data Import - E-Commerce Analytics",
    page_icon="📤",
    layout="wide"
)

# Initialize session state if not already done
if 'data' not in st.session_state:
    st.session_state.data = None

# Page header
st.title("Data Import")
st.markdown("Upload and manage your e-commerce data for analysis")

# Define required columns and their data types
REQUIRED_COLUMNS = ["date", "revenue"]
RECOMMENDED_COLUMNS = [
    "orders", 
    "customers", 
    "profit", 
    "product_id", 
    "product_name", 
    "category", 
    "customer_segment", 
    "customer_region", 
    "customer_acquisition_cost"
]

# Data upload section
st.header("Upload Data")

upload_method = st.radio(
    "Choose upload method:",
    ["Upload CSV File", "Use Sample Data"],
    horizontal=True
)

if upload_method == "Upload CSV File":
    st.markdown("""
    Upload your e-commerce data in CSV format. Your file should include at least the following columns:
    - `date`: Date of the transaction (YYYY-MM-DD format)
    - `revenue`: Revenue amount
    
    Recommended additional columns for richer analysis:
    - `orders`: Number of orders
    - `customers`: Number of customers
    - `profit`: Profit amount
    - `product_id` or `product_name`: Product identifier
    - `category`: Product category
    - `customer_segment`: Customer segment information
    - `customer_region`: Customer geographical region
    - `customer_acquisition_cost`: Cost to acquire customers
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the file
            df = pd.read_csv(uploaded_file)
            
            # Basic validation
            missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            
            if missing_required:
                st.error(f"Missing required columns: {', '.join(missing_required)}. Please ensure your data includes these columns.")
            else:
                # Convert date to datetime
                try:
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                except Exception as e:
                    st.error(f"Error processing date column: {str(e)}. Please ensure dates are in a standard format.")
                    st.stop()
                
                # Successfully loaded data
                st.session_state.data = df
                st.success(f"Data successfully loaded! {len(df)} rows and {len(df.columns)} columns imported.")
                
                # Show dataset preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Display data info
                st.subheader("Dataset Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Basic Statistics**")
                    st.write(f"- **Rows**: {len(df)}")
                    st.write(f"- **Columns**: {len(df.columns)}")
                    
                    if 'date' in df.columns:
                        min_date = pd.to_datetime(df['date']).min().strftime('%Y-%m-%d')
                        max_date = pd.to_datetime(df['date']).max().strftime('%Y-%m-%d')
                        date_range = (pd.to_datetime(max_date) - pd.to_datetime(min_date)).days
                        
                        st.write(f"- **Date Range**: {min_date} to {max_date} ({date_range} days)")
                
                with col2:
                    st.markdown("**Column Overview**")
                    
                    # List available columns
                    available_cols = df.columns.tolist()
                    
                    # Create a check list of available and missing columns
                    all_cols = set(REQUIRED_COLUMNS + RECOMMENDED_COLUMNS)
                    available_formatted = [f"✅ {col}" for col in available_cols if col in all_cols]
                    missing_formatted = [f"❌ {col}" for col in all_cols if col not in available_cols]
                    
                    # Display the list
                    for col in available_formatted:
                        st.write(col)
                    for col in missing_formatted:
                        st.write(col)
                
                # Analyze data quality
                st.subheader("Data Quality Analysis")
                
                # Check for missing values
                missing_values = df.isnull().sum()
                if missing_values.sum() > 0:
                    st.warning("Missing values detected in your dataset:")
                    missing_df = pd.DataFrame({
                        'Column': missing_values.index,
                        'Missing Values': missing_values.values,
                        'Percentage': (missing_values.values / len(df) * 100).round(2)
                    })
                    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("No missing values detected in your dataset.")
                
                # Check for negative revenue values
                if 'revenue' in df.columns and (df['revenue'] < 0).any():
                    st.warning(f"Your dataset contains {(df['revenue'] < 0).sum()} negative revenue values.")
                
                # Check for duplicate entries
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    st.warning(f"Your dataset contains {duplicates} duplicate rows.")
                else:
                    st.success("No duplicate rows detected.")
                
                # Provide data cleaning options
                st.subheader("Data Cleaning Options")
                
                clean_options = st.multiselect(
                    "Select data cleaning operations to perform:",
                    [
                        "Remove duplicate rows",
                        "Fill missing values with 0",
                        "Remove rows with missing values",
                        "Remove rows with negative revenue"
                    ]
                )
                
                if clean_options:
                    if st.button("Apply Data Cleaning"):
                        cleaned_df = df.copy()
                        
                        if "Remove duplicate rows" in clean_options:
                            original_rows = len(cleaned_df)
                            cleaned_df = cleaned_df.drop_duplicates()
                            st.write(f"Removed {original_rows - len(cleaned_df)} duplicate rows.")
                        
                        if "Fill missing values with 0" in clean_options:
                            cleaned_df = cleaned_df.fillna(0)
                            st.write("Filled all missing values with 0.")
                        
                        if "Remove rows with missing values" in clean_options:
                            original_rows = len(cleaned_df)
                            cleaned_df = cleaned_df.dropna()
                            st.write(f"Removed {original_rows - len(cleaned_df)} rows with missing values.")
                        
                        if "Remove rows with negative revenue" in clean_options and 'revenue' in cleaned_df.columns:
                            original_rows = len(cleaned_df)
                            cleaned_df = cleaned_df[cleaned_df['revenue'] >= 0]
                            st.write(f"Removed {original_rows - len(cleaned_df)} rows with negative revenue.")
                        
                        # Update the session state with cleaned data
                        st.session_state.data = cleaned_df
                        st.success(f"Data cleaning completed. Dataset now has {len(cleaned_df)} rows.")
        
        except Exception as e:
            st.error(f"Error loading the file: {str(e)}")

elif upload_method == "Use Sample Data":
    st.markdown("""
    Load sample e-commerce data to explore the dashboard features. 
    This is perfect for testing the dashboard before uploading your own data.
    """)
    
    if st.button("Load Sample Data", type="primary"):
        # Generate sample data
        try:
            # Generate dates for the past 90 days
            today = datetime.now()
            dates = [(today - pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(90)]
            
            # Create sample data with realistic e-commerce metrics
            sample_data = pd.DataFrame({
                'date': dates,
                'revenue': np.random.normal(1000, 200, 90) * np.linspace(1, 1.2, 90),  # Increasing trend
                'orders': np.random.randint(10, 50, 90),
                'customers': np.random.randint(8, 45, 90),
                'product_id': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], 90),
                'product_name': np.random.choice(['T-Shirt', 'Jeans', 'Sneakers', 'Hoodie', 'Hat'], 90),
                'category': np.random.choice(['Clothing', 'Footwear', 'Accessories'], 90),
                'cost': np.random.normal(400, 100, 90),
                'customer_acquisition_cost': np.random.normal(20, 5, 90),
                'customer_segment': np.random.choice(['New', 'Returning', 'Loyal'], 90),
                'customer_region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], 90),
            })
            
            # Calculate key metrics
            sample_data['profit'] = sample_data['revenue'] - sample_data['cost']
            sample_data['aov'] = sample_data['revenue'] / sample_data['orders']
            sample_data['conversion_rate'] = np.random.uniform(0.01, 0.05, 90)
            
            # Sort by date
            sample_data = sample_data.sort_values('date')
            
            # Store in session state
            st.session_state.data = sample_data
            st.success("Sample data successfully loaded!")
            
            # Show data preview
            st.subheader("Sample Data Preview")
            st.dataframe(sample_data.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating sample data: {str(e)}")

# Data management section
if st.session_state.data is not None:
    st.header("Manage Current Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Current Data"):
            st.session_state.data = None
            if 'filtered_data' in st.session_state:
                del st.session_state.filtered_data
            if 'forecast_df' in st.session_state:
                del st.session_state.forecast_df
                del st.session_state.forecast_metrics
                del st.session_state.forecast_generated
            st.success("Data cleared successfully. You can now upload new data.")
            st.experimental_rerun()
    
    with col2:
        # Download current data
        if st.session_state.data is not None:
            csv = st.session_state.data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Current Data",
                data=csv,
                file_name="ecommerce_data.csv",
                mime="text/csv"
            )

# Data transformation section
if st.session_state.data is not None:
    st.header("Data Transformation")
    
    with st.expander("Add Calculated Columns"):
        st.markdown("""
        Add useful calculated columns to enhance your analysis. 
        Select which calculated columns to add to your dataset:
        """)
        
        df = st.session_state.data
        
        transform_options = []
        
        if 'revenue' in df.columns and 'orders' in df.columns and 'aov' not in df.columns:
            transform_options.append("Average Order Value (AOV)")
        
        if 'revenue' in df.columns and 'cost' in df.columns and 'profit' not in df.columns:
            transform_options.append("Profit")
        
        if 'profit' in df.columns and 'revenue' in df.columns and 'profit_margin' not in df.columns:
            transform_options.append("Profit Margin")
        
        if not transform_options:
            st.info("No additional calculated columns available for your dataset.")
        else:
            selected_transforms = st.multiselect(
                "Select columns to add:",
                transform_options
            )
            
            if selected_transforms and st.button("Add Selected Columns"):
                transformed_df = df.copy()
                
                if "Average Order Value (AOV)" in selected_transforms:
                    transformed_df['aov'] = transformed_df['revenue'] / transformed_df['orders']
                    st.write("✅ Added Average Order Value (AOV) column")
                
                if "Profit" in selected_transforms:
                    transformed_df['profit'] = transformed_df['revenue'] - transformed_df['cost']
                    st.write("✅ Added Profit column")
                
                if "Profit Margin" in selected_transforms:
                    transformed_df['profit_margin'] = transformed_df['profit'] / transformed_df['revenue']
                    st.write("✅ Added Profit Margin column")
                
                # Update session state
                st.session_state.data = transformed_df
                st.success("Data transformation completed successfully.")

# CSV format guide
with st.expander("CSV Format Guide"):
    st.markdown("""
    ## CSV File Format Guidelines
    
    Your CSV file should follow these guidelines for optimal compatibility with the dashboard:
    
    ### Required Columns
    - `date`: Transaction date in YYYY-MM-DD format (e.g., 2023-06-15)
    - `revenue`: Revenue amount (numeric)
    
    ### Recommended Columns
    - `orders`: Number of orders (numeric)
    - `customers`: Number of customers (numeric)
    - `profit`: Profit amount (numeric)
    - `product_id`: Unique product identifier (string)
    - `product_name`: Product name (string)
    - `category`: Product category (string)
    - `cost`: Cost amount (numeric)
    - `customer_acquisition_cost`: Cost to acquire customers (numeric)
    - `customer_segment`: Customer segment information (string)
    - `customer_region`: Customer geographical region (string)
    - `customer_id`: Unique customer identifier (string)
    
    ### Sample CSV Format
    ```
    date,revenue,orders,customers,product_id,product_name,category,cost,profit,customer_segment,customer_region
    2023-06-01,1250.75,15,12,P001,T-Shirt,Clothing,625.30,625.45,New,North
    2023-06-01,876.50,8,7,P002,Jeans,Clothing,438.25,438.25,Returning,South
    2023-06-02,1500.25,12,10,P003,Sneakers,Footwear,750.10,750.15,Loyal,East
    ```
    
    ### Data Types
    - Date fields should be in YYYY-MM-DD format
    - Numeric fields should use period (.) as decimal separator
    - No currency symbols should be included in numeric fields
    - Text fields can contain spaces and special characters
    
    ### Best Practices
    - Ensure there are no blank rows at the beginning or end of the file
    - Remove any summary rows or totals from the data
    - Ensure column names match exactly (case-sensitive)
    - Export your data from spreadsheet software using the CSV format option
    """)

    # Show sample CSV template download
    sample_template = pd.DataFrame({
        'date': ['2023-06-01', '2023-06-01', '2023-06-02'],
        'revenue': [1250.75, 876.50, 1500.25],
        'orders': [15, 8, 12],
        'customers': [12, 7, 10],
        'product_id': ['P001', 'P002', 'P003'],
        'product_name': ['T-Shirt', 'Jeans', 'Sneakers'],
        'category': ['Clothing', 'Clothing', 'Footwear'],
        'cost': [625.30, 438.25, 750.10],
        'profit': [625.45, 438.25, 750.15],
        'customer_acquisition_cost': [15.20, 12.50, 18.75],
        'customer_segment': ['New', 'Returning', 'Loyal'],
        'customer_region': ['North', 'South', 'East']
    })
    
    csv_template = sample_template.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV Template",
        data=csv_template,
        file_name="ecommerce_template.csv",
        mime="text/csv"
    )

# Troubleshooting section
with st.expander("Troubleshooting"):
    st.markdown("""
    ## Common Data Import Issues
    
    ### File Format Problems
    - **Issue**: "Error loading the file" message
    - **Solution**: Ensure your file is a valid CSV format. Try opening it in a text editor to check for corruption.
    
    ### Missing Required Columns
    - **Issue**: "Missing required columns" error
    - **Solution**: Ensure your CSV has at least the 'date' and 'revenue' columns with exact spelling.
    
    ### Date Format Problems
    - **Issue**: "Error processing date column" message
    - **Solution**: Ensure dates are in a standard format like YYYY-MM-DD. You may need to reformat dates in your spreadsheet before exporting.
    
    ### Data Type Issues
    - **Issue**: Errors during analysis or unexpected results
    - **Solution**: Ensure numeric columns contain only numbers (no currency symbols or thousand separators).
    
    ### Large File Handling
    - **Issue**: Slow loading or browser crashes with very large files
    - **Solution**: Consider aggregating your data or using a smaller time range. Files under 50MB work best.
    
    ### Encoding Problems
    - **Issue**: Special characters appear incorrectly
    - **Solution**: Save your CSV with UTF-8 encoding.
    
    If you continue experiencing issues, try:
    1. Using the sample data to verify the dashboard works correctly
    2. Simplifying your dataset to identify problematic columns
    3. Checking for hidden columns or rows in your original spreadsheet
    """)

# Navigation help
st.markdown("""
---
## Next Steps

Now that you've imported your data, you can:

1. Go to the **Dashboard** to see an overview of your e-commerce performance
2. Use the **Filters** in the sidebar to focus on specific time periods or segments
3. Explore the **Sales History**, **Product Analysis**, and **Customer Segmentation** pages for detailed insights
4. Try the **Sales Forecast** feature to predict future performance

Use the navigation menu on the left to explore these features.
""")
