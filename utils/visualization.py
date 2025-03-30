import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_kpi_card(title, value, change=None):
    """
    Create a KPI card with title, value, and change indicator
    
    Args:
        title: KPI title
        value: KPI value
        change: Percentage change (positive = good, negative = bad)
    """
    # Format the title and value
    with st.container():
        st.markdown(f"### {title}")
        st.markdown(f"<h2 style='margin:0;padding:0'>{value}</h2>", unsafe_allow_html=True)
        
        if change is not None:
            color = "green" if change >= 0 else "red"
            icon = "↑" if change >= 0 else "↓"
            st.markdown(f"<span style='color:{color};font-size:1.2em'>{icon} {abs(change):.1%}</span>", unsafe_allow_html=True)

def create_trend_chart(df, metrics=['revenue', 'profit'], title='Revenue and Profit Trends'):
    """
    Create a time series trend chart
    
    Args:
        df: pandas DataFrame with time series data
        metrics: List of metrics to display
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Check if we're using the 'period' column from group_data_by_period function
    # or the 'date' column directly
    date_col = 'period' if 'period' in df.columns else 'date'
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    if 'revenue' in metrics and 'revenue' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df['revenue'],
                name="Revenue",
                line=dict(color='#1F77B4', width=3),
                mode='lines'
            ),
            secondary_y=False,
        )
    
    if 'profit' in metrics and 'profit' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df['profit'],
                name="Profit",
                line=dict(color='#2CA02C', width=3),
                mode='lines'
            ),
            secondary_y=True,
        )
    
    if 'orders' in metrics and 'orders' in df.columns:
        fig.add_trace(
            go.Bar(
                x=df[date_col],
                y=df['orders'],
                name="Orders",
                marker_color='#FF7F0E',
                opacity=0.7
            ),
            secondary_y=False,
        )
    
    # Set titles
    fig.update_layout(
        title=title,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Revenue", secondary_y=False)
    if 'profit' in metrics and 'profit' in df.columns:
        fig.update_yaxes(title_text="Profit", secondary_y=True)
    
    return fig

def create_product_comparison_chart(df, x='product_name', y='revenue', color='profit_margin', title='Product Comparison'):
    """
    Create a product comparison bar chart
    
    Args:
        df: pandas DataFrame with product data
        x: X-axis column (typically product name)
        y: Y-axis column (metric to compare)
        color: Color column (another metric to show via color)
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Sort dataframe by the y metric (descending)
    df = df.sort_values(y, ascending=False)
    
    # Create the figure
    fig = px.bar(
        df.head(15),  # Show top 15 products
        x=x,
        y=y,
        color=color,
        color_continuous_scale='RdYlGn',
        text=y,
        title=title,
        labels={
            x: x.replace('_', ' ').title(),
            y: y.replace('_', ' ').title(),
            color: color.replace('_', ' ').title()
        }
    )
    
    # Format the text on bars to show values
    fig.update_traces(
        texttemplate='%{text:.2s}',
        textposition='outside'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x.replace('_', ' ').title(),
        yaxis_title=y.replace('_', ' ').title(),
        coloraxis_colorbar_title=color.replace('_', ' ').title(),
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig

def create_customer_segment_chart(df, values='revenue', names='customer_segment', title='Revenue by Customer Segment'):
    """
    Create a pie chart for customer segmentation
    
    Args:
        df: pandas DataFrame with customer segment data
        values: Values column for the pie chart
        names: Names column for pie chart slices
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Create the figure
    fig = px.pie(
        df,
        values=values,
        names=names,
        title=title,
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Update layout
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent+value'
    )
    
    return fig

def create_region_map_chart(df, location_col='customer_region', values_col='revenue', title='Revenue by Region'):
    """
    Create a choropleth map chart for regional data
    
    Args:
        df: pandas DataFrame with regional data
        location_col: Column with location names
        values_col: Column with values for coloring
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Create the figure
    fig = px.choropleth(
        df,
        locations=location_col,
        color=values_col,
        title=title,
        color_continuous_scale='Viridis',
        locationmode='country names',
        labels={
            values_col: values_col.replace('_', ' ').title()
        }
    )
    
    # Update layout
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        )
    )
    
    return fig

def create_heatmap(df, title='Correlation Heatmap'):
    """
    Create a correlation heatmap of numeric columns
    
    Args:
        df: pandas DataFrame
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create the heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title=title
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
    )
    
    return fig

def create_pareto_chart(df, category_col, value_col, title='Pareto Analysis'):
    """
    Create a Pareto chart (combines bar chart and cumulative line)
    
    Args:
        df: pandas DataFrame
        category_col: Column with categories
        value_col: Column with values
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Sort DataFrame by value column (descending)
    df_sorted = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    
    # Calculate cumulative percentage
    total = df_sorted[value_col].sum()
    df_sorted['cumulative'] = df_sorted[value_col].cumsum()
    df_sorted['cumulative_percent'] = df_sorted['cumulative'] / total * 100
    
    # Create figure with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=df_sorted[category_col],
            y=df_sorted[value_col],
            name=value_col,
            marker_color='#1F77B4'
        ),
        secondary_y=False
    )
    
    # Add cumulative line
    fig.add_trace(
        go.Scatter(
            x=df_sorted[category_col],
            y=df_sorted['cumulative_percent'],
            name='Cumulative %',
            marker_color='#FF7F0E',
            line=dict(width=3)
        ),
        secondary_y=True
    )
    
    # Add 80% line
    fig.add_shape(
        type='line',
        x0=-0.5,
        y0=80,
        x1=len(df_sorted) - 0.5,
        y1=80,
        line=dict(
            color='red',
            width=2,
            dash='dash',
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=category_col.replace('_', ' ').title(),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_yaxes(title_text=value_col.replace('_', ' ').title(), secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 105])
    
    return fig

def create_scatter_plot(df, x, y, color=None, size=None, title='Scatter Plot'):
    """
    Create a scatter plot to show relationships between variables
    
    Args:
        df: pandas DataFrame
        x: X-axis column
        y: Y-axis column
        color: Column for point color
        size: Column for point size
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Create the figure
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=size,
        hover_name=df.index,
        title=title,
        labels={
            x: x.replace('_', ' ').title(),
            y: y.replace('_', ' ').title(),
            color: color.replace('_', ' ').title() if color else None,
            size: size.replace('_', ' ').title() if size else None
        }
    )
    
    # Add trendline
    fig.update_layout(
        xaxis_title=x.replace('_', ' ').title(),
        yaxis_title=y.replace('_', ' ').title(),
    )
    
    return fig
