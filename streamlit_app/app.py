import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from preprocessing.data_preprocessor import DataPreprocessor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.cluster import KMeans
from streamlit_autorefresh import st_autorefresh

# Step 1: Establish a connection to the PostgreSQL database using SQLAlchemy 
# This is the most important part the postgresql needs to be running and correctly configured
# postgres is the username and password is password established and database is called name of the database.
# my defualt port is 5432 but it can be different
# again this is the most important part of the code
engine = create_engine('postgresql+psycopg2://username:password@host:port/database')

# Initialize the DataPreprocessor
preprocessor = DataPreprocessor(engine)

# Load and preprocess data
def load_and_preprocess_data(selected_vendor=None, selected_product=None):
    query = """
    SELECT pn.category_id, p.vendor_id, SUM(o.total_amount) as order_contribution
    FROM orders o
    JOIN groups_carts gc ON o.groups_carts_id = gc.id
    JOIN groups g ON gc.group_id = g.id
    JOIN group_deals gd ON g.group_deals_id = gd.id
    JOIN products p ON gd.product_id = p.id
    JOIN product_names pn ON p.name_id = pn.id
    WHERE 1=1
    """
    if selected_vendor and selected_vendor != "All":
        query += f" AND p.vendor_id = '{selected_vendor}'"
    if selected_product and selected_product != "All":
        query += f" AND pn.name = '{selected_product}'"
    query += " GROUP BY pn.category_id, p.vendor_id"
    
    df = pd.read_sql(query, engine)
    # Convert UUIDs to strings and shorten them
    df['category_id_short'] = df['category_id'].astype(str).str[:8]
    df['vendor_id_short'] = df['vendor_id'].astype(str).str[:8]
    df['category_id'] = df['category_id'].astype(str)
    df['vendor_id'] = df['vendor_id'].astype(str)
    # Use DataPreprocessor to handle nulls and encode categorical variables if needed
    df = preprocessor.handle_nulls(df, strategy='mean')
    return df

# Load vendor and product data for filtering
def load_vendor_product_data():
    query = """
    SELECT DISTINCT p.vendor_id, pn.name as product_name
    FROM products p
    JOIN product_names pn ON p.name_id = pn.id
    """
    df = pd.read_sql(query, engine)
    df['vendor_id'] = df['vendor_id'].astype(str)
    return df

# Preprocess data for the heatmap
def preprocess_data(df):
    pivot_table = df.pivot_table(index="category_id_short", columns="vendor_id_short", values="order_contribution", aggfunc='sum')
    return pivot_table

# Create a heatmap using Plotly
def create_heatmap(df, pivot_table):
    hover_text = []
    for i in range(len(pivot_table.index)):
        hover_text.append([])
        for j in range(len(pivot_table.columns)):
            category_id_full = df[df['category_id_short'] == pivot_table.index[i]]['category_id'].values[0]
            vendor_id_full = df[df['vendor_id_short'] == pivot_table.columns[j]]['vendor_id'].values[0]
            hover_text[-1].append(f"Category ID: {category_id_full}<br>Vendor ID: {vendor_id_full}<br>Order Contribution: {pivot_table.iloc[i, j]}")
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        text=hover_text,
        hoverinfo="text",
        colorscale="YlGnBu"
    ))
    fig.update_layout(title="Correlation between Product Categories and Order Contribution across Vendors",
                      xaxis_title="Vendor ID", yaxis_title="Category ID")
    st.plotly_chart(fig, use_container_width=True)

# Load and preprocess time-series data
def load_and_preprocess_timeseries_data(selected_vendor=None, selected_product=None):
    query = """
    SELECT o.created_at, SUM(o.total_amount) as total_amount
    FROM orders o
    JOIN groups_carts gc ON o.groups_carts_id = gc.id
    JOIN groups g ON gc.group_id = g.id
    JOIN group_deals gd ON g.group_deals_id = gd.id
    JOIN products p ON gd.product_id = p.id
    JOIN product_names pn ON p.name_id = pn.id
    WHERE 1=1
    """
    if selected_vendor and selected_vendor != "All":
        query += f" AND p.vendor_id = '{selected_vendor}'"
    if selected_product and selected_product != "All":
        query += f" AND pn.name = '{selected_product}'"
    query += " GROUP BY o.created_at ORDER BY o.created_at"
    
    df = pd.read_sql(query, engine)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df.set_index('created_at', inplace=True)
    df = df.resample('D').sum().fillna(0)  # Resample to daily frequency and fill missing values
    return df

# Fit ARIMA model and forecast
def forecast_arima(df):
    model = ARIMA(df['total_amount'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    return forecast

# Fit Prophet model and forecast
def forecast_prophet(df):
    df_prophet = df.reset_index().rename(columns={'created_at': 'ds', 'total_amount': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast

# Create a time-series plot using Plotly
def create_timeseries_plot(df, forecast, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['total_amount'], mode='lines', name='Historical'))
    if model_name == 'ARIMA':
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))
    elif model_name == 'Prophet':
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='red')))
    fig.update_layout(title=f"Order Trends Forecast using {model_name}",
                      xaxis_title="Date", yaxis_title="Total Amount")
    st.plotly_chart(fig, use_container_width=True)

# Load and preprocess data for grouped bar chart
def load_and_preprocess_grouped_bar_data():
    query = """
    SELECT
        CASE
            WHEN o.groups_carts_id IS NOT NULL THEN 'Group Deal'
            ELSE 'Individual Deal'
        END AS deal_type,
        COUNT(o.id) AS order_quantity
    FROM orders o
    WHERE o.status = 'COMPLETED'
    GROUP BY deal_type
    """
    df = pd.read_sql(query, engine)
    return df

# Create a grouped bar chart using Plotly
def create_grouped_bar_chart(df):
    fig = px.bar(df, x='deal_type', y='order_quantity', color='deal_type',
                 labels={'order_quantity': 'Order Quantity', 'deal_type': 'Deal Type'},
                 title='Order Quantities for Group Deals vs. Individual Deals',
                 log_y=True)  # Use logarithmic scale for y-axis
    st.plotly_chart(fig, use_container_width=True)

# Load and preprocess data for performance metrics
def load_and_preprocess_performance_data(selected_metrics):
    query = """
    SELECT pn.category_id::text as category_id, 
           SUM(o.total_amount) as revenue, 
           COUNT(DISTINCT gc.user_id) as user_retention, 
           COUNT(o.id) / COUNT(DISTINCT gc.user_id) as conversion_rate
    FROM orders o
    JOIN groups_carts gc ON o.groups_carts_id = gc.id
    JOIN groups g ON gc.group_id = g.id
    JOIN group_deals gd ON g.group_deals_id = gd.id
    JOIN products p ON gd.product_id = p.id
    JOIN product_names pn ON p.name_id = pn.id
    WHERE o.status = 'COMPLETED'
    GROUP BY pn.category_id
    """
    df = pd.read_sql(query, engine)
    df = df[selected_metrics + ['category_id']]
    return df

# Create a performance metrics chart using Plotly
def create_performance_metrics_chart(df, selected_metrics):
    fig = px.bar(df, x='category_id', y=selected_metrics, barmode='group',
                 title='Performance Metrics for Different Product Categories')
    st.plotly_chart(fig, use_container_width=True)

# Load and preprocess data for clustering
def load_and_preprocess_clustering_data():
    query = """
    SELECT gc.user_id::text as user_id, 
           SUM(o.total_amount) as total_spent, 
           COUNT(o.id) as order_count
    FROM orders o
    JOIN groups_carts gc ON o.groups_carts_id = gc.id
    WHERE o.status = 'COMPLETED'
    GROUP BY gc.user_id
    """
    df = pd.read_sql(query, engine)
    return df

# Perform K-Means clustering
def perform_clustering(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    df['cluster'] = kmeans.fit_predict(df[['total_spent', 'order_count']])
    return df

# Create a clustering chart using Plotly
def create_clustering_chart(df):
    fig = px.scatter(df, x='total_spent', y='order_count', color='cluster', 
                     title='User Segmentation based on Spending and Order Count',
                     labels={'total_spent': 'Total Spent', 'order_count': 'Order Count', 'cluster': 'Cluster'})
    st.plotly_chart(fig, use_container_width=True)

# Streamlit app
def main():
    st.title("ChipChip Dashboard")

    # Auto-refresh every 5 minutes
    st_autorefresh(interval=5 * 60 * 1000, key="data_refresh")

    # Load vendor and product data for filtering
    vendor_product_df = load_vendor_product_data()

    # Sidebar filters
    st.sidebar.header("Filters")
    vendor_options = ["All"] + vendor_product_df['vendor_id'].unique().tolist()
    selected_vendor = st.sidebar.selectbox("Select Vendor", vendor_options)

    if selected_vendor == "All":
        filtered_products = vendor_product_df['product_name'].unique()
    else:
        filtered_products = vendor_product_df[vendor_product_df['vendor_id'] == selected_vendor]['product_name'].unique()

    product_options = ["All"] + filtered_products.tolist()
    selected_product = st.sidebar.selectbox("Select Product", product_options)

    # Multi-select widget for performance metrics
    st.sidebar.header("Performance Metrics")
    metrics_options = ["revenue", "conversion_rate", "user_retention"]
    selected_metrics = st.sidebar.multiselect("Select Metrics", metrics_options, default=metrics_options)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Heatmap", "Time-Series Forecast", "Grouped Bar Chart", "Performance Metrics", "User Segmentation"])

    with tab1:
        st.header("Dynamic Heatmap")
        df = load_and_preprocess_data(selected_vendor, selected_product)
        pivot_table = preprocess_data(df)
        create_heatmap(df, pivot_table)

    with tab2:
        st.header("Time-Series Forecast")
        df_ts = load_and_preprocess_timeseries_data(selected_vendor, selected_product)
        model_option = st.selectbox("Select Model", ["ARIMA", "Prophet"])
        if model_option == "ARIMA":
            forecast = forecast_arima(df_ts)
            create_timeseries_plot(df_ts, forecast, "ARIMA")
        elif model_option == "Prophet":
            forecast = forecast_prophet(df_ts)
            create_timeseries_plot(df_ts, forecast, "Prophet")

    with tab3:
        st.header("Grouped Bar Chart")
        df_bar = load_and_preprocess_grouped_bar_data()
        create_grouped_bar_chart(df_bar)

    with tab4:
        st.header("Performance Metrics")
        df_perf = load_and_preprocess_performance_data(selected_metrics)
        create_performance_metrics_chart(df_perf, selected_metrics)

    with tab5:
        st.header("User Segmentation")
        n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
        df_clustering = load_and_preprocess_clustering_data()
        df_clustered = perform_clustering(df_clustering, n_clusters)
        create_clustering_chart(df_clustered)

if __name__ == "__main__":
    main()