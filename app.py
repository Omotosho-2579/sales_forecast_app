import os
import sys
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Handle environment compatibility
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['MPLCONFIGDIR'] = os.getcwd()

# Workaround for Streamlit/PyTorch compatibility
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from forecasting_model import *

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stFileUploader>div>div>div>button {background-color: #2196F3; color: white;}
    .css-1aumxhk {background-color: #ffffff; padding: 20px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# App title
st.title("📊 Sales Forecasting with LSTM")
st.markdown("Predict future sales using PyTorch's LSTM neural networks")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar controls
with st.sidebar:
    st.header("Data Input")
    data_option = st.radio(
        "Select data source:",
        ("Use sample data", "Upload your CSV"),
        index=0
    )
    
    if data_option == "Upload your CSV":
        uploaded_file = st.file_uploader(
            "Upload sales data (CSV)", 
            type=["csv"],
            help="CSV should contain 'date' and 'sales' columns"
        )
        
        if uploaded_file:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.df['date'] = pd.to_datetime(st.session_state.df['date'])
            st.session_state.df = st.session_state.df.sort_values('date').reset_index(drop=True)
    
    st.header("Model Configuration")
    window_size = st.slider(
        "Window size (days)", 
        min_value=7, 
        max_value=90, 
        value=30,
        help="Number of historical days to use for each prediction"
    )
    
    epochs = st.slider(
        "Training epochs", 
        min_value=10, 
        max_value=500, 
        value=100,
        help="Number of training iterations"
    )
    
    forecast_days = st.slider(
        "Forecast horizon (days)", 
        min_value=7, 
        max_value=180, 
        value=90,
        help="How many days to predict into the future"
    )
    
    if st.button("Train Model", use_container_width=True):
        st.session_state.model_trained = True
        with st.spinner("Training model..."):
            # Create or get data
            if st.session_state.df is None:
                st.session_state.df = create_sample_data()
            
            # Prepare data
            X, y, scaler = prepare_data(st.session_state.df, window_size)
            
            # Train model
            device = 'cpu'  # Use CPU for Streamlit compatibility
            model, X_test, y_test = train_model(
                X, y, 
                window_size=window_size, 
                epochs=epochs, 
                device=device
            )
            
            # Save to session state
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

# Main content area
tab1, tab2, tab3 = st.tabs(["Data Preview", "Model Training", "Forecasting"])

with tab1:
    st.header("Sales Data")
    
    if st.session_state.df is None:
        st.info("Using sample data. Upload your own data in the sidebar.")
        st.session_state.df = create_sample_data()
    
    # Show data preview
    st.dataframe(st.session_state.df.head(), use_container_width=True)
    
    # Plot sales data
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(st.session_state.df['date'], st.session_state.df['sales'])
    ax.set_title("Historical Sales Data")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.grid(True)
    st.pyplot(fig)

with tab2:
    st.header("Model Training")
    
    if st.session_state.model_trained:
        # Evaluate model
        predictions, actuals, mae = evaluate_model(
            st.session_state.model,
            st.session_state.X_test,
            st.session_state.y_test,
            st.session_state.scaler
        )
        
        # Create test dates
        test_dates = st.session_state.df['date'][-len(actuals):]
        
        # Metrics
        col1, col2 = st.columns(2)
        col1.metric("Test Samples", len(actuals))
        col2.metric("Mean Absolute Error", f"${mae:.2f}")
        
        # Plot predictions
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test_dates, actuals, 'b-', label='Actual Sales', alpha=0.7)
        ax.plot(test_dates, predictions, 'r--', label='Predicted Sales', linewidth=2)
        ax.set_title("Actual vs Predicted Sales")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("Train the model using the sidebar controls")

with tab3:
    st.header("Future Sales Forecast")
    
    if st.session_state.model_trained:
        # Generate forecast
        future_sales = make_forecast(
            st.session_state.model,
            st.session_state.scaler,
            st.session_state.df,
            window_size=window_size,
            forecast_days=forecast_days
        )
        
        # Generate future dates
        last_date = st.session_state.df['date'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=forecast_days
        )
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': future_sales
        })
        
        # Show forecast data
        st.dataframe(forecast_df.head(10), use_container_width=True)
        
        # Plot forecast
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Historical data (last 180 days)
        hist_df = st.session_state.df[-180:]
        ax.plot(hist_df['date'], hist_df['sales'], 'b-', label='Historical Sales')
        
        # Forecast
        ax.plot(future_dates, future_sales, 'r-', label='Forecasted Sales')
        
        # Confidence interval
        ax.fill_between(
            future_dates, 
            future_sales * 0.9, 
            future_sales * 1.1, 
            color='red', alpha=0.1, label='Confidence Range'
        )
        
        ax.set_title(f"{forecast_days}-Day Sales Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # Export forecast
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name=f"sales_forecast_{pd.Timestamp.now().date()}.csv",
            mime="text/csv"
        )
    else:
        st.info("Train the model first to generate forecasts")

# Footer
st.markdown("---")
st.markdown("© 2023 Sales Forecasting App | Built with PyTorch and Streamlit")