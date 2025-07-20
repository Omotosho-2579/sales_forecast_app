# Sales Forecasting App

Predict future sales using PyTorch LSTM models with this Streamlit application.


## Features

- Upload your own sales data or use sample data
- Interactive model configuration
- Visualize historical sales patterns
- Generate future sales forecasts
- Download forecast results as CSV

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Omotosho-2579/sales-forecast-app.git
cd sales-forecast-app
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## How to Use

1. **Data Input**: 
   - Upload your CSV file with 'date' and 'sales' columns
   - Or use the sample data provided

2. **Model Configuration**:
   - Adjust the window size (historical days to consider)
   - Set the number of training epochs
   - Choose forecast horizon (days to predict)

3. **Train Model**:
   - Click "Train Model" in the sidebar
   - View training results and model performance

4. **Generate Forecast**:
   - Navigate to the Forecasting tab
   - View and download the sales forecast

## Deployment to Streamlit Sharing

1. Create a GitHub repository with these files
2. Go to [Streamlit Sharing](https://share.streamlit.io/)
3. Click "New app" and connect your GitHub repository
4. Specify the file path as `app.py`
5. Click "Deploy"

## File Structure
```
sales-forecast-app/
├── app.py                  # Main Streamlit application
├── forecasting_model.py    # PyTorch model and helper functions
├── requirements.txt        # Python dependencies
└── README.md               # Instructions
```

## License
MIT License