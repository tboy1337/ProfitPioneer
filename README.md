# InsightCommerce Pro

![InsightCommerce Pro](generated-icon.png)

A comprehensive e-commerce analytics dashboard built with Streamlit that provides business intelligence insights for online retail businesses.

## Features

- **Dashboard Overview**: Real-time KPIs and performance metrics
- **Sales History Analysis**: Track and visualize sales trends over time
- **Product Analysis**: Deep dive into product performance and inventory metrics
- **Customer Segmentation**: Analyze customer behavior and segment profitability
- **Sales Forecasting**: Predictive analytics for future revenue planning
- **Data Import**: Flexible data import capabilities for CSV/Excel files

## Tech Stack

- Python 3.11+
- Streamlit for web interface
- Pandas for data processing
- Plotly for interactive visualizations
- Scikit-learn for forecasting and analytics

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ProfitPioneer.git
   cd ProfitPioneer
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   
   Or using a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. To get started:
   - Use the sample data option in the sidebar
   - Or upload your own e-commerce data through the Data Import page
   - Navigate through the different analysis modules using the sidebar menu

## Project Structure

```
ProfitPioneer/
├── app.py                 # Main application entry point
├── pages/                 # Multi-page app components
│   ├── 01_Sales_History.py
│   ├── 02_Product_Analysis.py
│   ├── 03_Customer_Segmentation.py
│   ├── 04_Sales_Forecast.py
│   └── 05_Data_Import.py
├── utils/                 # Utility modules
│   ├── data_processor.py  # Data preparation and transformation
│   ├── forecasting.py     # Time series forecasting models
│   └── visualization.py   # Chart and visualization components
├── .streamlit/            # Streamlit configuration
├── pyproject.toml         # Project dependencies
└── README.md              # This documentation
```

## Data Format

The application expects e-commerce data with the following attributes:
- Transaction ID
- Date
- Customer ID
- Product information (ID, name, category)
- Revenue and cost metrics
- Customer segmentation data (optional)
- Geographic information (optional)

Sample data is provided within the application for demonstration purposes.

## Customization

You can customize the dashboard by:
- Modifying the CSS styles in the app.py file
- Adding new visualizations in the utils/visualization.py module
- Extending analytics capabilities in the respective page files

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 