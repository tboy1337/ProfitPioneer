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
   git clone https://github.com/tboy1337/ProfitPioneer.git
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

## Deployment

### Streamlit Cloud

The easiest way to deploy InsightCommerce Pro is using Streamlit Cloud:

1. Push your code to a GitHub repository
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your GitHub repository
4. Select app.py as the main file
5. Click "Deploy"

### Heroku

To deploy on Heroku:

1. Create a `requirements.txt` file (if not already present):
   ```
   pip freeze > requirements.txt
   ```

2. Create a `Procfile` in the project root:
   ```
   echo "web: streamlit run app.py --server.port \$PORT --server.enableCORS false" > Procfile
   ```

3. Deploy to Heroku:
   ```
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

### Docker

To containerize the application:

1. Create a `Dockerfile` in the project root:
   ```
   FROM python:3.11-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. Build and run the Docker container:
   ```
   docker build -t insightcommerce-pro .
   docker run -p 8501:8501 insightcommerce-pro
   ```

3. For cloud deployment (e.g., AWS ECS, Google Cloud Run):
   - Push your image to a container registry
   - Configure your cloud service to use this image
   - Set appropriate environment variables for your database and any API keys

### Self-hosted (nginx)

To deploy on your own server with nginx:

1. Set up a systemd service (on Linux):
   ```
   [Unit]
   Description=InsightCommerce Pro Streamlit App
   After=network.target

   [Service]
   User=your_user
   WorkingDirectory=/path/to/ProfitPioneer
   ExecStart=/path/to/ProfitPioneer/venv/bin/streamlit run app.py --server.port 8501
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

2. Configure nginx as a reverse proxy:
   ```
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header Host $host;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_read_timeout 86400;
       }
   }
   ```

3. Secure with SSL using Certbot/Let's Encrypt

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
├── requirements.txt       # Dependency list for deployment
├── Procfile               # For Heroku deployment
├── Dockerfile             # For Docker deployment
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