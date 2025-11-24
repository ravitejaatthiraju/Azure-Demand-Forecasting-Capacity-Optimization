Azure Demand Forecasting & Capacity Optimization
📖 Project Overview
This project is a full-stack web application designed to forecast resource demand on the Azure cloud platform. It leverages a machine learning model to predict future CPU usage and provides a comprehensive dashboard for capacity planning, model monitoring, and data visualization. The goal is to empower capacity planners and data analysts to make data-driven decisions, preventing both costly over-provisioning and service-impacting resource shortages.

This application was built following an agile methodology across four key milestones, culminating in a production-ready tool.

✨ Key Features
Interactive Forecast Dashboard: Visualize historical demand and future forecasts for specific Azure regions and services (VM, Storage, Container).

Model Performance Analysis: A dedicated dashboard to compare the performance metrics (MAE, RMSE, MAPE, Training Time) of different ML models.

Capacity Planning & Recommendations: An intelligent dashboard that compares forecasted demand against available capacity, providing clear risk indicators (Red, Yellow, Green) and actionable recommendations.

Model Health Monitoring: A monitoring UI that tracks the forecasting model's accuracy over time, provides a "traffic light" health status, and alerts users to performance degradation.

Downloadable Reports: Users can download detailed forecast and capacity analysis reports in CSV format for offline use.

🛠️ Technology Stack
This project is built with a modern, scalable technology stack for both the backend and frontend.

Backend
Technology

Description

Python

The core language for data processing and API development.

Flask

A lightweight web framework used to build the REST API.

Gunicorn

A production-grade WSGI server for running the Flask application.

Pandas & NumPy

Libraries for efficient data manipulation and numerical analysis.

Scikit-learn & XGBoost

Machine learning libraries used to train and serve the forecasting model.

Docker

The backend application is containerized for easy deployment and scalability.

Frontend
Technology

Description

React.js

A JavaScript library for building the dynamic and interactive user interface.

Chart.js

Used for creating responsive and beautiful data visualizations.

Tailwind CSS

A utility-first CSS framework for styling the application.

Lucide React

A library for clean and consistent icons.

📂 Project Structure
The project is organized into two main parts: a backend Flask API and a frontend React application.

/azure-demand-forecasting
├── backend/
│   ├── api.py              # Main Flask application with all API endpoints
│   ├── models/
│   │   └── xgboost_model.joblib # The trained and serialized XGBoost model
│   ├── data/
│   │   └── processed/
│   │       └── featured_dataset.csv # The dataset used by the API
│   ├── Dockerfile          # Instructions to build the backend Docker image
│   └── requirements.txt    # Python dependencies
│
├── frontend/
│   ├── src/
│   │   └── App.jsx         # The main React component for the dashboard
│   ├── package.json        # Node.js dependencies
│   └── ...                 # Other React project files
│
├── LICENSE                 # Project's MIT License
└── README.md               # This file

🚀 Getting Started
To run this project locally, you will need to set up both the backend server and the frontend application.

Prerequisites
Python 3.8+

Node.js v16+ and npm

Docker (Optional, for containerized backend deployment)

1. Backend Setup
Navigate to the backend directory:

cd backend

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install Python dependencies:

pip install -r requirements.txt

Run the Flask API server:

flask run

The backend server will start and be available at http://127.0.0.1:5000.

2. Frontend Setup
Navigate to the frontend directory in a new terminal:

cd frontend

Install Node.js dependencies:

npm install

Run the React development server:

npm start

The frontend application will start and open automatically in your browser at http://localhost:3000.

⚙️ API Endpoints
The backend provides several endpoints to support the dashboard:

Method

Endpoint

Description

GET

/api/v1/demand-data

Retrieves the full historical dataset.

GET

/api/model-comparison

Returns performance metrics for all trained models.

GET

/api/forecast

Generates a 30-day forecast for a given region and service.

GET

/api/capacity-planning

Provides a capacity plan with recommendations and a risk level.

GET

/api/monitoring

Returns model health status and accuracy trends.

GET

/api/report

Generates and serves a downloadable CSV report.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

Copyright (c) 2025 Vidzai Digital