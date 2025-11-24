# Azure Demand Forecasting & Capacity Optimization ☁️ 📊

![Project Status](https://img.shields.io/badge/Status-Completed-green)
![Tech Stack](https://img.shields.io/badge/Stack-MERN%20%2B%20Flask-blue)
![Model](https://img.shields.io/badge/Best%20Model-XGBoost-orange)

## 📖 Project Overview

This project is a full-stack data science application designed to predict future cloud resource consumption (CPU, Storage) for Microsoft Azure services across different regions. The goal is to assist organizations in capacity planning and cost optimization by providing accurate, data-driven demand forecasts.

The system ingests historical usage data, performs advanced feature engineering, trains multiple machine learning models (ARIMA, XGBoost, LSTM), and visualizes the results on an interactive React dashboard.

## ✨ Key Features

* **Data Pipeline:** Robust data cleaning and merging of Azure usage logs with external economic factors.
* **Feature Engineering:** Creation of time-based features (Year-Week, Month) and trend indicators (7-day rolling averages) to improve model accuracy.
* **Multi-Model Evaluation:** Comparative analysis of three distinct forecasting approaches:
    * **ARIMA:** Statistical baseline.
    * **XGBoost:** Gradient boosting (Selected as the Best Model).
    * **LSTM:** Deep learning (Recurrent Neural Network).
* **Interactive Dashboard:** A responsive UI built with React and Chart.js featuring:
    * **Live Forecasting:** 30-day future predictions with confidence intervals.
    * **Model Comparison:** Side-by-side metrics (MAE, RMSE, MAPE, Inference Speed).
    * **Usage Trends:** Historical data visualization.
    * **Correlation Analysis:** Heatmaps showing relationships between variables.
* **REST API:** A Python Flask backend serving predictions and processed data.

## 🛠️ Tech Stack

### **Frontend**
* **React.js (Vite):** User Interface.
* **Tailwind CSS:** Styling and responsive design.
* **Chart.js / React-Chartjs-2:** Data visualization and forecasting charts.
* **Lucide React:** Icons.

### **Backend & Machine Learning**
* **Python:** Core programming language.
* **Flask:** API development.
* **Pandas & NumPy:** Data manipulation and processing.
* **Scikit-Learn:** Data scaling and metrics.
* **XGBoost:** Gradient boosting framework.
* **TensorFlow/Keras:** LSTM neural network construction.
* **Joblib:** Model serialization.

## 📊 Model Performance

After rigorous backtesting and evaluation, **XGBoost** was selected as the production model due to its balance of high accuracy and low latency.

| Model | MAE | RMSE | MAPE | Inference Speed |
|-------|-----|------|------|-----------------|
| **XGBoost** | **24.64** | **31.55** | **3.44%** | **0.04ms** |
| ARIMA | 89.26 | 105.79 | 13.06% | 114.12ms |
| LSTM | 132.03 | 145.29 | 18.23% | 2.15ms |

## 🚀 Getting Started

### Prerequisites
* Node.js & npm
* Python 3.8+

### Installation

**1. Clone the Repository**
```bash
git clone [https://github.com/yourusername/Azure-Demand-Forecasting.git](https://github.com/yourusername/Azure-Demand-Forecasting.git)
cd Azure-Demand-Forecasting

2. Backend Setup
cd backend
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API server
python api.py

3. Frontend Setup
cd frontend-dashboard
npm install
npm run dev
The application will be available at http://localhost:5173 (or the port specified by Vite).
📂 Project Structure
├── data/
│   ├── raw/             # Original datasets
│   └── processed/       # Cleaned and featured datasets
├── notebooks/           # Jupyter notebooks for cleaning, training, and analysis
├── models/              # Saved .joblib models
├── backend/             # Flask API (api.py)
└── frontend-dashboard/  # React Application

Author
Atthiraju Raviteja

Role: Full Stack Data Science Intern @ Infosys Springboard