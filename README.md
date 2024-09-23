# **Stock Price Prediction Using Hybrid LSTM-GNN Model**

## **Overview**
This project implements a hybrid model that combines **Long Short-Term Memory (LSTM)** networks for **time-series analysis** and **Graph Neural Networks (GNN)** for capturing **relational dependencies** between stocks. By leveraging both models, the system aims to improve the accuracy of stock price predictions in dynamic and volatile financial markets.

The project is developed as part of the dissertation for the Master of Science in Data Science and Advanced Computing at the University of Reading.

## **Features**
- **LSTM**: Handles sequential data to capture temporal patterns in stock prices.
- **GNN**: Captures relational dynamics between stocks using a graph structure based on Pearson correlation and association rules.
- **Hybrid Model**: Combines the LSTM and GNN outputs for more accurate stock price predictions.
- **Expanding Window Validation**: Dynamic training process that adapts to new data for real-time prediction.
  
## **Project Structure**
- **`data/`**: Contains the datasets used for training and testing.
- **`models/`**: Contains the implementations of LSTM, GNN, and the hybrid model.
- **`scripts/`**: Contains the scripts for preprocessing, training, and evaluating the models.
- **`notebooks/`**: Jupyter notebooks demonstrating the data analysis and model training.
- **`results/`**: Contains the performance metrics and visualizations from model evaluations.

## **Getting Started**

### **Prerequisites**
To run this project, ensure you have the following installed:
- Python 3.7+
- `tensorflow`
- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `networkx`
- `matplotlib`
- `yfinance`

### **Datasets**
The stock data used in this project comes from **Kaggle** (via **YFinance API**). Key features include:
- Open price
- Close price
- High/Low prices
- Volume

### **Data Preprocessing**
1. **Normalization**: Stock prices are normalized using Min-Max scaling.
2. **Graph Construction**: Stocks are represented as nodes in a graph, and their relationships (edges) are weighted based on Pearson correlation and association analysis.
3. **Batching**: Data is batched to feed into the LSTM model for capturing time-series patterns.

### **Model Training**
- **LSTM** is used to capture the sequential behavior of stock prices.
- **GNN** models the relationships between different stocks.
- The **Hybrid LSTM-GNN** model integrates both outputs and produces the final predictions.
- Training is conducted using **expanding window validation**, where new data is incrementally added to simulate real-world market conditions.

### **Evaluation Metrics**
The model is evaluated using **Mean Squared Error (MSE)**, with comparisons made against the following baseline models:
1. **Linear Regression**
2. **Convolutional Neural Network (CNN)**
3. **Dense Neural Network (DNN)**
4. **Standalone LSTM**

### **Project Results**
- The hybrid LSTM-GNN model outperformed traditional models like Linear Regression and CNN in terms of predictive accuracy.
- The **Mean Squared Error (MSE)** was consistently lower with the hybrid model across multiple stock datasets.
- Visualizations of the model's performance, including MSE across different stocks, can be found in the `results/` directory.

## **Acknowledgements**
- **University of Reading**: For providing the resources to conduct this research.
- **Professor Atta Badii**: For supervision and guidance throughout the project.
- **Kaggle and YFinance**: For providing the historical stock market data.
