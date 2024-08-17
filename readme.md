# Stock Analysis Chatbot (Finley & Lex)
![Screenshot (90)](https://github.com/user-attachments/assets/d862e83e-5e6a-4c92-a8f1-67d7d7085d91)

## Description

This project implements a stock analysis chatbot that provides various functionalities related to stock analysis and financial advice. It leverages multiple APIs and libraries including OpenAI, yfinance, Streamlit, and Pinecone for natural language processing, stock data retrieval, and conversation management.

## Features

- **Stock Price Retrieval:** Get the latest stock price given the ticker symbol of a company.
- **Simple Moving Average (SMA) Calculation:** Calculate the simple moving average for a given stock ticker and window.
- **Exponential Moving Average (EMA) Calculation:** Calculate the exponential moving average for a given stock ticker and window.
- **Relative Strength Index (RSI) Calculation:** Calculate the RSI for a given stock ticker.
- **Moving Average Convergence Divergence (MACD) Calculation:** Calculate the MACD for a given stock ticker.
- **Stock Price Plotting:** Plot the stock price for the last year given the ticker symbol of a company.
- **Batch Stock Quotes Retrieval:** Retrieves the latest stock prices for a list of ticker symbols and returns them as a formatted string.
- **Investment Advice Generation:** Gives investment advice by analyzing dividend rates, return on equity, and PEG ratios of specified stocks.

## Deployed App

You can access the deployed app at: [Deployed App Link](https://financialbot-finley.streamlit.app/)
  
![Screenshot (89)](https://github.com/user-attachments/assets/4a43291e-ae08-42ab-9c31-8de8e4e7ac1f)

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    
3. **Set up environment variables:**

  Create a .env file in the root directory and add your API keys for OpenAI and Pinecone
  ```bash
  OPENAI_API_KEY=your_openai_api_key
  PINECONE_API_KEY=your_pinecone_api_key
  ```

4. **Run the application:**
  ```bash
  streamlit run app.py
  ```

