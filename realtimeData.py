import json
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain

load_dotenv()

api_key= os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)
print(client)


llm_chain = LLMChain(llm=client)

def query_llm(question):
    print(llm_chain.invoke({'question': question})['text'])





def get_stock_price(ticker):
    return str(yf.Ticker(ticker).history(period='1y').iloc[-1].Close)

def calculate_SMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.rolling(window=window).mean().iloc[-1])

def calculate_EMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])

def calculate_RSI(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14-1, adjust=False).mean()
    ema_down = down.ewm(com=14-1, adjust=False).mean()
    rs = ema_up / ema_down
    return str(100 - (100 / (1+rs)).iloc[-1])

def calculate_MACD(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()
    macd = short_EMA - long_EMA
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_histogram = macd-signal
    return f'{macd[-1]}, {signal[-1]}, {macd_histogram[-1]}'

def plot_stock_price(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.Close)
    plt.title(f'{ticker} Stock Price Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True)
    plt.savefig('stock.png')
    plt.close()


functions =  [
    {
      "name": "get_stock_price",
      "description": "Gets the latest stock price given the ticker symbol of a company.",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)."
          }
        },
        "required": ["ticker"]
      }
    },
    {
      "name": "calculate_SMA",
      "description": "Calculate the simple moving average for a given stock ticker and a window.",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)."
          },
          "window": {
            "type": "integer",
            "description": "The timeframe to consider when calculating the SMA."
          }
        },
        "required": ["ticker", "window"]
      }
    },
    {
      "name": "calculate_EMA",
      "description": "Calculate the exponential moving average for a given stock ticker and a window.",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)."
          },
          "window": {
            "type": "integer",
            "description": "The timeframe to consider when calculating the EMA."
          }
        },
        "required": ["ticker", "window"]
      }
    },
    {
      "name": "calculate_RSI",
      "description": "Calculate the RSI for a given stock ticker.",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)."
          },
        },
        "required": ["ticker"]
      }
    },
    {
      "name": "calculate_MACD",
      "description": "Calculate the MACD for a given stock ticker",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)."
          },
        },
        "required": ["ticker"]
      }
    },
    {
      "name": "plot_stock_price",
      "description": "Plot the stock price for the last year given the ticker symbol of a company.",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)."
          }
        },
        "required": ["ticker"]
      }
    }
  ]

available_functions= {
    'get_stock_price': get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_EMA': calculate_EMA,
    'calculate_RSI': calculate_RSI,
    'calculate_MACD': calculate_MACD,
    'plot_stock_price': plot_stock_price,
}

if 'messages' not in st.session_state:
    st.session_state['messages']= []
  
if "chat_history" not in st.session_state:
        print("hereeeeeeeeeeeeeeeee")
        st.session_state.chat_history = None

embeddings = OpenAIEmbeddings()
vectordb = PineconeVectorStore.from_existing_index(index_name="fibot", embedding=embeddings)
memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3), vectordb.as_retriever(), memory=memory)

st.title("Stock analysis Chatbot")

user_input = st.text_input('Your input: ')

if user_input:
    try:
        st.session_state['messages'].append({'role':'user', 'content': f'{user_input}'})
        # response = client.chat.completions.create(
        #     model= 'gpt-3.5-turbo',
        #     messages= st.session_state['messages'],
        #     functions=functions,
        #     function_call='auto',
        # )
        response = query_llm(user_input)
        print("my response: ",response)

        response_message =  response.choices[0].message
        print(response_message)
        # st.session_state.chat_history = response_message
        print(response_message)
        
        if response_message.function_call:
            function_name = response_message.function_call.name
            function_args = json.loads(response_message.function_call.arguments)

            if function_name in ['get_stock_price', 'calculate_RSI', 'calculate_MACD', 'plot_stock_price']:
                args_dict = {'ticker': function_args.get('ticker')}
            elif function_name in ['calculate_SMA', 'calculate_EMA']:
                args_dict = {'ticker': function_args.get('ticker'), 'window': function_args.get('window')}

            function_to_call = available_functions[function_name]
            function_response = function_to_call(**args_dict)

            if function_name == 'plot_stock_price':
                st.image('stock.png')
            else:
                st.session_state['messages'].append(response_message)
                st.session_state['messages'].append({
                    'role': 'function',
                    'name': function_name,
                    'content': function_response
                })

            second_response= client.chat.completions.create(
                model= 'gpt-3.5-turbo',
                messages= st.session_state['messages'],
            )

            st.text(second_response.choices[0].message.content)
            st.session_state['messages'].append({'role': 'assistant', 'content': second_response.choices[0].message.content})
        else:
            #Add here a retriver

            
            answer = (qa_chain({"question": user_input}))

            print(answer)
            # print("chain: ", qa_chain)

            



            print("here")
            st.text(answer['answer'])
            st.session_state['messages'].append({'role': 'assistant', 'content': response_message.content})
            st.session_state['messages'].append({'role': 'user', 'content': f"{answer}" })
            st.session_state.chat_history = answer
            print("chat history: ", memory)

    except Exception as e:
        raise(e)