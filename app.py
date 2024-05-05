from http import client
import pinecone
import json
import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from pinecone import Pinecone, ServerlessSpec
import yfinance as yf
import matplotlib.pyplot as plt
from langchain.docstore.document import Document
# from psx import stocks, tickers
import datetime 


load_dotenv()
# Initialize Pinecone client
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    docs = [Document(page_content=text) for text in chunks]
    return docs


def get_vectorstore(docs):
    embeddings = OpenAIEmbeddings()

    
    vectorstore_from_docs = PineconeVectorStore.from_documents(
        docs,
        index_name="fibot",
        embedding=embeddings
    )

    return vectorstore_from_docs


def get_conversation_chain():
    embeddings = OpenAIEmbeddings()

    vectorstore = PineconeVectorStore.from_existing_index(index_name="fibot", embedding=embeddings)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    llm_with_tools= llm.bind_tools([get_stock_price, calculate_EMA,calculate_MACD,calculate_RSI,calculate_SMA,plot_stock_price])
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_with_tools,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        # functions = functions,
        # function_call = 'auto'
    )
    return conversation_chain

def submit():
    st.session_state.something = st.session_state.widget
    st.session_state.widget = ''

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    print("response", response)
    
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

            
# real time data
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
            
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if 'messages' not in st.session_state:
        st.session_state['messages']= []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                


if __name__ == '__main__':
    main()
