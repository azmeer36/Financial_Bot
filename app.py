from http import client
import pinecone
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
from langchain import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

from langchain.docstore.document import Document
# from psx import tickers
# import datetime 

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

    # vectorstore_from_docs = PineconeVectorStore.add_documents(documents=docs, embedding=embeddings)

    # vectorstore_from_docs = []
    
    return vectorstore_from_docs


def get_conversation_chain():
    embeddings = OpenAIEmbeddings()

    
    # financial_prompt = ChatPromptTemplate.from_messages([
    #         SystemMessagePromptTemplate.from_template(
    #             "The following is an informative conversation between a human and an AI financial adviser. The financial adviser will ask lots of questions. The financial adviser will attempt to answer any question asked and will probe for the human's risk appetite by asking questions of its own. If the human's risk appetite is low it will offer conservative financial advice, if the risk appetite of the human is higher it will offer more aggressive advice "
    #         ),
    #     ])

    # llm = ChatOpenAI()
    # # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    # # print("Type of retriever:", type(vectorstore.as_retriever()))
    # memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # print("memory is: ", memory)
    vectorstore = PineconeVectorStore.from_existing_index(index_name="fibot", embedding=embeddings)
    
    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=docsearch.as_retriever(),
    #     memory=memory
    # )
   
    # return conversation_chain
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    sys_prompt = "You are an AI financial adviser named Finley. You will attempt to answer any question asked and will probe for the human's risk appetite by asking questions of its own. If the human's risk appetite is low you will offer conservative financial advice, if the risk appetite of the human is higher you will offer more aggressive advice"
    conversation_chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template( sys_prompt)
    
    return conversation_chain

def submit():
    st.session_state.something = st.session_state.widget
    st.session_state.widget = ''


def handle_userinput(user_question):
    # st.session_state.conversation = get_conversation_chain()
    # print("user_question", user_question)
    # print("conversation", st.session_state.conversation)
    
    response = st.session_state.conversation({'question': user_question})
    # print("response is: ", response)
    st.session_state.chat_history = response['chat_history']

    # print("chat history is: ", st.session_state.chat_history)

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            

# def get_data():
#     ticker_list = tickers()
#     # data = stocks("SILK", start=datetime.date(2020, 1, 1), end=datetime.date.today())
#     print(ticker_list)
#     # print(data)


def main():
    # get_data()
   
    # print("I am here againnnnnnnnnnnnnnnnnnnnnnnnnnnn")
    # st.session_state.conversation = get_conversation_chain()
    
    
    # pc = Pinecone(api_key="147350ef-5846-457f-85e7-55f6bf459f85")
    # index = pc.Index("financialbot")

    # print(index.describe_index_stats())

    # index.upsert(vectors=to_upsert)
    
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        
        st.session_state.conversation = get_conversation_chain()
    if "chat_history" not in st.session_state:
        
        st.session_state.chat_history = None


    st.header("Chat with Finley :dollar:")
    user_question = st.text_input("Ask a questions on financials:")
    if user_question:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your neccessary documents here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                # vectorstore = get_vectorstore(text_chunks)
                vectorstore = get_vectorstore(text_chunks)
                # get_vectorstore(text_chunks)

                # pc = Pinecone(api_key="147350ef-5846-457f-85e7-55f6bf459f85")
                # index = pc.Index("financialbot")

                # print(index.describe_index_stats())

                # vectors_with_ids = [(f"id_{i}", vector) for i, vector in enumerate(query_result)]

                # index.upsert(vectors=vectors_with_ids)

                

                # create conversation chain
                # st.session_state.conversation = get_conversation_chain(
                #     vectorstore)


if __name__ == '__main__':
    main()
