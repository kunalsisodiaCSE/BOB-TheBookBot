import streamlit as stream
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlUserBot import css, botSetting, userSetting
from langchain.llms import HuggingFaceHub

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
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = stream.session_state.conversation({'question': user_question})
    stream.session_state.chat_history = response['chat_history']

    for i, message in enumerate(stream.session_state.chat_history):
        if i % 2 == 0:
            stream.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            stream.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    stream.set_page_config(page_title="Bob-The BookBot",
                       page_icon="🤖")
    stream.write(css, unsafe_allow_html=True)

    if "conversation" not in stream.session_state:
        stream.session_state.conversation = None
    if "chat_history" not in stream.session_state:
        stream.session_state.chat_history = None

    stream.header("BOB-The BookBot 🤖")
    user_question = stream.text_input("Ask a question related to you pdfs:")
    if user_question:
        handle_userinput(user_question)

    with stream.sidebar:
        stream.subheader("Your Files")
        pdf_docs = stream.file_uploader("Upload your PDFs below, then Click on 'Process'", accept_multiple_files=True)
        if stream.button("Process"):
            with stream.spinner("Processing"):
                # Extracting text from pdf
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks from raw text data
                text_chunks = get_text_chunks(raw_text)

                # creating vector store from the chunks of Text
                vectorstore = get_vectorstore(text_chunks)

                # Create a chain for converted vectorStore
                stream.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
