import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

OPENAI_API_KEY = "Paste your OpenAI API key here"

st.header("📘 NoteBot - AI Study Assistant")

with st.sidebar:
    st.title("Upload Notes")
    file = st.file_uploader("Upload a PDF and start asking questions", type="pdf")

if file is not None:
    pdf_reader = PdfReader(file)
    text = "".join([page.extract_text() for page in pdf_reader.pages])

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embeddings)

    user_query = st.text_input("Ask me anything from your notes")

    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)

        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=300
        )

        prompt = ChatPromptTemplate.from_template(
            """You are my study assistant. Answer the question based on the given context.
            If the answer is not found, reply with: "I don't know Jenny". 

            Context:
            {context}

            Question: {input}
            """
        )

        chain = create_stuff_documents_chain(llm, prompt)
        output = chain.invoke({"input": user_query, "input_documents": matching_chunks})

        st.subheader("Answer:")
        st.write(output)
