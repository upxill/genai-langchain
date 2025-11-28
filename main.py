import streamlit as st
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
import langchain

langchain.debug = True
load_dotenv()  # take environment variables from .env (especially openai api key)


def load_documents(urls):
    """Load documents from a list of URLs."""
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()
    return documents


def split_documents(documents):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_documents(documents)


def create_vector_store(documents):
    """Create a vector store from the documents."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def create_qa_chain(vector_store):
    """Create a QA chain using the vector store."""
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), return_source_documents=True
    )
    return qa_chain


def main():
    st.title("Document Question Answering System")

    urls = st.text_area("Enter URLs (one per line):", height=200)
    if st.button("Load Documents"):
        if urls:
            urls_list = urls.splitlines()
            with st.spinner("Loading documents..."):
                documents = load_documents(urls_list)
                st.session_state["documents"] = documents
                st.success(f"Loaded {len(documents)} documents.")
        else:
            st.error("Please enter at least one URL.")

    if "documents" in st.session_state:
        if st.button("Split Documents"):
            with st.spinner("Splitting documents..."):
                split_docs = split_documents(st.session_state["documents"])
                st.session_state["split_docs"] = split_docs
                st.success(f"Split into {len(split_docs)} chunks.")

        if "split_docs" in st.session_state and st.button("Create Vector Store"):
            with st.spinner("Creating vector store..."):
                vector_store = create_vector_store(st.session_state["split_docs"])
                st.session_state["vector_store"] = vector_store
                st.success("Vector store created.")

        if "vector_store" in st.session_state and st.button("Create QA Chain"):
            with st.spinner("Creating QA chain..."):
                qa_chain = create_qa_chain(st.session_state["vector_store"])
                st.session_state["qa_chain"] = qa_chain
                st.success("QA chain created.")

        question = st.text_input("Ask a question:")
        if question and "qa_chain" in st.session_state:
            with st.spinner("Getting answer..."):
                result = st.session_state["qa_chain"]({"question": question})
                answer = result["answer"]
                sources = result["sources"]
                st.write(f"**Answer:** {answer}")
                st.write(f"**Sources:** {', '.join(sources)}")


if __name__ == "__main__":
    # Enable debug mode for LangChain
    main()
