from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from multiprocessing import Lock


with st.sidebar:
    st.title('LLM chat App')
    st.markdown("""
                This app is about chatting with your own application
                - [Streamlit](https://streamlit.io)
                - [LangChain](https://python.langchain.com)
                """)
    add_vertical_space(5)
    st.write('Made by Santhosh')

def main():
    load_dotenv()
    
    st.header('Chat with your PDF')
    pdf=st.file_uploader("Upload your pdf file")
    
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks=text_splitter.split_text(text)
        
        store_name=pdf.name[:-4]
        st.write(f'{store_name}')
        
        embeddings=OpenAIEmbeddings()
        VectorStore= FAISS.from_texts(chunks, embedding=embeddings)
        
        # lock = Lock()
        # if os.path.exists(f'{store_name}.pkl'):
        #     with open(f'{store_name}.pkl', 'rb') as f:
        #         with lock:
        #             VectorStore = pickle.load(f)
        # else:
        #     embeddings=OpenAIEmbeddings()
        #     VectorStore= FAISS.from_texts(chunks, embedding=embeddings)
        #     with open(f'{store_name}.pkl', 'wb') as f:
        #         pickle.dump(VectorStore, f)
        
        query=st.text_input("Ask question about your pdf file:")
        
        if query:
            docs=VectorStore.similarity_search(query=query)            
            chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question = query)
                print(cb)
                st.write(response)
                
                
if __name__ == '__main__':
    main()