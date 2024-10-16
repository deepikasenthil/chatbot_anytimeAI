from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import HuggingFaceEmbeddings #SentenceTransformerEmbeddings 
# from langchain.vectorstores import Chroma 
from langchain_community.vectorstores import Chroma
import os 
from constants import CHROMA_SETTINGS

persist_directory = "db"

def main():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))
    documents = loader.load()
    print("splitting into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    print("Loading sentence transformers model")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #SentenceTransformerEmbeddings
    # **********************
    # D = Union[str, List[str]]  # Adjust based on your document format (single string or list of strings)
    # embeddings: embeddings[D] = embeddings
    # **************
    #create vector store here
    print(f"Creating embeddings. May take some minutes...")
    collection_name = "langchain"
    db = Chroma.from_documents(documents=texts, embedding=embeddings,persist_directory=persist_directory)
    # db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, ids = None,collection_name="langchain", client_settings=CHROMA_SETTINGS)
    db.persist()
    db=None 

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")

if __name__ == "__main__":
    main()