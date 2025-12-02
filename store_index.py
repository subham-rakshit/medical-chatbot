from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Access env keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Set as a environment variable
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Extract text from pdf file
extracted_data = load_pdf_files("data/")

# Filter minical docs
minimal_docs = filter_to_minimal_docs(extracted_data)

# Split the data into Text Chunks
text_chunk = text_split(minimal_docs)

# Download Embeddings
embeddings = download_embeddings()

# Pass pinecone api key
pinecone_api_key = PINECONE_API_KEY

# Authenticate pinecone account
pc = Pinecone(api_key=pinecone_api_key)

# Creating index (means creating DB, which will show into pinecone platform)
index_name = "medical-chatbot"
if not pc.has_index(index_name):
    # Create index
    pc.create_index(
        name = index_name,
        dimension=384,  # Dimension of the vectors (We are using sentence-tranformer embedding model which gives 384 dimensions).
        metric="cosine",  # Metric type used to measure the distance between vectors.
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )
index = pc.Index(index_name)

# Store Vectors (It will take all of the text chunks, and use the embedding model and try to convert them as vector embedding and it will store in the Pincone vector DB.)
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embeddings,
    index_name=index_name,
)