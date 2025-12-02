from flask import Flask, render_template, request, jsonify
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Initialize Flask app
app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load the embeddings model
embeddings = download_embeddings()

# Give index name
index_name = "medical-chatbot"
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

# Creating Chain
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initializing Ollama Chat Model
chatModel = ChatOllama(
    model="llama2",   # Best for medical Q&A
    temperature=0.1
)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Chain
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# Create Default Route
@app.route("/")
def index():
    return render_template("chat.html")


# Create CHAT Route
@app.route("/ragsearch", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(f"Requesting input --> '{input}'")
    response = rag_chain.invoke({"input": msg})
    print(f"Response --> '{response['answer']}'")
    return str(response['answer'])


# To execute the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)