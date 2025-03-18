import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from transformers import pipeline

# Load environment variables
load_dotenv()

# Get ChromaDB storage path
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "db").strip('"')

# Load stored vector database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

# Convert vector database into retriever
retriever = vector_db.as_retriever()

# Load Hugging Face model for text generation
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Given the following documents:\n{context}\nAnswer the question: {question}",
)


# Create a Lambda Runnable to format input
def format_input(data):
    return prompt_template.format(context=data["context"], question=data["question"])


# Chain with formatting and pipeline execution
qa_chain = RunnableLambda(format_input) | qa_pipeline


def get_response(query):
    if not query or not isinstance(query, str):
        return "Invalid query. Please provide a valid question."

    query = query.strip().replace("\n", " ")

    try:
        # Retrieve relevant context from vector database
        retrieved_docs = retriever.invoke(query)
        print("Retrieved Documents:", retrieved_docs)
        context = (
            " ".join([doc.page_content for doc in retrieved_docs])
            or "No relevant context found."
        )

        # Ensure the input is correctly structured
        formatted_input = {
            "context": context,
            "question": query,
        }

        # Call the chain with a properly formatted dictionary
        response = qa_chain.invoke(formatted_input)

        # Ensure the response is a string before returning
        return (
            response[0]["generated_text"]
            if isinstance(response, list)
            else str(response)
        )

    except Exception as e:
        return f"Error processing request: {str(e)}"
