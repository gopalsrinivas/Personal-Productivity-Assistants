import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Get ChromaDB storage path
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "db").strip('"')

# Sample Documents
documents = [
    Document(page_content="Meeting notes: Discuss project X deliverables."),
    Document(page_content="Reminder: Submit report by Friday."),
    Document(page_content="Upcoming event: Tech conference next Wednesday."),
    Document(
        page_content="What is Generative AI? Generative AI creates new content based on learned patterns."
    ),
    Document(
        page_content="AI can assist with productivity by automating repetitive tasks."
    ),
]


# Use Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Split text for better search
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Store embeddings in ChromaDB
vector_db = Chroma.from_documents(
    docs, embedding=embeddings, persist_directory=CHROMA_DB_PATH
)

print("✅ Documents successfully indexed into ChromaDB!")
