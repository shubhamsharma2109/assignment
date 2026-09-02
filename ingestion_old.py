import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import Chroma

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()



DATA_DIR = os.getenv("DATA_DIR")
CHROMA_DIR = os.getenv("CHROMA_DIR")

# Open-source embedding model
EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300


def load_documents():

    documents = []

    for filename in os.listdir(DATA_DIR):

        path = os.path.join(
            DATA_DIR,
            filename
        )

        if filename.lower().endswith(".pdf"):

            loader = PyPDFLoader(path)
            docs = loader.load()

        elif filename.lower().endswith(".docx"):

            loader = Docx2txtLoader(path)
            docs = loader.load()

        elif filename.lower().endswith(".txt"):

            loader = TextLoader(
                path,
                encoding="utf-8"
            )

            docs = loader.load()

        else:
            print(f"Skipping: {filename}")
            continue

        for doc in docs:
            doc.metadata["source"] = filename

        documents.extend(docs)

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


def create_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }
    )


def create_chroma(chunks):

    embeddings = create_embeddings()

    print(
        f"Creating Chroma database "
        f"with {len(chunks)} chunks..."
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="rag_documents"
    )

    print(
        "Chroma database created successfully."
    )

    return vectorstore


if __name__ == "__main__":

    print("Loading documents...")

    documents = load_documents()

    if not documents:

        raise RuntimeError(
            "No supported documents found in data/"
        )

    print(
        f"Loaded {len(documents)} documents/pages."
    )

    print("Splitting documents...")

    chunks = split_documents(documents)

    print(
        f"Created {len(chunks)} chunks."
    )

    create_chroma(chunks)
