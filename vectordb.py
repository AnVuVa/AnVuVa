import os

import shutil
from typing import List
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from camel.embeddings import MistralEmbedding, OpenAIEmbedding
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from langchain_unstructured import UnstructuredLoader
from concurrent.futures import ThreadPoolExecutor, as_completed


# ================== FUNCTION: Load Document ==================
def load_chunks_from_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Loads and chunks a document file using UnstructuredIO and LangChain's splitter.
    """
    loader = UnstructuredLoader(file_path)
    elements = loader.load()
    content = ("\n\n".join([el.page_content for el in elements]))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([content])


# ================== FUNCTION: Get Embedding Model ==================
def get_embedding_model(model_name: str):
    """
    Returns CAMEL-compatible embedding model based on user selection.
    """
    if model_name.lower() == "mistral":
        return MistralEmbedding(api_key=os.environ["MISTRAL_API_KEY"])
    elif model_name.lower() == "openai":
        return OpenAIEmbedding(api_key=os.environ["OPENAI_API_KEY"])
    elif model_name.lower() == "hf":
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
    else:
        raise ValueError("Model must be 'mistral', 'openai', or 'hf'")


# ================== FUNCTION: Store Chunks to FAISS ==================
def store_chunks_to_faiss(chunks: List[Document], embedding_model, faiss_path: str):
    """
    Adds embedded chunks to an existing FAISS vector store or creates a new one.

    Parameters:
        chunks (List[Document]): List of chunked documents with embeddings in metadata.
        embedding_model: The embedding model (Mistral, OpenAI, HuggingFace).
        faiss_path (str): Directory path to save the FAISS index.
    """
    if os.path.exists(faiss_path):
        # Load the existing FAISS index
        vectorstore = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
        # Add new documents to the existing index
        vectorstore.add_documents(chunks)
    else:
        # Create a new FAISS index
        vectorstore = FAISS.from_documents(chunks, embedding_model)
    
    # Save the updated index
    vectorstore.save_local(faiss_path)


# ================== FUNCTION: Connect Semantically Related Chunks ==================
def connect_semantically_related_chunks(chunks: List[str], model_name: str = "mistral", max_length: int = 1024) -> List[str]:
    """
    Uses an LLM to determine semantic connections between chunks and rechunks them based on similarity and max length.

    Parameters:
        chunks (List[str]): List of text chunks.
        model_name (str): Model type ('mistral' or 'openai').
        max_length (int): Maximum length of a single chunk after merging.

    Returns:
        List[str]: Rechunked chunks based on semantic connections.
    """
    model = ModelFactory.create(
        model_type=ModelType.MISTRAL_LARGE if model_name == "mistral" else ModelType.GPT_4O_MINI,
        model_platform=ModelPlatformType.OLLAMA if model_name == "mistral" else ModelPlatformType.OPENAI
    )
    return chunks # keep it for now
    rechunked_chunks = []
    current_chunk = chunks[0]  # Start with the first chunk

    for i in tqdm(range(1, len(chunks)), desc="🔍 Rechunking chunks"):
        chunk_a = current_chunk
        chunk_b = chunks[i]  

        messages = [
            {"role": "system", "content": "You are a helpful assistant that determines semantic connections between text chunks."},
            {"role": "user", "content": f"Chunk A:\n{chunk_a}\n\nChunk B:\n{chunk_b}\n\nAre these two chunks semantically related and should be connected in a document-level context? Respond with 'yes' or 'no' only."}
        ]

        response = model.run(messages)
        if response == "yes" and len(current_chunk) + len(chunks[i]) <= max_length:
            # Merge the current chunk with the next chunk
            current_chunk += "\n\n" + chunks[i]
        else:
            # Add the current chunk to the rechunked list and start a new chunk
            rechunked_chunks.append(current_chunk)
            current_chunk = chunks[i]

    # Add the last chunk
    rechunked_chunks.append(current_chunk)

    return rechunked_chunks


# ================== FUNCTION: Process Batch ==================
def process_batch(batch: List[Document], embedding_model) -> List[Document]:
    """
    Helper function to process a single batch of document chunks.

    Parameters:
        batch (List[Document]): List of document chunks to embed.
        embedding_model: The embedding model (Mistral, OpenAI, HuggingFace).

    Returns:
        List[Document]: Embedded document chunks.
    """
    embeddings = embedding_model.embed_documents([chunk.page_content for chunk in batch])
    return [
        Document(page_content=chunk.page_content, metadata={"embedding": embedding})
        for chunk, embedding in zip(batch, embeddings)
    ]


# ================== FUNCTION: Parallel Embedding ==================
from concurrent.futures import ThreadPoolExecutor, as_completed

def parallel_embedding(chunks: List[Document], embedding_model, batch_size: int = 8, n_workers: int = 1) -> List[Document]:
    """
    Parallelizes the embedding of chunks with the option to batch them.

    Parameters:
        chunks (List[Document]): List of document chunks to embed.
        embedding_model: The embedding model (Mistral, OpenAI, HuggingFace).
        batch_size (int): Number of chunks to embed at a time.
        n_workers (int): Number of parallel workers.

    Returns:
        List[Document]: Embedded chunks.
    """

    # Split chunks into batches
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

    if n_workers > 1:
        # Use ThreadPoolExecutor for parallel execution
        embedded_chunks = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_batch, batch, embedding_model): batch for batch in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc="🔄 Embedding batches"):
                try:
                    embedded_chunks.extend(future.result())
                except Exception as e:
                    print(f"[Error] Batch failed: {e}")
    else:
        # Sequential processing for single worker with progress tracking
        embedded_chunks = [chunk for batch in tqdm(batches, desc="🔄 Embedding batches") for chunk in process_batch(batch, embedding_model)]

    # Flatten the list of results
    return embedded_chunks

# ================== FUNCTION: Document to RAG Pipeline ==================
def document_to_rag_pipeline(file_paths: List[str],
                             model_name: str = "mistral",
                             faiss_path: str = "faiss_index",
                             n_workers: int = 1,
                             batch_size: int = 8):
    """
    Complete RAG pipeline with parallel batch processing and Hugging Face support.
    """
    print("Loading and chunking document...")
    chunks = []
    for file_path in tqdm(file_paths, desc="📄 Loading documents"):
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            continue
        # Load and chunk the document
        chunks += load_chunks_from_document(file_path)
        print(f"Loaded {len(chunks)} chunks from {file_path}")

    print("Finding semantic connections with LLM...")
    rchunks = connect_semantically_related_chunks(chunks, model_name=model_name)

    print("Embedding chunks in parallel...")
    embedding_model = get_embedding_model(model_name)
    embedded_chunks = parallel_embedding(rchunks, embedding_model, batch_size=batch_size, n_workers=n_workers)

    print("Storing chunks in FAISS...")
    store_chunks_to_faiss(embedded_chunks, embedding_model, faiss_path)


if __name__ == "__main__":
    document_to_rag_pipeline(
        file_paths = [r"/workspaces/YuE/books/Atlas of HEART FAILURE _ Cardiac Function and Dysfunction -- Arnold M_ Katz (auth_), Wilson S_ Colucci MD (eds_) -- Softcover reprint of the original -- 9781475745580 -- 5bec97274a4e43ffe879a28356cddf22 -- Anna’s.pdf"],
        model_name = "hf",  # Change to "mistral", "openai", or "hf"
        faiss_path = "faiss_index",
        n_workers = 4,
        batch_size = 32
    )


# ================== FUNCTION: Create Retriever ==================
def create_retriever(faiss_index_path: str, k: int = 5):
    """
    Create a retriever from a FAISS index.

    Parameters:
        faiss_index_path (str): Path to the FAISS index directory.
        embedding_model_name (str): Name of the embedding model.

    Returns:
        retriever: A retriever object for querying the FAISS index.
    """
    # Initialize the embedding model
    embeddings = get_embedding_model("hf")

    # Load the FAISS index
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

    # Create a retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever


# ================== FUNCTION: Add Books to FAISS ==================
def add_books_to_faiss(faiss_path: str, book_paths: List[str], model_name: str = "mistral", n_workers: int = 1, batch_size: int = 8):
    """
    Add multiple books to a FAISS index.

    Parameters:
        faiss_path (str): Path to the FAISS index directory.
        book_paths (List[str]): List of paths to the book files.
        model_name (str): Model type ('mistral', 'openai', or 'hf').
        n_workers (int): Number of parallel workers.
        batch_size (int): Number of chunks to process in each batch.
    """
    document_to_rag_pipeline(book_paths, model_name=model_name, faiss_path=faiss_path, n_workers=n_workers, batch_size=batch_size)


def clean_db(faiss_path: str):
    """
    Clean the FAISS index by removing all files in the specified directory.

    Parameters:
        faiss_path (str): Path to the FAISS index directory.
    """
    if os.path.exists(faiss_path):
        for file in os.listdir(faiss_path):
            file_path = os.path.join(faiss_path, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"Directory {faiss_path} does not exist.")