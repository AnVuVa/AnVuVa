# ================== CONFIGURATION ==================
import os
#os.environ["OPENAI_API_KEY"] = "your-openai-key"
# os.environ["MISTRAL_API_KEY"] =

NEO4J_URI = "neo4j_url"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

from typing import List
from tqdm import tqdm
import concurrent.futures
import numpy as np
import time

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import MistralConfig, ChatGPTConfig, OllamaConfig
from camel.loaders import UnstructuredIO
from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ================== FUNCTION: Get Model ==================
def get_model(model_name: str = "mistral") -> ModelFactory:
        """
        Retrieve the appropriate model based on the user's choice.

        Returns:
            Model: The initialized model.
        """
        if "gpt" in model_name:
            return ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
                model_config_dict=ChatGPTConfig().as_dict(),
            )
        elif "mistral" in model_name:
            return ModelFactory.create(
                model_platform=ModelPlatformType.MISTRAL,
                model_type=ModelType.MISTRAL_LARGE,
                model_config_dict=MistralConfig(temperature=0.2).as_dict(),
            )
        else:
            return ModelFactory.create(
                model_platform=ModelPlatformType.OLLAMA,
                model_type=model_name,
                model_config_dict=OllamaConfig(temperature=0.2, max_tokens=2048).as_dict(),
            )


# ================== FUNCTION: Load Document ==================
def load_document(file_path: str) -> str:
    """
    Load raw text content from a file using UnstructuredIO.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        str: Combined text content.
    """
    loader = UnstructuredLoader(file_path)
    elements = loader.load()
    content = ("\n\n".join([el.page_content for el in elements]))
    return content


# ================== FUNCTION: Semantic Chunking ==================

def semantic_chunking(chunks: List[str], max_length: int = 1024, threshold: float = 0.8, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[str]:
    """
    Reduce the number of chunks by merging semantically related chunks using HuggingFaceEmbedding.

    Parameters:
        chunks (List[str]): List of text chunks.
        max_length (int): Maximum length of a single chunk after merging.
        threshold (float): Cosine similarity threshold for merging chunks.
        model_name (str): Hugging Face model name for embedding.

    Returns:
        List[str]: List of semantically merged chunks.
    """
    # Initialize HuggingFaceEmbedding
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)

    schunks = []  # List to store semantically merged chunks

    for chunk in chunks:
        if not schunks:
            # Add the first chunk to schunks
            schunks.append(chunk)
        else:
            # Get the last chunk in schunks
            last_chunk = schunks[-1]
            combined_chunk = last_chunk + " " + chunk

            # Check if the combined length exceeds max_length
            if len(combined_chunk) > max_length:
                schunks.append(chunk)
                continue

            # Calculate embeddings and cosine similarity
            vec1 = embedding_model.embed_query(last_chunk)
            vec2 = embedding_model.embed_query(combined_chunk)
            similarity = cosine_similarity(vec1, vec2)

            if similarity >= threshold:
                # Merge the chunks if similarity is above the threshold
                schunks[-1] = combined_chunk
            else:
                # Add the current chunk as a new chunk in schunks
                schunks.append(chunk)

    print(len(chunks), "-->", len(schunks))  # Debugging: Print the reduction in chunks
    return schunks


# ================== FUNCTION: Chunk Text ==================
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split raw text into overlapping chunks.

    Parameters:
        text (str): Raw input text.
        chunk_size (int): Length of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_text(text)
    return semantic_chunking(chunks)


# ================== FUNCTION: Create Element ==================
def create_element(chunk: str, chunk_id: str) -> dict:
    """
    Wrap chunk into a CAMEL-compatible element.

    Parameters:
        chunk (str): Text chunk.
        chunk_id (str): Unique ID for the chunk.

    Returns:
        dict: Element dictionary.
    """
    uio = UnstructuredIO()
    return uio.create_element_from_text(text=chunk, element_id=chunk_id)


# ================== FUNCTION: Process Single Chunk ==================
def process_chunk(idx: int, chunk: str, model) -> dict:
    """
    Process a single chunk to extract and return graph elements.

    Parameters:
        idx (int): Chunk index.
        chunk (str): Chunk text.
        model_name (str): Model type ('mistral' or 'openai').

    Returns:
        dict: Graph element.
    """
    element = create_element(chunk, f"chunk-{idx}")
    agent = KnowledgeGraphAgent(model=model)
    graph_data = agent.run(element, parse_graph_elements=True)
    return graph_data


# ================== FUNCTION: Extract + Push All Chunks ==================
def extract_and_push_graph_elements(chunks: List[str], extract_model:str, n_workers: int = 1, gid: int = 1):
    """
    Convert all chunks to graphs and upload to Neo4j.

    Parameters:
        chunks (List[str]): List of text chunks.
        model_name (str): Model name to use.
        n_workers (int): Number of parallel workers.
        gid (int): Graph ID to associate with the graph elements.
    """
    n4j = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

    if n_workers == 1:
        model = get_model(extract_model)
        for idx, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Processing chunks"):
            while True:  # Retry loop
                try:
                    graph_data = process_chunk(idx, chunk, model)
                    
                    # Add gid to each node in graph_data
                    if hasattr(graph_data, "nodes"):
                        for node in graph_data.nodes:
                            if hasattr(node, "properties"):
                                node.properties["gid"] = gid  # Add gid to the node's properties
                            else:
                                raise AttributeError("Node object does not have a 'properties' attribute.")
                    else:
                        raise AttributeError("GraphElement object does not have 'nodes' attribute.")
                    
                    # Push to Neo4j
                    n4j.add_graph_elements(graph_elements=[graph_data])
                    break  # Exit the retry loop if successful
                except Exception as e:
                    if "rate limit" in str(e).lower():  # Check if the error is related to rate limiting
                        # print("[Rate Limit] Waiting for 1 second before retrying...")
                        time.sleep(2)  # Wait for 1 second
                    else:
                        raise  # Re-raise other exceptions
    else:
        # Threaded execution for parallel chunk processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            model = get_model(extract_model)
            futures = {
                executor.submit(process_chunk, idx, chunk, model): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing in parallel"):
                while True:  # Retry loop
                    try:
                        graph_data = future.result()
                        
                        # Add gid to each node in graph_data
                        for node in graph_data.nodes:
                            node.properties["gid"] = gid  # Add gid to the node's properties
                        
                        # Push to Neo4j
                        n4j.add_graph_elements(graph_elements=[graph_data])
                        break  # Exit the retry loop if successful
                    except Exception as e:
                        if "rate limit" in str(e).lower():  # Check if the error is related to rate limiting
                            # print("[Rate Limit] Waiting for 1 second before retrying...")
                            time.sleep(2)  # Wait for 1 second
                        else:
                            raise  # Re-raise other exceptions


# ================== FUNCTION: Shrink Graph ==================
def shrink_graph(gid: int = 1):
    """
    Shrink the graph by merging nodes with the same 'id' property (case-insensitive).

    Parameters:
        gid (int): Graph ID to identify the graph in Neo4j.
    """
    n4j = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

    # Query to find and merge nodes with the same 'id' (case-insensitive)
    merge_query = f"""
    MATCH (n {{gid: {gid}}})
    WITH n.id AS original_id, LOWER(n.id) AS lower_id, COLLECT(n) AS nodes
    WHERE SIZE(nodes) > 1
    WITH nodes, nodes[0] AS target_node
    UNWIND [node IN nodes WHERE node <> target_node] AS dup
    SET target_node += properties(dup)
    WITH dup, target_node
    MATCH (dup)-[r]->(m)
    CALL apoc.create.relationship(target_node, type(r), properties(r), m) YIELD rel AS r_new
    DELETE r
    WITH dup, target_node
    MATCH (m)-[r]->(dup)
    CALL apoc.create.relationship(m, type(r), properties(r), target_node) YIELD rel AS r_new
    DELETE r
    DELETE dup
    RETURN COUNT(target_node) AS merged_nodes
    """
    result = n4j.query(query=merge_query)
    print(f"Number of merged nodes: {result[0]['merged_nodes']}")


# ================== MAIN PIPELINE ==================
def process_document_to_neo4j(file_path: str, extract_model: str = "mistral", n_workers: int = 1, gid: int = 1):  
    """
    End-to-end pipeline: file → chunks → graph → Neo4j

    Parameters:
        file_path (str): Input document path.
        model_name (str): 'mistral' or 'openai'
        n_workers (int): Number of threads (1 = sequential)
    """
    text = load_document(file_path)
    chunks = chunk_text(text)
    print(f"Number of chunks: {len(chunks)}")
    
    extract_and_push_graph_elements(chunks, extract_model, n_workers, gid)


if __name__ == "__main__":
    # Example usage
    file_path = r"E:\\Git_clone\\RAG\\qa_dataset\\data_clean\\textbooks\\en\\Anatomy_Gray.txt"
    model_name = "mistral"
    n_workers = 3
    gid = 1
    
    process_document_to_neo4j(file_path, model_name, n_workers, gid)
    # shrink_graph(gid)

# ================== FUNCTION: Extract Entities ==================
def gretriever(query: str , extract_model:str = "nuextract"):
    """
    Extract entities from a query using the specified model.

    Parameters:
        query (str): Input query.
        model_name (str): 'mistral' or 'openai'
        n_workers (int): Number of threads (1 = sequential)
    """
    # Initialize the model
    model = get_model(extract_model)
    
    agent = KnowledgeGraphAgent(model=model)
    query = create_element(query, "query")
    ans_element = agent.run(query, parse_graph_elements=True)

    n4j = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
    kg_result = []
    for node in ans_element.nodes:
        n4j_query = f"""
    MATCH (n {{id: '{node.id}'}})-[r]->(m)
    RETURN 'Node ' + n.id + ' (label: ' + labels(n)[0] + ') has relationship ' + type(r) + ' with Node ' + m.id + ' (label: ' + labels(m)[0] + ')' AS Description
    UNION
    MATCH (n)<-[r]-(m {{id: '{node.id}'}})
    RETURN 'Node ' + m.id + ' (label: ' + labels(m)[0] + ') has relationship ' + type(r) + ' with Node ' + n.id + ' (label: ' + labels(n)[0] + ')' AS Description
    """
        result = n4j.query(query=n4j_query)
        kg_result.extend(result)

    kg_result = [item['Description'] for item in kg_result]

    kg_result = list(set(kg_result))

    return kg_result[:50]
