import os
import sys
from pathlib import Path
from typing import Sequence
import chromadb
from docx import Document as DocxDocument
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DOCUMENTS = Path("data")

def return_file_string(path: Path) -> str:
    """Reads a single supported file and returns its text as a string"""
    suffix = path.suffix.lower()
    
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    
    if suffix == ".pdf":
        reader = PdfReader(str(path)) 
        pages = [] 
        for page in reader.pages: 
            text = page.extract_text() or ""
            if text.strip(): 
                pages.append(text) 
        return "\n".join(pages)

    if suffix == ".docx":
        doc = DocxDocument(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    print(f"Nicht unterstützter Dateityp wird übersprungen ({path.name})")
    return None

def find_supported_files(data_documents: Path) -> list[Path]:
    """Identifies the documents that are supported file types and returns their paths"""
    
    allowed_suffixes = {".txt", ".md", ".pdf", ".docx"}
    files = [path for path in data_documents.rglob("*") if path.is_file() and path.suffix.lower() in allowed_suffixes]

    return sorted(files)


CHUNK_SIZE = 900
CHUNK_OVERLAP = 150 

def chunk_text(text: str) -> list[str]:
    """Returns a list of smaller, overlapping text string chunks from a larger text string"""
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)
    
    return chunks


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_embedding_model() -> SentenceTransformer:
    """Creates an embedding model"""
    return SentenceTransformer(EMBEDDING_MODEL_NAME) 

def embed_text(model: SentenceTransformer, texts: Sequence[str]) -> list[list[float]]:
    """Converts a list of text strings to a list of vector. Each vector is a lists of floats"""
    
    vectors = model.encode(list(texts), normalize_embeddings=True)

    return vectors.tolist()


CHROMA_DB_FOLDER = Path("chroma_db")
CHROMA_DB_COLLECTION = "rag_dokumente"
TOP_K = 4

def build_vector_database(model: SentenceTransformer):
    """Builds a new vector database each run. It converts each document to a text string, splits the text string into chunks, and embeds the text string chunks"""
    
    client = chromadb.PersistentClient(path=str(CHROMA_DB_FOLDER)) 
    
    try: 
        client.delete_collection(CHROMA_DB_COLLECTION)
    except Exception:
        pass
    
    collection = client.create_collection(name=CHROMA_DB_COLLECTION, metadata={"hnsw:space": "cosine"})
    files = find_supported_files(DATA_DOCUMENTS) 
    
    if not files: 
        raise RuntimeError(f"Keine unterstützten Dateien gefunden in ({DATA_DOCUMENTS})")
    
    all_ids: list[str] = [] 
    all_texts: list[str] = [] 
    all_metadatas: list[dict[str, object]] = [] 
    
    for file_path in files: 
        raw_text = return_file_string(file_path) 
        chunks = chunk_text(raw_text) 
        for index, chunk in enumerate(chunks): 
            relative_name = file_path.relative_to(DATA_DOCUMENTS).as_posix().replace("/", "_")
            chunk_id = f"{relative_name}_{index}"
            all_ids.append(chunk_id) 
            all_texts.append(chunk) 
            all_metadatas.append({"source": file_path.name, "path": str(file_path), "chunk": index}) 

    if not all_texts: 
        raise RuntimeError("Aus den Dokumenten konnte kein Text entnommen werden")
    embeddings = embed_text(model, all_texts) 
    collection.add(ids=all_ids, documents=all_texts, metadatas=all_metadatas, embeddings=embeddings)

    return collection 

def find_similar_chunks(collection, model: SentenceTransformer, question: str, top_k: int = TOP_K) -> list[dict[str, object]]:
    """Embeds user query and retrieves chunks that are the most similar to it"""
    
    query_embedding = embed_text(model, [question])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0] 
    distances = results.get("distances", [[]])[0] 
    matches: list[dict[str, object]] = [] 
    
    for doc, meta, distance in zip(documents, metadatas, distances): 
        matches.append({"text": doc, "metadata": meta, "distance": distance}) 
    
    return matches


CHAT_MODEL_NAME = "openrouter/free" 

def answer_question(question: str, chunks: list[dict[str, object]]) -> str:
    """Sends the question as a string of text and the list of dictionaries corresponding with the retrieved document chunks to the chat model to return an answer"""
    
    if not chunks: 
        return "Ich konnte in den Dokumenten keinen passenden Inhalt finden."
    
    context_parts = []
    
    for item in chunks: 
        meta = item["metadata"]
        context_parts.append(f"Quelle: {meta['source']} | Chunk {meta['chunk']}\n{item['text']}")

    context = "\n\n---\n\n".join(context_parts) 
    model = ChatOpenRouter(model=CHAT_MODEL_NAME, temperature=0)
    messages = [{"role": "system", "content": 
                ("Du bist ein hilfreicher Assistent. "
                "Antworte in der Sprache der Nutzerfrage. "
                "Nutze nur den gegebenen Kontext. "
                "Wenn der Kontext nicht reicht, sage das ehrlich.")},
        {"role": "user", "content": f"Frage:\n{question}\n\nKontext:\n{context}"},] 
    response = model.invoke(messages) 
    
    return response.content 

def main() -> None:
    """Runs the full RAG pipeline from the command line"""
    """Loads enviromental variables, checks API key, reads the query, embeds data and builds vector database, retrieves relevant chunks, and generates response"""
    
    load_dotenv() 
    
    if not os.getenv("OPENROUTER_API_KEY"): 
        raise RuntimeError("Die Variable OPENROUTER_API_KEY fehlt. Fügen Sie sie Ihrer .env Datei hinzu.")
    
    question = " ".join(sys.argv[1:]).strip() 

    if not question:
        question = input("Stell eine Frage zu den Dokumenten: ").strip() 
    
    if not question:
        raise RuntimeError("No question was provided.") 
    
    embedding_model = create_embedding_model()
    collection = build_vector_database(embedding_model)
    chunks = find_similar_chunks(collection, embedding_model, question)
    
    print("\nGefundene Stellen:") 
    
    for item in chunks: 
        meta = item["metadata"]
        print(f"- {meta['source']} (Chunk {meta['chunk']}, Distanz {item['distance']:.4f})") 
        
    print("\nAntwort:\n") 
    print(answer_question(question, chunks)) 

if __name__ == "__main__": 
    main()