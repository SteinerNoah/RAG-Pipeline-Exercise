#[0] Download all necessary libraries
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

#[1] Loading text from different file types and iterating source files
DATA_DOCUMENTS = Path("data") #Path object that points towards the data folder with the documents to reference
def return_file_string(path: Path) -> str:
    """Reads a single supported file and returns its text as a string"""
    suffix = path.suffix.lower() #takes the suffix of the file (e.g. .pdf or .txt) converts it to lowercase (so a .PDF and a .pdf are the same)
    #Plain text files (.txt) and markdown files(.md) are read directly (e.g. without additional library)
    if suffix in {".txt", ".md"}: #Reads the file using UTF-8 encoding, and skips errors (characters it cannot encode) instead of crashing
        return path.read_text(encoding="utf-8", errors="ignore")
    #PDF files (.pdf) are read via a library (PdfReader)
    if suffix == ".pdf":
        reader = PdfReader(str(path)) #Creates a PDF reader object that converts the path object into a string
        pages = [] #Stores the string text from each page of the PDF in a list
        for page in reader.pages: #Repeats for each page in the PDF
            text = page.extract_text() or "" #Extracts text from the page or adds an empty string if extract_text() returns None
            if text.strip(): #Removes whitespace (spaces, tabs, newlines)
                pages.append(text) #Only appends text if pages contain text (e.g. if text.strip() != "")
        return "\n".join(pages) #Joins each page text in pages into a single string seperated by newlines (enter)
    #Word files (.docx) are read with a reader (DocxDocument)
    if suffix == ".docx":
        doc = DocxDocument(str(path)) #Converts the path object into a string and loads the Word document
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()] #Extracts text from each paragraph that contains text (p.text.strip() != "")
        return "\n".join(paragraphs) #Joins each page text in pages into a single string seperated by newlines (enter)
    #Non-supported files will not be read (return None) and will print a warning
    print(f"Nicht unterstützter Dateityp wird übersprungen ({path.name})")
    return None
    #Could also raise a ValueError: raise ValueError(f"WARNING: Unsupported file type ({path.name})"
def find_supported_files(data_documents: Path) -> list[Path]:
    """Identifies the documents that are supported file types and returns their paths"""
    allowed_suffixes = {".txt", ".md", ".pdf", ".docx"} #Lists the accepted file types by suffix
    files = [path for path in data_documents.rglob("*") if path.is_file() and path.suffix.lower() in allowed_suffixes]
    #Recursively searches all files and subfolders for files (not directories) that are acceptable file types
    return sorted(files) #Returns a list of files sorted alphabetically for consistency in reruns

#[2] Splitting / chunking documents
CHUNK_SIZE = 900 #Defines the maximum number of characters per chunk
CHUNK_OVERLAP = 150 #Defines the number of characters that overlap (repeat) between consecutive chunks to preserve context across their boundries
def chunk_text(text: str) -> list[str]:
    """Returns a list of smaller, overlapping text string chunks from a larger text string"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    #Creates splitter object that splits text strings first by paragraph (/n/n), then line break (/n), sentences, and finally characters when necessary
    chunks = splitter.split_text(text) #Creates list of strings, with one string for each chunk
    return chunks

#[3] Local embedding model for documents
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" #Defines the embedding model used
def create_embedding_model() -> SentenceTransformer:
    """Creates an embedding model"""
    return SentenceTransformer(EMBEDDING_MODEL_NAME) 
    #Downloads the model from SentenceTransformer in the first run, and loads from local cash in any later runs
def embed_text(model: SentenceTransformer, texts: Sequence[str]) -> list[list[float]]:
    """Converts a list of text strings to a list of vector. Each vector is a lists of floats"""
    vectors = model.encode(list(texts), normalize_embeddings=True)
    #Makes sure texts is a list, and converts each of its text strings to a numerical vector
    #Each vector is normalized to unit length so that only direction (not magnitude) are relevant, since ChromaDB uses cosine similarity (depends on direction)
    return vectors.tolist() #Converts the numpy array output from model.encode() to a list of lists since ChromaDB expects plain python data

#[4] Create vector database using ChromaDB
CHROMA_DB_FOLDER = Path("chroma_db") #Defines the folder on the disk where ChromaDB will store the database files
CHROMA_DB_COLLECTION = "rag_dokumente" #Name of ChromaDB collection inside the database
TOP_K = 4 #Number of chunks to retrieve as context for each question
def build_vector_database(model: SentenceTransformer):
    """Builds a new vector database each run. It converts each document to a text string, splits the text string into chunks, and embeds the text string chunks"""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_FOLDER)) #Creates 'persistent' ChromaDB client (database is saved on disk, not only in memory)
    #Restarts each run to prevent adding the same chunk twice (duplicating) in reruns
    try: #tries to delete the old collection if it exists so it can be cleanly rebuilt each run
        client.delete_collection(CHROMA_DB_COLLECTION)
    except Exception:
        pass #if an error occurs (e.g. an old collection does not exist) it does nothing
    collection = client.create_collection(name=CHROMA_DB_COLLECTION, metadata={"hnsw:space": "cosine"})
    #Creates a new ChromaDB collection using cosine distance (similarity)
    #Using cosine because similar texts would probably have similar vector directions so works well for embeddings
    files = find_supported_files(DATA_DOCUMENTS) #finds all files that are a supported file type in the data folder
    if not files: #stops early if no supported files are found. Rest of the pipeline needs input documents to work
        raise RuntimeError(f"Keine unterstützten Dateien gefunden in ({DATA_DOCUMENTS})")
    all_ids: list[str] = [] #Stores the unique ID string for each chunk in a list
    all_texts: list[str] = [] #Stores the corresponding chunk text strings in a list
    all_metadatas: list[dict[str, object]] = [] #Stores the metadatas (file name, chunk index, etc) for the corresponding chunks in a list 
    for file_path in files: #Repeats for each document file in data folder
        raw_text = return_file_string(file_path) #Reads the file and returns its text as a string (using return_file_string() function)
        chunks = chunk_text(raw_text) #Returns a list of smaller, overlapping text string chunks from the file's larger text string from raw_text (using chunk_text() function)
        for index, chunk in enumerate(chunks): #Gives each chunk in chunks a numeric index (0,1,...)
            relative_name = file_path.relative_to(DATA_DOCUMENTS).as_posix().replace("/", "_")
            #Returns the file path relative to the data folder
            #Converts it to a path with forward slashes
            #Replaces those slashes with underscores so it can be used in an ID
            chunk_id = f"{relative_name}_{index}" #Creates a unique ID for the chunk using the file name and index number
            all_ids.append(chunk_id) #Adds the chunk ID to the list of IDs
            all_texts.append(chunk) #Adds the chunk text string to the list of chunk text strings
            all_metadatas.append({"source": file_path.name, "path": str(file_path), "chunk": index}) 
            #Adds the chunk metadata to the list of chunk metadata (source is the file name, path is the file path, chunk is the index number)
    if not all_texts: #Raises RunTimeError if files are empty or the text extraction fails
        raise RuntimeError("Aus den Dokumenten konnte kein Text entnommen werden")
    embeddings = embed_text(model, all_texts) #Converts the list of chunks to a list of vectors (using embed_text() function)
    collection.add(ids=all_ids, documents=all_texts, metadatas=all_metadatas, embeddings=embeddings)
    #Adds all the IDs, texts, metadatas, and embeddings to the ChromaDB collection
    #Could print a summary to confirm indexing worked: print(f"Indexed {len(all_texts)} chunks from {len(files)} files.")
    return collection #Returns the ChromaDB collection so that it can be queried later
def find_similar_chunks(collection, model: SentenceTransformer, question: str, top_k: int = TOP_K) -> list[dict[str, object]]:
    """Embeds user query and retrieves chunks that are the most similar to it"""
    query_embedding = embed_text(model, [question])[0]
    #Embeds the query using the same embedding model as the documents
    #Makes a one item list [question] since embed_text() needs a list input
    #Takes the first (and only) embedding (vector) [0] out of returned list to make a single vector (rather than a one item list of vectors)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    #Has ChromaDB find the top_k chunks nearest the question embedding
    #A list of embeddings is expected so it makes a one item list [query_embedding]
    documents = results.get("documents", [[]])[0]
    #Extracts the chunk text strings from the query results
    #results.get("documents", [[]]) safely returns an empty nested list if the dictionary key is missing, 
    #since ChromaDB returns a dictionary with lists of results, which are nested since you can have multiple queries and corresponding results)
    #Takes the first (and only) query result [0] out of returned list to make a single result list (rather than a one item list of result lists)
    metadatas = results.get("metadatas", [[]])[0] #Extracts the chunk metadata for each returned chunk (which file and chunk each result is from)
    distances = results.get("distances", [[]])[0] #Extracts the similarity distances for each chunk (smaller usually means closer match)
    matches: list[dict[str, object]] = [] #Prepares a list of dictionaries with string keys (documents, metadatas, distances) and object values (anything) 
    for doc, meta, distance in zip(documents, metadatas, distances): #Pairs items from the three lists by index
        matches.append({"text": doc, "metadata": meta, "distance": distance}) #Adds a retrieved result dictionary to the matches list
    return matches #Returns the matches list of retrieved chunks so it can be used as context by the LLM later

#[5] Answer user questions using a chat model
CHAT_MODEL_NAME = "openrouter/free" #Defines a free OpenRouter chat model
def answer_question(question: str, chunks: list[dict[str, object]]) -> str:
    """Sends the question as a string of text and the list of dictionaries corresponding with the retrieved document chunks to the chat model to return an answer"""
    if not chunks: #Stops early if the retrieval is empty (no documents) to avoid sending the chat model empty context
        return "Ich konnte in den Dokumenten keinen passenden Inhalt finden."
    context_parts = [] #Lists the formatted chunk from the retrieved chunks 
    for item in chunks: #Repeats for each chunk in chunks
        meta = item["metadata"] #Retrieves the metadata from the dictionary of each chunk (source file and index number)
        context_parts.append(f"Quelle: {meta['source']} | Chunk {meta['chunk']}\n{item['text']}")
        #Creates a readable text block with the name, index number, and, after a newline, the text string of the source file
    context = "\n\n---\n\n".join(context_parts) 
    #Combines the chunks onto one long string, divided into into paragraphs by two double newlines (/n/n), seperated by a visual divider (---) 
    model = ChatOpenRouter(model=CHAT_MODEL_NAME, temperature=0)
    #Creates a chat model object with the chosen model and a low output randomness (low temperature means more deterministic outputs from the model)
    messages = [{"role": "system", "content": #Defines the system message as a string (rules and behaviours of the model to reduce hallucination)
                ("Du bist ein hilfreicher Assistent. "
                "Antworte in der Sprache der Nutzerfrage. "
                "Nutze nur den gegebenen Kontext. "
                "Wenn der Kontext nicht reicht, sage das ehrlich.")},
        {"role": "user", "content": f"Frage:\n{question}\n\nKontext:\n{context}"},] #Defines the user query and the context (retrieved chunks), seperated by newlines (/n)
    response = model.invoke(messages) #Sends the messages (prompt) to the chat model, which returns a response
    return response.content #Returns just the text part of the model response
def main() -> None:
    """Runs the full RAG pipeline from the command line"""
    """Loads enviromental variables, checks API key, reads the query, embeds data and builds vector database, retrieves relevant chunks, and generates response"""
    load_dotenv() #Loads variables from .env file into the python process environment (e.g. the OPENROUTER_API_KEY necessary to use the OpenRouter chat models)
    if not os.getenv("OPENROUTER_API_KEY"): #If the OpenRouter API key is missing, os.getenv() returns None and a RuntimeError is raised
        raise RuntimeError("Die Variable OPENROUTER_API_KEY fehlt. Fügen Sie sie Ihrer .env Datei hinzu.")
    question = " ".join(sys.argv[1:]).strip() 
    #The query is passed as a command line argument (sys.argv), where sys.argv[0] is the script name and sys.argv[1:] is the question (all subsequent words)
    #They are combined into one string, seperated by spaces, with " ".join(), and have any leading or trailing spaces removed with .strip()
    if not question: #If a command-line question is not provided (e.g. question is empty), the user is prompted
        question = input("Stell eine Frage zu den Dokumenten: ").strip() #input() waits for the user to type their query into the terminal
    if not question: #If the user still does not provide a question (type something into the terminal), a RuntimeError is raised
        raise RuntimeError("No question was provided.") #Prevents running pipeline without an empty input
    embedding_model = create_embedding_model() #Loads the embedding model for the document chunks and the query
    collection = build_vector_database(embedding_model) #Reads the documents, splits them into chunks, embeds the chunks, and stores them in the ChromaDB database
    chunks = find_similar_chunks(collection, embedding_model, question) #Embeds the user question and finds the most similiar chunks in the vector database
    print("\nGefundene Stellen:") #Prints a header to show which chunks were chosen and used
    for item in chunks: #Repeats for each chunk in the list of chosen chunks
        meta = item["metadata"] #Extracts the chunk metadata
        print(f"- {meta['source']} (Chunk {meta['chunk']}, Distanz {item['distance']:.4f})") 
        #Prints the source file, its index number, and its similarity distance (to 4 decimal places .4f)
    print("\nAntwort:\n") #Prints a header before the final model answer
    print(answer_question(question, chunks)) #Calls the anser_question() function and prints the chat model's final response
if __name__ == "__main__": #Makes sure that main() is run only if the file is executed directly
    main() #Prevents main from running automatically if the file is imported from somewhere else