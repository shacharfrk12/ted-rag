from create_index import *
from collections import defaultdict
import json
from langchain_openai import ChatOpenAI
from pathlib import Path


def query_index(index,embeddings_model, query_text: str, top_k: int = TOP_K):
    """ Query the Pinecone index with the given text and return the top_k most similar chunks """
    
    query_embedding = embeddings_model.embed_query(query_text)
    
    results = index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True
    )
    
    return results['matches']

def join_query_results(matches):
    talk_chunks = defaultdict(list)

    for match in matches:
        talk_id = match["metadata"]["talk_id"]
        
        # get only relevant metadata
        if talk_id not in talk_chunks:
            metadata = match["metadata"].copy()
            metadata.pop("chunk_transcript", None)
            metadata.pop("chunk_id", None)
            metadata.pop("talk_id", None)
            talk_chunks[talk_id] = {
                "metadata": metadata,
                "transcripts": []
            }
        talk_chunks[talk_id]["transcripts"].append((match["metadata"]["chunk_transcript"], match["score"]))
    return talk_chunks


def build_context_from_matches(talk_chunks, max_chunks_per_talk):
    """
    Build a string context for GPT from retrieved chunks grouped by talk.
    Each chunk includes talk metadata and is sorted by similarity score.
    
    talk_chunks: dict
        {
            talk_id: {
                "metadata": {...},            # talk-level metadata like title, speaker
                "transcripts": [(chunk_text, score), ...]
            },
            ...
        }
    max_chunks_per_talk: int
        Maximum number of chunks to include per talk
    """
    context = ""
    
    for talk_id, data in talk_chunks.items():
        metadata = data["metadata"]
        context += f"--- TED Talk ---\n"
        context += f"Talk ID: {talk_id}\n"
        # Add ALL metadata fields dynamically
        for key, value in metadata.items():
            context += f"{key}: {value}\n"
        
        # Sort chunks by similarity score descending
        chunks = data["transcripts"]
        sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
        
        # Include top N chunks per talk
        for chunk_text, score in sorted_chunks[:max_chunks_per_talk]:
            
            context += f"Chunk (score: {score:.4f}):\n{chunk_text}\n\n"

    return context


def prepare_context_for_qa(index, embeddings_model, query_text: str, top_k: int, max_chunks_per_talk: int):
    matches = query_index(index, embeddings_model, query_text, top_k)
    talk_chunks = join_query_results(matches)
    context = build_context_from_matches(talk_chunks, max_chunks_per_talk)
    return context, talk_chunks

def run_query_in_model(context, system_prompt_path, user_query, chat_model):
    with open(system_prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    user_prompt = f"Context:\n{context}\n\nQuestion:\n{user_query}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = chat_model.invoke(messages)
    return response.content, system_prompt, user_prompt

def format_rag_response(response, retrieved_chunks, system_prompt, user_prompt):
    """ Format output in the assignment-required JSON structure """
    output = {
        "response": response,
        "context": [
            {
                "talk_id": talk_id,
                "title": data["metadata"].get("title", "Unknown Title"),
                "chunk": chunk_text,
                "score": score
            }
            for talk_id, data in retrieved_chunks.items()
            for chunk_text, score in data["transcripts"][:MAX_CHUNKS_PER_TALK]
        ],
        "Augmented_prompt": {
            "System": system_prompt,
            "User": user_prompt
        }
    }
    output_json = json.dumps(output, indent=4)  # indent makes it pretty-printed

    return output_json

def full_query_pipeline(query_data, index, embeddings_model, system_prompt_path, chat_model) -> json:
    """ Run the entire RAG pipeline from JSON question to formatted output """

    user_query = query_data["question"]

    # retrieve context
    context_str, retrieved_chunks = prepare_context_for_qa(index, embeddings_model, user_query, TOP_K, MAX_CHUNKS_PER_TALK)

    # run the model
    response_text, system_prompt, user_prompt = run_query_in_model(context_str, system_prompt_path, user_query, chat_model)

    # format the final output
    output_json = format_rag_response(response_text, retrieved_chunks, system_prompt, user_prompt)

    return output_json

def full_query_pipeline_from_path(query_json_path, index, embeddings_model, system_prompt_path, chat_model, save_path):
    """ Run the entire RAG pipeline from JSON question to formatted output """
    # load query
    with open(query_json_path, 'r') as f:
        query_data = json.load(f)
    
    output_json = full_query_pipeline(query_data, index, embeddings_model, system_prompt_path, chat_model)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    return output_json


def multiple_queries_pipeline(query_json_dir, index, embeddings_model, system_prompt_path, chat_model, save_dir):
    
    folder_path = Path(query_json_dir)

    json_files = list(folder_path.glob("*.json"))

    for query_json_path in json_files:
        save_path = Path(save_dir) / Path(query_json_path).name
        full_query_pipeline_from_path(query_json_path, index, embeddings_model, system_prompt_path, chat_model, save_path)



def main():
    index = get_vector_index(PINECONE_INDEX)
    embeddings_model = OpenAIEmbeddings(
        api_key=LLMOD_API_KEY, 
        base_url=LLMOD_URL,
        model=EMBEDDING_MODEL
    )
    chat_model = ChatOpenAI(
        model_name=LLM_MODEL,
        api_key=LLMOD_API_KEY,      
        base_url=LLMOD_URL
    )
    multiple_queries_pipeline("queries", index, embeddings_model, "system_prompt.text", chat_model, "results")

if __name__ == "__main__":
    main()