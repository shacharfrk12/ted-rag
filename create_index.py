import pandas as pd 
from Constants import *
import os
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm


## this code will use the embedding model to create the vector index 

# TODO: change to not small when finishing
load_dotenv('.env.local')
LLMOD_API_KEY = os.getenv("LLMOD_API_KEY") 
PINECONE_INDEX_SMALL = os.getenv("PINECONE_INDEX_SMALL")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LLMOD_URL = os.getenv("LLMOD_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
LLMOD_URL = os.getenv("LLMOD_URL")

def create_small_csv(path_to_big_csv:str, path_to_small_csv:str, num_rows:int):
    """ create a small csv file from the big csv file for testing purposes """
    df = pd.read_csv(path_to_big_csv)
    print(len(df))
    small_df = df.head(num_rows)
    small_df.to_csv(path_to_small_csv, index=False)

def chunk_one_script(script: str, chunk_size: int = CHUNK_SIZE, overlap_ratio: float = OVERLAP_RATIO):
    """ chunk one script into smaller chunks with overlap """
    chunks = []
    start = 0
    script_length = len(script)
    overlap = int(chunk_size * overlap_ratio)
    
    while start < script_length:
        end = min(start + chunk_size, script_length)
        chunk = script[start:end]
        chunks.append(chunk)
        
        if end == script_length:
            break
        
        start += chunk_size - overlap
    
    return chunks

def create_all_chunks(df: pd.DataFrame, chunk_size: int = CHUNK_SIZE, overlap_ratio: int = OVERLAP_RATIO):
    """ create chunks for all scripts """
    
    rows = []
    
    for _, row in df.iterrows():
        script = row['transcript']
        talk_id = row['talk_id']
        chunks = chunk_one_script(script, chunk_size, overlap_ratio)
        
        for i, chunk in enumerate(chunks):
            rows.append({
                'talk_id': talk_id,
                'chunk_id': i + 1,  # serial number starting from 1
                'chunk_transcript': chunk
            })
    
    chunks_df = pd.DataFrame(rows)
    # df.drop(columns=['transcript'], inplace=True)
    # merged_df = pd.merge(chunks_df, df, on='talk_id', how='left')
    return chunks_df
    
# def add_chunk_embedding(chunks_df: pd.DataFrame):

#     chunks_df['embedding'] = chunks_df['chunk_transcript'].apply(lambda x: embeddings_model.embed_query(x))

#     return chunks_df

def embed_one_batch(chunk_transcripts: list, embedding_model):

    embeddings = embedding_model.embed_documents(
        chunk_transcripts
    )
    return embeddings


def get_vector_index(index_name):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    return index

# def add_chunks_to_index(index, chunks_df: pd.DataFrame, og_df_path: str):
#     og_df = pd.read_csv(og_df_path).drop(columns=['transcript'])
#     metadata_cols= og_df.columns.tolist() + ['chunk_id', 'chunk_transcript']
#     all_df = chunks_df.merge(og_df, on='talk_id', how='left')
#     all_df["vec_id"] = all_df.apply(lambda row: f"{row['talk_id']}_chunk_{row['chunk_id']}", axis=1)
#     n = len(all_df)
#     n_batches = (n + batch_size - 1) // batch_size  # ceil division

#     for _, row in tqdm(all_df.iterrows(), total=len(all_df), desc="Upserting vectors"):
#         vector_id = row["vec_id"]
#         vector = row['embedding']
#         metadata = {col: row[col] for col in metadata_cols}
            
#         index.upsert(vectors=[(vector_id, vector, metadata)])

def add_chunks_to_index_batch(index,embedding_model, og_df_path: str, batch_size):
    og_df = pd.read_csv(og_df_path).drop(columns = METADATA_COLS_TO_DROP)
    metadata_cols = og_df.columns.tolist() + ['chunk_id', 'chunk_transcript']
    metadata_cols.remove('transcript')
    # all_df = chunks_df_no_emb.merge(og_df, on='talk_id', how='left')
    # all_df["vec_id"] = all_df.apply(lambda row: f"{row['talk_id']}_chunk_{row['chunk_id']}", axis=1)
    
    # Number of batches
    n = len(og_df)
    n_batches = (n + batch_size - 1) // batch_size  # ceil division

    for i in tqdm(range(n_batches), desc="Upserting vectors"):
        batch_df = og_df.iloc[i * batch_size : (i + 1) * batch_size].copy()

        chunks_df = create_all_chunks(batch_df, CHUNK_SIZE, OVERLAP_RATIO)

        batch_df = batch_df.drop(columns=['transcript'])
        full_chunks_df = chunks_df.merge(batch_df)
        full_chunks_df["vec_id"] = full_chunks_df.apply(lambda row: f"{row['talk_id']}_chunk_{row['chunk_id']}", axis=1)
    
        chunk_ids = full_chunks_df["vec_id"].tolist()
        chunk_transcripts = full_chunks_df["chunk_transcript"].tolist()
        chunk_metadata = (
            full_chunks_df[metadata_cols]
            .where(pd.notnull(full_chunks_df[metadata_cols]), "missing")
            .to_dict(orient="records")
        )

        # Embed the batch
        chunk_embeddings = embed_one_batch(chunk_transcripts, embedding_model)


        n_upsert = len(chunk_embeddings)
        n_batches_upsert = (n_upsert + BATCH_SIZE_UPSERT - 1) // BATCH_SIZE_UPSERT  # ceil division
        for j in range(n_batches_upsert):
            # Prepare vectors and upsert
            start = j * BATCH_SIZE_UPSERT
            end = (j + 1) * BATCH_SIZE_UPSERT
            vectors_batch = list(zip(chunk_ids[start: end], chunk_embeddings[start: end], chunk_metadata[start: end]))
            index.upsert(vectors=vectors_batch)
            

# def run_full_pipeline(data_csv_path: str, og_df_path: str, index_name: str):
#     chunks_df = create_all_chunks(data_csv_path)
#     chunks_with_embeddings_df = add_chunk_embedding(chunks_df)
#     index = get_vector_index(index_name)
#     add_chunks_to_index(index, chunks_with_embeddings_df, og_df_path)
def empty_index(index):
    index.delete(delete_all=True)

def run_full_pipeline_batch(data_csv_path: str, index_name: str, batch_size):
    embeddings_model = OpenAIEmbeddings(
        api_key=LLMOD_API_KEY,
        base_url=LLMOD_URL,
        model=EMBEDDING_MODEL
    )
    index = get_vector_index(index_name)
    empty_index(index)
    add_chunks_to_index_batch(index, embeddings_model, data_csv_path, batch_size=batch_size)

def main():
    # create_small_csv(BIG_CSV_PATH, SMALL_CSV_PATH, num_rows=LEN_SMALL_CSV)
    run_full_pipeline_batch(BIG_CSV_PATH, PINECONE_INDEX, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    main()

