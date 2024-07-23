from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from openai import OpenAI
import tiktoken
from pinecone import Pinecone, ServerlessSpec
import time
import os
import traceback
import logging
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

# Access your API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = "9c097a58-6008-409a-859a-668a002320f6"
INDEX_NAME = "gradient-cyber"
BATCH_SIZE = 100
MAX_RESULTS = 1000

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index already exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Get the index
pinecone_index = pc.Index(INDEX_NAME)

# Define helper functions
def truncate_text(text, max_tokens):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[:max_tokens])

def generate_embedding(text):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding, response.usage.total_tokens
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Error creating embedding after {max_retries} attempts: {str(e)}")
                return None, 0
            time.sleep(2 ** attempt)  # Exponential backoff

def upsert_in_batches(index, vectors, batch_size=100):
    batches = [vectors[i:i + batch_size] for i in range(0, len(vectors), batch_size)]
    for batch in batches:
        try:
            index.upsert(vectors=batch, namespace="ns1")
        except Exception as e:
            logging.error(f"Error upserting batch: {e}")

def semantic_similarity(text1, text2):
    embedding1, _ = generate_embedding(text1)
    embedding2, _ = generate_embedding(text2)
    if embedding1 is None or embedding2 is None:
        return 0
    return sum(a*b for a, b in zip(embedding1, embedding2))

def expand_query(original_query):
    try:
        expansion_prompt = f"Expand the following query into 3-5 related questions or terms: '{original_query}'"
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": expansion_prompt}],
            max_tokens=100,
            temperature=0.7
        )
        return original_query + " " + response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in query expansion: {str(e)}")
        return original_query  # Return original query if expansion fails

def truncate_context(context, max_tokens=14000):
    encoding = tiktoken.get_encoding("cl100k_base")
    encoded = encoding.encode(context)
    truncated = encoded[:max_tokens]
    return encoding.decode(truncated)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')

@app.route('/script.js')
def script():
    return send_from_directory('.', 'script.js')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        data = request.json
        df = pd.DataFrame(data)
        
        def create_meaningful_sentence(row):
            return '. '.join([f"{col.replace('_', ' ')}: {row[col]}" for col in df.columns])

        df['combined_text'] = df.apply(create_meaningful_sentence, axis=1)

        vectors = []
        total_tokens_used = 0
        total_requests = 0
        for i, row in df.iterrows():
            text = row['combined_text']
            text = truncate_text(text, max_tokens=8192)
            embedding, tokens_used = generate_embedding(text)
            if embedding is not None:
                total_tokens_used += tokens_used
                total_requests += 1
                
                def truncate_field(field, max_length=500):
                    return str(field)[:max_length] if not pd.isna(field) else ''

                metadata = {col: truncate_field(row.get(col, '')) for col in df.columns}
                metadata['combined_text'] = text

                vectors.append({'id': str(row['ID']), 'values': embedding, 'metadata': metadata})

        if vectors:
            upsert_in_batches(pinecone_index, vectors, BATCH_SIZE)
            return jsonify({
                "message": "Data uploaded successfully",
                "totalTokensUsed": total_tokens_used,
                "totalRequests": total_requests
            })
        else:
            return jsonify({"error": "No embeddings were generated"}), 400
    except Exception as e:
        logging.error(f"Error in upload_file: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def process_query():
    try:
        query = request.json['query']
        logging.info(f"Received query: {query}")
        
        expanded_query = expand_query(query)
        logging.info(f"Expanded query: {expanded_query}")
        
        query_embedding, tokens_used = generate_embedding(expanded_query)
        if query_embedding is None:
            logging.error("Failed to generate query embedding")
            return jsonify({"error": "Failed to generate query embedding"}), 400
        
        logging.info(f"Generated embedding, tokens used: {tokens_used}")
        
        results = pinecone_index.query(
            namespace="ns1",
            vector=query_embedding,
            top_k=50,
            include_metadata=True
        )

        filtered_results = sorted(
                results['matches'],
                key=lambda x: semantic_similarity(query, x['metadata']['combined_text']),
                reverse=True
            )[:10]

        context = "\n".join([
                f"ID: {match['id']}\n" +
                f"Event Date/Time: {match['metadata'].get('eventDtgTime', 'N/A')}\n" +
                f"Display Title: {match['metadata'].get('displayTitle', 'N/A')}\n" +
                f"Status: {match['metadata'].get('status', 'N/A')}\n" +
                f"Combined Text: {match['metadata'].get('combined_text', 'N/A')}\n" +
                "---"
                for match in filtered_results
            ])

        truncated_context = truncate_context(context)

        system_prompt = """You are an AI assistant specializing in analyzing SITREP data. Provide a comprehensive answer based on the given context. Focus on key patterns, trends, and relevant details. If information is missing, state what is known and what is uncertain."""
        user_prompt = f"""Query: {query}
    Relevant Information:
    {truncated_context}
    Provide a clear, concise, and comprehensive answer. Synthesize information from multiple entries if necessary. Cite specific details and examples when applicable. If information is missing, state what is known and what remains uncertain."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        logging.error(f"Error in process_query: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {str(e)}")
    logging.error(traceback.format_exc())
    return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)