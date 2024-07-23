from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from openai import OpenAI
import tiktoken
from pinecone import Pinecone, ServerlessSpec
import time
import os
import traceback
import logging
import json

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

from dotenv import load_dotenv
load_dotenv()

# Access your API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = "9c097a58-6008-409a-859a-668a002320f6"
INDEX_NAME = "gradient-cyber"
BATCH_SIZE = 100
MAX_RESULTS = 1000

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

def truncate_context(context, max_tokens=6000):
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

@app.route('/api/query', methods=['POST', 'OPTIONS'])
def process_query():
    if request.method == 'OPTIONS':
        return '', 204
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

        truncated_context = truncate_context(context, max_tokens=6000)

        system_prompt = """You are an AI assistant specializing in analyzing SITREP data. Provide a comprehensive answer based on the given context. Focus on key patterns, trends, and relevant details. If information is missing, state what is known and what is uncertain. Structure your response with the following sections:

1. **Summary**: A brief overview of the main points.
2. **Key Findings**: List the most important discoveries or insights.
3. **Details**: Elaborate on the key findings with specific information from the context.
4. **Uncertainties**: Mention any gaps in the information or areas that require further investigation.
5. **Recommendations**: Suggest next steps or areas for further analysis.

Use markdown formatting to structure your response, including bold text for emphasis and bullet points for lists."""

        user_prompt = f"""Query: {query}
    Relevant Information:
    {truncated_context}
    Provide a clear, concise, and comprehensive answer following the structure outlined in the system prompt. Synthesize information from multiple entries if necessary. Cite specific details and examples when applicable."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Calculate total tokens
        encoding = tiktoken.get_encoding("cl100k_base")
        total_tokens = sum(len(encoding.encode(msg["content"])) for msg in messages)

        # If total tokens exceed the limit, further truncate the context
        if total_tokens > 8000:
            max_context_tokens = 8000 - len(encoding.encode(system_prompt)) - len(encoding.encode(query)) - 100  # 100 tokens buffer
            truncated_context = truncate_context(truncated_context, max_tokens=max_context_tokens)
            user_prompt = f"""Query: {query}
    Relevant Information:
    {truncated_context}
    Provide a clear, concise, and comprehensive answer following the structure outlined in the system prompt. Synthesize information from multiple entries if necessary. Cite specific details and examples when applicable."""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )

        answer = response.choices[0].message.content
        
        # Create an interactive structure
        interactive_answer = {
            "summary": "",
            "key_findings": [],
            "details": "",
            "uncertainties": [],
            "recommendations": []
        }
        
        # Parse the GPT-4 response to fill the interactive structure
        current_section = None
        for line in answer.split('\n'):
            if '**Summary**' in line:
                current_section = "summary"
            elif '**Key Findings**' in line:
                current_section = "key_findings"
            elif '**Details**' in line:
                current_section = "details"
            elif '**Uncertainties**' in line:
                current_section = "uncertainties"
            elif '**Recommendations**' in line:
                current_section = "recommendations"
            elif current_section:
                if current_section in ["key_findings", "uncertainties", "recommendations"]:
                    if line.strip().startswith('- '):
                        interactive_answer[current_section].append(line.strip()[2:])
                else:
                    interactive_answer[current_section] += line + "\n"
        
        logging.info("Successfully processed query and generated response")
        return jsonify({
            "structured_answer": interactive_answer,
            "full_answer": answer
        })
    except Exception as e:
        logging.error(f"Error in process_query: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": f"An error occurred while processing the query: {str(e)}"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {str(e)}")
    logging.error(traceback.format_exc())
    return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)
