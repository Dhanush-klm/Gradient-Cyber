from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import pandas as pd
from openai import OpenAI
import tiktoken
from pinecone import Pinecone
import time
import logging
import traceback
import os

app = Flask(__name__, static_folder='.')
CORS(app)
from dotenv import load_dotenv
load_dotenv()

# Access your API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = "9c097a58-6008-409a-859a-668a002320f6"
INDEX_NAME = "gradient-cyber"
BATCH_SIZE = 100

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI and Pinecone
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(INDEX_NAME)

def truncate_text(text, max_tokens=4000):
    encoding = tiktoken.get_encoding("cl100k_base")
    encoded = encoding.encode(text)
    return encoding.decode(encoded[:max_tokens])

def generate_embedding(text):
    try:
        response = client.embeddings.create(model="text-embedding-ada-002", input=text)
        return response.data[0].embedding, response.usage.total_tokens
    except Exception as e:
        logging.error(f"Error generating embedding: {str(e)}")
        return None, 0

def process_query_logic(query):
    query_embedding, _ = generate_embedding(query)
    if query_embedding is None:
        return jsonify({"error": "Failed to generate query embedding"}), 400

    results = pinecone_index.query(
        namespace="ns1",
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    context = "\n".join([
        f"ID: {match['id']}\n" +
        f"Event Date/Time: {match['metadata'].get('eventDtgTime', 'N/A')}\n" +
        f"Display Title: {match['metadata'].get('displayTitle', 'N/A')}\n" +
        f"Status: {match['metadata'].get('status', 'N/A')}\n" +
        f"Combined Text: {match['metadata'].get('combined_text', 'N/A')}\n" +
        "---"
        for match in results['matches']
    ])

    truncated_context = truncate_text(context, max_tokens=4000)

    system_prompt = """You are an AI assistant specializing in analyzing SITREP data. Provide a concise answer based on the given context. Structure your response with: 1. Summary, 2. Key Findings, 3. Recommendations."""

    user_prompt = f"Query: {query}\nRelevant Information:\n{truncated_context}\nProvide a structured answer based on the given information."

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return jsonify({"answer": response.choices[0].message.content})
    except Exception as e:
        logging.error(f"Error in OpenAI API call: {str(e)}")
        return jsonify({"error": "An error occurred while processing the query"}), 500

@app.route('/')
def index():
    return render_template('.','index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')

@app.route('/script.js')
def script():
    return send_from_directory('.', 'script.js')

@app.route('/api/query', methods=['POST'])
def process_query():
    logging.info(f"Received POST request to /api/query")
    logging.info(f"Request headers: {request.headers}")
    logging.info(f"Request data: {request.get_data(as_text=True)}")

    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON"}), 400
        query = data.get('query')

        if not query:
            return jsonify({"error": "No query provided"}), 400

        logging.info(f"Processing query: {query}")
        return process_query_logic(query)
    except Exception as e:
        logging.error(f"Error in process_query: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": f"An error occurred while processing the query: {str(e)}"}), 500

@app.route('/api/upload', methods=['POST'])
def upload_data():
    logging.info(f"Received POST request to /api/upload")
    logging.info(f"Request headers: {request.headers}")

    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON"}), 400

        # Process and upload data to Pinecone
        # This is a placeholder - you'll need to implement the actual upload logic
        total_tokens_used = 0
        total_requests = 0

        # Simulate processing
        time.sleep(2)

        return jsonify({
            "message": "Upload successful",
            "totalTokensUsed": total_tokens_used,
            "totalRequests": total_requests
        })
    except Exception as e:
        logging.error(f"Error in upload_data: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": f"An error occurred while uploading data: {str(e)}"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {str(e)}")
    logging.error(traceback.format_exc())
    return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
