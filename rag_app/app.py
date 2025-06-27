import os
from flask import Flask, request, render_template, jsonify
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for the RAG system
vector_store = None
documents = []
embedding_model = None # Initialize lazily

# Function to load the embedding model
def load_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

# Function to process PDF and build vector store
def process_pdf(file_path):
    global vector_store, documents
    reader = PdfReader(file_path)
    text_chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            # Simple chunking: split by paragraphs or sentences.
            # For more advanced RAG, consider recursive character text splitter.
            chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
            text_chunks.extend(chunks)

    documents = text_chunks
    model = load_embedding_model()
    embeddings = model.encode(documents)
    embeddings = np.array(embeddings).astype('float32')

    vector_store = faiss.IndexFlatL2(embeddings.shape[1])
    vector_store.add(embeddings)
    print(f"Processed PDF and added {len(documents)} chunks to vector store.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        process_pdf(filepath)
        return jsonify({"message": "PDF processed successfully!"}), 200

@app.route('/ask_question', methods=['POST'])
def ask_question():
    if vector_store is None:
        return jsonify({"error": "Please upload a PDF first."}), 400

    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"error": "No question provided."}), 400

    model = load_embedding_model()
    question_embedding = model.encode([user_question]).astype('float32')

    # Search top k relevant documents
    k = 3 # Number of relevant chunks to retrieve
    D, I = vector_store.search(question_embedding, k)
    retrieved_chunks = [documents[i] for i in I[0]]

    # Construct prompt for LM Studio
    context = "\n".join(retrieved_chunks)
    prompt = f"Context: {context}\n\nQuestion: {user_question}\n\nAnswer:"

    # Call LM Studio API
    lm_studio_url = "http://localhost:1234/v1/chat/completions" # Default LM Studio API endpoint
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "local-model", # Replace with your LM Studio model name if different
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(lm_studio_url, headers=headers, json=data)
        response.raise_for_status() # Raise an exception for HTTP errors
        lm_response = response.json()
        answer = lm_response['choices'][0]['message']['content']
        return jsonify({"answer": answer}), 200
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Could not connect to LM Studio API. Please ensure it is running at http://localhost:1234."}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
