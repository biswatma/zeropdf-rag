import os
from flask import Flask, request, render_template, jsonify
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
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
            chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
            text_chunks.extend(chunks)

    if not text_chunks:
        raise ValueError("No readable text found in PDF")

    documents = text_chunks
    model = load_embedding_model()
    embeddings = model.encode(documents)
    embeddings = np.array(embeddings).astype('float32')

    if len(embeddings.shape) != 2 or embeddings.shape[0] == 0:
        raise ValueError("Failed to generate valid embeddings from text")

    vector_store = faiss.IndexFlatL2(embeddings.shape[1])
    vector_store.add(embeddings)
    print(f"Processed PDF and added {len(documents)} chunks to vector store.")

@app.route('/')
def index():
    app.logger.info("Serving index.html")
    return render_template('index.html')

@app.route('/test')
def test_model_load():
    return render_template('test_model_load.html')

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

@app.route('/get_context', methods=['POST'])
def get_context():
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

    context = "\n".join(retrieved_chunks)
    return jsonify({"context": context}), 200

if __name__ == '__main__':
    app.run(debug=True)
