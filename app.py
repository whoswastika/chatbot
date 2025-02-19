import os
import tempfile
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

# Configure Gemini Embedding Model
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
Settings.embed_model = GeminiEmbedding(api_key=GOOGLE_API_KEY)

# Configure Gemini LLM
Settings.llm = Gemini(api_key=GOOGLE_API_KEY)

app = Flask(__name__, static_folder="static")
CORS(app)  # Enable CORS for frontend requests

@app.route("/")
def home():
    """Serve the frontend page."""
    return render_template("index.html")

def process_pdf(pdf_file):
    """Extracts text from PDF and creates an index."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        pdf_file.save(temp_pdf.name)

    # Load and index PDF content
    reader = SimpleDirectoryReader(input_files=[temp_pdf.name])
    docs = reader.load_data()

    # Create a vector index
    index = VectorStoreIndex.from_documents(docs)
    return index

def chatbot_response(query, index):
    """Queries the LlamaIndex for an answer."""
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response.response

@app.route("/chat", methods=["POST"])
def chat():
    """Handles user queries and returns AI-generated answers."""
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400

    pdf_file = request.files["pdf"]
    user_query = request.form.get("query")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Process PDF and create an index
    index = process_pdf(pdf_file)

    # Generate AI response
    response = chatbot_response(user_query, index)

    return jsonify({"answer": response})

if __name__ == "__main__":
       port = int(os.environ.get("PORT", 5000))
       app.run(host="0.0.0.0", port=port)