from flask import Flask, request, jsonify, Response
import os

from llm.ollama_client import OllamaClient
from nlp.processor import NLPProcessor
from pipeline.chat_pipeline import ChatPipeline
from pipeline.image_pipeline import ImagePipeline

app = Flask(__name__)

# Initialize components
print("Initializing NLP Processor...")
nlp = NLPProcessor()
print("Initializing Ollama Client...")
llm = OllamaClient()
print("Initializing Pipelines...")
chat_pipeline = ChatPipeline(nlp, llm)
image_pipeline = ImagePipeline()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    text = data.get('text', '')
    agent_type = data.get('agent_type', 'patient')
    domain = data.get('domain', 'general')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    def generate():
        try:
            for chunk in chat_pipeline.run_chat(text, agent_type, domain):
                yield chunk
        except Exception as e:
            yield f"\nError: {str(e)}"

    return Response(generate(), mimetype='text/plain')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    domain = request.form.get('domain', 'brain')
    agent_type = request.form.get('agent_type', 'patient')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save temp file
    temp_path = "temp_upload" + os.path.splitext(file.filename)[1]
    file.save(temp_path)

    try:
        # Get image prediction
        prediction_text = image_pipeline.predict(temp_path, domain)
        
        # Stream explanation using ChatPipeline
        def generate():
            yield f"[{prediction_text}]\n\n"
            try:
                for chunk in chat_pipeline.run_chat(
                    text="", 
                    agent_type=agent_type, 
                    domain=domain, 
                    is_image_result=True, 
                    image_prediction=prediction_text
                ):
                    yield chunk
            except Exception as e:
                yield f"\nError in generation: {str(e)}"
                
        return Response(generate(), mimetype='text/plain')
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    print("Starting Server...")
    app.run(debug=True, port=5000)
