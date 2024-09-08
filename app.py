from flask import Flask, render_template, request, jsonify
import base64
import logging
import torch
from PIL import Image
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import io

# Initialize the Flask application
app = Flask(__name__)

# Load the GPT-NeoX model and tokenizer for text generation
gpt_neox_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
gpt_neox_tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

# Load the question-answering pipeline using Google's FLAN-T5 XXL model
qa_model = pipeline('question-answering', model="google/flan-t5-xxl")

# Load the BLIP model and processor for image captioning
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")

# Load the CLIP model and processor for image-text matching
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Set up logging for debugging purposes
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    # Render the main HTML page
    return render_template('index.html')

def generate_text_neox(prompt):
    """
    Generate text based on the given prompt using the GPT-NeoX model.
    
    Args:
        prompt (str): The input text prompt to generate a response for.
    
    Returns:
        str: The generated text from the model.
    """
    inputs = gpt_neox_tokenizer(prompt, return_tensors="pt")
    outputs = gpt_neox_model.generate(inputs['input_ids'], max_length=200)
    generated_text = gpt_neox_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.strip()

def process_image(image_data):
    """
    Process the given image data to generate a caption using the BLIP model.
    
    Args:
        image_data (bytes): The image data in bytes format.
    
    Returns:
        str: The generated caption for the image.
    """
    image = Image.open(io.BytesIO(image_data))
    inputs = blip_processor(image, return_tensors="pt")
    generated_ids = blip_model.generate(**inputs)
    caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption

@app.route('/generate_content', methods=['POST'])
def generate_content():
    """
    Handle POST requests to generate content from both text and image.
    
    The request should contain both text and image data in base64 format.
    
    Returns:
        json: A JSON response containing generated text and image caption.
    """
    try:
        # Get the JSON payload from the request
        payload = request.json
        logging.debug("Request payload: %s", payload)

        # Check if the payload contains valid content
        if 'contents' not in payload or not payload['contents']:
            return jsonify({'error': 'Invalid request payload.'}), 400

        # Extract text and image data from the payload
        text_input = payload['contents'][0]['parts'][0]['text']
        image_base64 = payload['contents'][0]['parts'][1]['inlineData']['data']
        image_data = base64.b64decode(image_base64)

        # Generate text and process the image
        generated_text = generate_text_neox(text_input)
        image_caption = process_image(image_data)

        # Return the results as JSON
        return jsonify({
            'generated_text': generated_text,
            'image_caption': image_caption
        })

    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        return jsonify({'error': 'Internal Server Error.'}), 500

@app.route('/chat', methods=['POST'])
def chat_response():
    """
    Handle POST requests for chat responses.
    
    The request should contain user input text.
    
    Returns:
        json: A JSON response containing the user input and the generated bot response.
    """
    try:
        # Get user input from the request
        user_input = request.form['user_input']
        logging.debug("User input: %s", user_input)

        # Generate a response based on user input
        if user_input.lower() != 'exit':
            bot_response = generate_text_neox(user_input)
        else:
            bot_response = "Chat ended. Please refresh the page to start a new chat."

        # Return the user input and bot response as JSON
        return jsonify(user_input=user_input, bot_response=bot_response)

    except Exception as e:
        logging.error("An error occurred in chat: %s", str(e))
        return jsonify({'error': 'Internal Server Error.'}), 500

@app.route('/question_answering', methods=['POST'])
def question_answering():
    """
    Handle POST requests for question answering.
    
    The request should contain a question and context text.
    
    Returns:
        json: A JSON response containing the answer to the question based on the context.
    """
    try:
        # Get the JSON data from the request
        data = request.json
        question = data.get('question', '')
        context = data.get('context', '')

        # Check if both question and context are provided
        if not question or not context:
            return jsonify({'error': 'Invalid input. Provide both question and context.'}), 400

        # Use the question-answering model to get the answer
        result = qa_model(question=question, context=context)
        return jsonify(result)

    except Exception as e:
        logging.error("An error occurred during question answering: %s", str(e))
        return jsonify({'error': 'Internal Server Error.'}), 500

@app.route('/image_text_matching', methods=['POST'])
def image_text_matching():
    """
    Handle POST requests for image-text matching.
    
    The request should contain text and image data in base64 format.
    
    Returns:
        json: A JSON response containing the similarity score between the text and the image.
    """
    try:
        # Get the JSON data from the request
        data = request.json
        text_input = data.get('text', '')
        image_base64 = data.get('image', '')

        # Check if both text and image are provided
        if not text_input or not image_base64:
            return jsonify({'error': 'Invalid input. Provide both text and image.'}), 400

        # Decode and process the image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))

        # Prepare the inputs for the CLIP model
        inputs = clip_processor(text=[text_input], images=image, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)

        # Calculate the match score
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).tolist()

        # Return the match score as JSON
        return jsonify({
            'text': text_input,
            'image_match_score': probs[0]
        })

    except Exception as e:
        logging.error("An error occurred during image-text matching: %s", str(e))
        return jsonify({'error': 'Internal Server Error.'}), 500

# Run the Flask application in debug mode
if __name__ == '__main__':
    app.run(debug=True)
