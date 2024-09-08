# I.R.I.S - Image Recognition Insight System

**I.R.I.S** (Image Recognition Insight System) is a Flask-based web application designed for generating content from user inputs and image data. It integrates multiple state-of-the-art models to provide features such as text generation, image captioning, question answering, and image-text matching. 

## Features

1. **Text Generation**: Generates text responses based on user inputs using the GPT-NeoX model.
2. **Image Captioning**: Provides captions for images uploaded by the user using the BLIP model.
3. **Question Answering**: Answers questions based on a provided context using Google's FLAN-T5 XXL model.
4. **Image-Text Matching**: Calculates similarity scores between text and images using the CLIP model.

## Technologies Used

- **Flask**: For creating the web application.
- **Transformers**: For various natural language processing (NLP) and vision tasks.
- **PyTorch**: As the backend framework for the models.
- **PIL**: For image processing.
- **JavaScript**: For front-end interactivity and AJAX requests.

## Usage

- **Chat Interface**: Enter your message or upload an image to receive a response or caption.
- **Text Generation**: Enter text in the chat input to get a generated response.
- **Image Captioning**: Upload an image to receive a descriptive caption.
- **Question Answering**: Submit a question along with the context to get an answer.
- **Image-Text Matching**: Provide an image and a text to receive a similarity score.

## API Endpoints

1. **`/`**: Renders the main HTML page.
2. **`/generate_content`**: Handles text and image data, generates content and captions.
   - **Method**: `POST`
   - **Payload**: JSON with text and image data.
   - **Response**: JSON with generated text and image caption.
3. **`/chat`**: Handles chat messages.
   - **Method**: `POST`
   - **Payload**: Form data with user input.
   - **Response**: JSON with bot response.
4. **`/question_answering`**: Answers questions based on provided context.
   - **Method**: `POST`
   - **Payload**: JSON with question and context.
   - **Response**: JSON with the answer.
5. **`/image_text_matching`**: Matches text with image and provides a similarity score.
   - **Method**: `POST`
   - **Payload**: JSON with text and base64 encoded image data.
   - **Response**: JSON with the similarity score.

## Front-End

The front-end of the application is built with HTML, CSS, and JavaScript. It includes:

- A chat interface for sending messages and uploading images.
- A loader animation for showing processing status.
- Image preview functionality before sending.
