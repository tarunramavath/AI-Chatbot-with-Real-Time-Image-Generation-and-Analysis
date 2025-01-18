# AI-Chatbot-with-Real-Time-Image-Generation-and-Analysis

## Project Overview
This Streamlit application integrates with Google Generative AI to process user prompts and optionally handle uploaded images. Users can generate responses to text inputs using a powerful generative model, providing a seamless and interactive experience.

## Features
- **Text Prompt Processing**: Users can enter a text prompt to receive AI-generated content.
- **Image Upload**: Supports uploading multiple images in `.png`, `.jpg`, and `.jpeg` formats.
- **Secure API Integration**: Configures Google Generative AI via API key for secure access.

## Prerequisites
- Python 3.8 or higher
- Required Libraries:
  - `streamlit`
  - `google-generativeai`

## Usage
1. **Set Up API Key**:
   - Securely store your Google Generative AI API key.
   - Update the `GEMINI_API_KEY` environment variable in the code or use Streamlit secrets for better security.

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
3. **Interact with the App**:
   - Enter a text prompt in the input field.
   - Optionally, upload images.
   - Click the "Submit" button to process the input and generate a response.

## File Structure
- `app.py`: Main application file.

## Notes
- Ensure your API key is valid and has access to the required Google Generative AI services.
- The current implementation does not integrate image processing with the generative model. This feature can be added in future updates.

## Future Enhancements
- Implement image-based generative AI features.
- Enhance UI for better user experience.
- Add advanced error handling and logging.

---

Feel free to contribute to this project or suggest improvements!

