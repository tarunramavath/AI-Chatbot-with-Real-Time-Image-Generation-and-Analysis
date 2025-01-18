import os
import streamlit as st
import google.generativeai as genai

# Streamlit App Title
st.title("My AI with Google Generative AI")

# Set up the API key securely
# os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]  # Use Streamlit secrets for security
os.environ["GEMINI_API_KEY"] = "AIzaSyDJv4XrqWFhuuIHFb_j0kt3YgAUVmD9CjY"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Load the Generative AI model
try:
    model = genai.GenerativeModel("gemini-1.5-flash-8b-exp-0827")
except Exception as e:
    st.error(f"Error initializing the Generative AI model: {e}")

# User input for a text prompt
query = st.text_input("Enter your prompt:")

# Option to upload an image (accept multiple files)
uploaded_images = st.file_uploader("Upload your image(s):", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

# Submit button to process the input
if st.button("Submit"):
    if not query:
        st.error("Please enter a text prompt.")
    else:
        try:
            # Generate response from the AI model for the text prompt
            response = model.generate_content(query)
            st.success("Response:")
            st.text(response.text)  # Display the text response

            # Process uploaded images if available
            if uploaded_images:
                st.info("You have uploaded the following images:")
                for image in uploaded_images:
                    st.image(image, caption=image.name)
                # Optional: Associate images with the text response
                st.warning("Note: Image handling with the generative model is not currently implemented.")
        except Exception as e:
            st.error(f"Error generating content: {e}")
