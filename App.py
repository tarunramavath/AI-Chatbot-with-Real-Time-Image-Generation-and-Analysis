import os
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
import torch
import google.generativeai as genai

# Set up Gemini API Key
os.environ["GEMINI_API_KEY"] = os.getenv('API_KEY')
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Configuration for Stable Diffusion
class Configure:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device=device).manual_seed(seed)
    image_gen_steps = 15
    image_gen_model_id = "stabilityai/stable-diffusion-2-1"
    image_gen_size = (512, 512)
    image_gen_guidance_scale = 9

# Load Stable Diffusion model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    Configure.image_gen_model_id,
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True,
)
image_gen_model = image_gen_model.to(Configure.device)

# Function to generate text using Gemini
def generate_text(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-8b-exp-0827")
        response = model.generate_content(prompt)
        st.success("Response:")
        return response.text
    except Exception as e:
        st.error(f"Error generating text: {e}")
        return None

# Function to generate an image
def generate_image(prompt):
    try:
        image = image_gen_model(
            prompt=prompt,
            num_inference_steps=Configure.image_gen_steps,
            generator=Configure.generator,
        ).images[0]
        return image
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

# Function to summarize an image
def summarize_image(uploaded_image):
    try:
          # Load processor and model
          processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
          model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

          # Move the model to the correct device
          model = model.to(Configure.device)  # Ensure the model is on the correct device
          model.eval()  # Set the model to evaluation mode

          # Open the uploaded image and preprocess it
          image = Image.open(uploaded_image).convert("RGB")

          # Move inputs to the same device as the model
          inputs = processor(image, return_tensors="pt").to(Configure.device)  # Ensure inputs are on the correct device

          # Generate the output
          output = model.generate(**inputs)

          # Decode and return the caption
          caption = processor.decode(output[0], skip_special_tokens=True)
          return caption
    except Exception as e:
        st.error(f"Error summarizing image: {e}")
        return None

# Streamlit Application
st.title("Chatbot with Gemini API and Image Processing")

# Menu options
menu = st.radio("Choose an option:", ["Text Generation", "Image Generation", "Image Summary"])

if menu == "Text Generation":
    st.subheader("Generate Text")
    prompt = st.text_input("Enter your prompt:")
    if st.button("Generate Text"):
        if prompt:
            response = generate_text(prompt)
            if response:
                st.markdown(response)
        else:
            st.error("Please enter a prompt.")

elif menu == "Image Generation":
    st.subheader("Generate Image")
    prompt = st.text_input("Enter your prompt for the image:")
    if st.button("Generate Image"):
        if prompt:
            image = generate_image(prompt)
            if image:
                st.image(image, caption="Generated Image", use_column_width=True)
        else:
            st.error("Please enter a prompt.")

elif menu == "Image Summary":
    st.subheader("Summarize Image")
    uploaded_image = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
    if st.button("Summarize Image"):
        if uploaded_image:
            # Display the uploaded image
            image = Image.open(uploaded_image).convert("RGB")
            # Generate explanation
            with st.spinner("Analyzing the image..."):
                caption = summarize_image(image)
            # Display explanation
            st.write(caption)
        else:
            st.error("Please upload an image.")
