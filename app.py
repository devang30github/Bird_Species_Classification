import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import base64
from io import BytesIO

# Load environment variables for API keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Load class names from classes.txt
def load_class_names(filepath):
    class_names = {}
    with open(filepath, 'r') as file:
        for line in file:
            index, name = line.strip().split(' ')
            class_names[int(index)] = name.strip()
    return class_names

# Load class names (ensure correct path to "classes.txt")
class_names = load_class_names("classes.txt")

# Device setup for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and modify DenseNet model
model = models.densenet121(pretrained=True)

# Freeze initial layers as per setup
for param in model.parameters():
    param.requires_grad = False

# Unfreeze specific layers as per your training setup
for param in model.features.denseblock3.parameters():
    param.requires_grad = True
for param in model.features.denseblock4.parameters():
    param.requires_grad = True

# Modify the classifier to match 200 classes
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier.in_features, 200)
)

# Load model weights
model.load_state_dict(torch.load("model_densenet121.pth", map_location=device))
model = model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prediction function
def predict_bird_species(image, threshold=0.6):
    # Apply transformations
    image = transform(image).unsqueeze(0).to(device)
    # Get model prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)
        
        if max_prob.item() < threshold:
            return None, max_prob.item()  # Indicating unrecognized species
         
        species_name = class_names[predicted.item() + 1]
        species_name = species_name.split('.', 1)[1].strip()
    
    return species_name, max_prob.item()

# Initialize ChatGroq for fun facts
try:
    llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma-7b-It")
except Exception as e:
    st.error(f"Error initializing ChatGroq: {e}")

# Generate a fun fact using LLM
def get_bird_fact(species_name):
    prompt = f"Provide a unique fun fact or interesting information about the bird species {species_name}. Please provide minimum 50 words information"
    
    try:
        response = llm.invoke(prompt)
        
        if hasattr(response, 'content'):
            return response.content
        else:
            st.error("No response content found.")
            return "Could not retrieve fun fact."
    
    except Exception as e:
        st.error(f"Error fetching fun fact: {e}")
        return "Could not retrieve fun fact."

# Streamlit app setup with custom CSS styling
st.set_page_config(page_title="Bird Species Classifier", page_icon=":bird:", layout="centered")

# Function to convert an image to Base64
def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Custom CSS for improved styling
st.markdown("""
    <style>
        /* CSS for styling uploaded image */
        .centered-image {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 90%;
            max-width: 500px;
            margin-bottom: 10px;
        }
        /* General app styling */
        .stApp {
            background-color: #f7f9fc;
        }
        /* Custom font size for st.write elements */
        .stMarkdown p {
            font-size: 1.2rem; /* Adjust font size as needed */
            color: #333;
        }
        /* Style for alerts, text areas, headers, and other elements */
        .stTextArea, .stFileUploader {
            border-radius: 8px;
            border: 1px solid #d1d1d1;
        }
        .stAlert {
            background-color: #f1f9ff;
            color: #333;
            border-left: 5px solid #3b82f6;
            font-size: 1.5rem;
        }
        .stAlert[data-baseweb="notification"] div[role="alert"] {
            font-size: 1.5rem; /* Adjust font size for st.success and st.info */
            color: #333;
        }
        /* Styling for info and success message content */
        .stAlert[data-baseweb="notification"] div[role="alert"] strong {
            font-size: 1.5rem; /* Adjust font size for bold text */
        }
        .stSpinner {
            color: #3b82f6;
        }
        h1, h2, h3 {
            color: #3b82f6;
            font-family: 'Segoe UI', Tahoma, Geneva, sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# Main title and instructions
st.title("🐦 Bird Species Classification")
st.write("Upload an image of a bird to get the predicted species and learn a fun fact!")

# Sidebar for class list
st.sidebar.title("Bird Species List")
for index, name in class_names.items():
    st.sidebar.write(f"{name}")

# File uploader with instructions
uploaded_file = st.file_uploader("Upload a bird image (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    img_base64 = get_image_base64(image)
    st.markdown(
        f"""
        <div class="centered-image">
            <img src="data:image/png;base64,{img_base64}" alt="Uploaded Bird Image" class="centered-image" />
        </div>
        """,
        unsafe_allow_html=True
    )

    species_name, confidence = predict_bird_species(image)

    if species_name:
        st.success(f"**Species Identified:** {species_name}")
        # Fetch fun fact as usual
        with st.spinner("Fetching a fun fact..."):
            fact = get_bird_fact(species_name)
        st.info(f"**Did you know?** {fact}")
    else:
        st.warning("The uploaded image may contain a bird species outside the model's 200-class list.")

else:
    st.write("Please upload an image of a bird to continue.")
