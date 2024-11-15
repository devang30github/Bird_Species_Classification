# 🐦 Bird Species Identification App

This project is a machine learning-powered web application that identifies bird species from uploaded images and provides fun facts about the species using an LLM (Large Language Model).

## ✨ Features

- Upload an image of a bird to classify its species from a dataset of 200 classes.
- Learn interesting facts about the bird species with the help of a language model.
- User-friendly web interface built with Streamlit.

---

## 🛠️ Tech Stack

- **Frontend & Backend**: Streamlit
- **Machine Learning**: PyTorch , CNN , DenseNet121 , torchvision
- **Language Model**: ChatGroq, Gemma-7b-It , LangChain
- **Environment Variables**: `dotenv`
- **Image Processing**: `Pillow`

---

## 🖼️ App Demo

1. **Upload an image**: Choose a bird image in JPG/PNG format.
2. **View Prediction**: The app predicts the bird species from 200 species.
3. **Learn Facts**: A fun fact about the species is shown below the result.

---

## 📂 Directory Structure

|bird_species_identification /  
├── app.py # Main application file  
├──bird-species-identification.ipynb #kaggle notebook where model get trained  
├── classes.txt # List of bird classes  
├── model_densenet121.pth # Trained PyTorch model  
├── .env # Environment variables file  
├── requirements.txt # Python dependencies  
└── README.md # Project documentation

## 🔧 Installation and Setup

Follow these steps to set up and run the project:

```bash
1️⃣ Clone the Repository
git clone https://github.com/devang30github/bird-species-identification.git
cd bird-species-identification

2️⃣ Create a Virtual Environment
python -m venv myenv
Activate the environment:
On Linux/Mac:
source venv/bin/activate
On Windows:
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set Up Environment Variables
Create a .env file in the project root with the following content:
GROQ_API_KEY=your_groq_api_key
Replace your_groq_api_key with your actual API key for the LangChain Groq API.

🚀 Running the Application
Start the Streamlit app:
streamlit run app.py
Open the URL provided in the terminal (default: http://localhost:8501) to access the app.
```

---

## 📝 Key Notes

Model Weights: The app uses a trained DenseNet121 model (model_densenet121.pth). Ensure this file is in the project root directory.
Classes File: The classes.txt file contains bird class mappings.
LLM Integration: The app integrates the Groq API to fetch fun facts. Ensure your .env file is correctly configured.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
Fork this repository
Submit issues or feature requests
Open pull requests with improvements

---

## 💬 Contact

For queries or suggestions, contact:

Name: Devang Gawade
Email: gawadedevang@gmail.com
LinkedIn: https://www.linkedin.com/in/devang-gawade-a82074262/

---

## 🎉 Happy Coding and Bird Watching! 🐦✨
