🚀 AI Chatbot (RAG + Streamlit)

This project is an AI-powered chatbot built using Python, Streamlit, and Retrieval-Augmented Generation (RAG).
It supports file uploads, embeddings, and intelligent conversation handling.

📥 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/pinnu1/AI_Chat_bot.git
cd AI_Chat_bot

2️⃣ Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Linux/Mac

3️⃣ Install all dependencies
pip install -r requirements.txt

▶️ Run the Application

To start the chatbot UI:

streamlit run streamlit_frontend.py


This will launch the Streamlit interface in your browser.

📁 Project Structure
AI_Chat_bot/
│── backend.py              # Backend logic, embeddings, RAG pipeline
│── streamlit_frontend.py   # Streamlit UI
│── requirements.txt        # Python dependencies
│── chat_history.db         # Local database (auto-created)
│── README.md               # Project documentation

📌 Features

🧠 AI-powered chatbot

📄 RAG-based document querying

📂 File upload for knowledge base

💬 Conversation history

⚡ Fast & lightweight

🌐 Streamlit web interface

🛠 Technologies Used

Python

Streamlit

LangChain / Gemini / Embeddings

SQLite

RAG architecture

🤝 Contributing

Pull requests are welcome.
For major changes, open an issue first to discuss what you’d like to modify.


