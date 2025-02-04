
# Medical Chatbot with RLHF

A Streamlit-based medical chatbot that uses LangChain, Groq, and various other technologies to provide medical information and answer health-related questions.
- Example 1

![Example2](img/2.jpeg)
- Example 2
![Example1](img/1.jpeg)

## Features

- Interactive chat interface with medical knowledge
- Multiple language model options including LLaMA, Gemma, and Mixtral
- Document retrieval and RAG (Retrieval Augmented Generation)
- User feedback collection
- Vector similarity search using Qdrant
- Conversation memory management

## Prerequisites

- Python 3.10+
- Pip package manager
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Goodnight77/Medical_chatbot_with_rlhf.git
cd Medical_chatbot_with_rlhf
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Environment Setup

1. Create a `.env` file in the root directory:
```bash
touch .env
```

2. Add the following environment variables to your `.env` file:
```plaintext
# API Keys
GROQ_API_KEY=your_groq_api_key
HF_API_KEY=your_huggingface_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
COHERE_API_KEY=your_cohere_api_key

# LangSmith Configuration
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=your_langsmith_endpoint
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=your_langsmith_project

# Collection Configuration
collection_name=your_collection_name
```

Replace all `your_*` values with your actual API keys and configuration settings.

## Required API Keys and Services

1. **Groq**: Sign up at [groq.com](https://groq.com) to get your GROQ_API_KEY
2. **Hugging Face**: Create an account at [huggingface.co](https://huggingface.co) to get your HF_API_KEY
3. **Qdrant**: Set up a Qdrant instance and get your QDRANT_API_KEY and QDRANT_URL
4. **Cohere**: Register at [cohere.ai](https://cohere.ai) to get your COHERE_API_KEY
5. **LangSmith**: Set up LangSmith for tracking and monitoring

## Project Structure

```
medical-chatbot/
├── app.py              # Main Streamlit application
├── utils.py           # Utility functions
├── requirements.txt    # Project dependencies
├── .env               # Environment variables
├── chromadb/         # Vector database directory
├── README.md          # Project documentation
└── venv              # Virtual env
```

## Dependencies

Create a `requirements.txt` file with the following packages:

```plaintext
streamlit
langchain
langsmith
streamlit-feedback
python-dotenv
qdrant-client
langchain-groq
langchain-community
langchain-core
langchain-chroma
nomic
cohere
chromadb
langchain_huggingface
langchain_nomic
"nomic[local]"
```

## Running the Application

1. Ensure all environment variables are set in the `.env` file
2. Activate your virtual environment if not already activated
3. Run the Streamlit application:
```bash
streamlit run app.py
```

## Usage

1. Select a model from the sidebar dropdown
2. Adjust the temperature and number of retrieved documents as needed
3. Type your medical question in the chat input
4. View the response and provide feedback using the feedback widget
5. Access consulted documents from the sidebar

