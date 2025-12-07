# PRISM - Agentic RAG-Based Learning System

PRISM: Personalized Retrieval-Integrated System for Multimodal Adaptive Learning

PRISM is an adaptive learning application designed for students, leveraging an agentic retrieval-augmented generation (RAG) system to answer questions based on course materials or internet search when necessary.

## Features

- **Course-Relevant Answers**: Generates responses based on course materials from vector store
- **Internet Search**: Uses Tavily API for questions related to course topics but not in materials
- **Query Classification**: Intelligently routes queries (course-relevant, out-of-scope, irrelevant)
- **Follow-up Questions**: Handles vague queries by asking clarifying questions
- **Response Evaluation**: Evaluates and refines responses using mathematical metrics
- **Personalization**: Tailors responses to student's academic level and major

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the `.env.example` file to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your:
- `OPENAI_API_KEY`: Your OpenAI API key
- `TAVILY_API_KEY`: Your Tavily API key for internet search

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
PRISM Code/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── ui/                             # UI components
│   ├── styling.py                  # Theme and CSS styling
│   ├── sidebar.py                  # Sidebar components
│   ├── chat.py                     # Chat interface
│   └── session.py                  # Session management
├── config/                         # Configuration management
├── core/                           # Core agentic RAG logic
├── retrieval/                      # Vector store and retrieval
├── generation/                     # LLM and response generation
├── search/                         # Internet search integration
└── utils/                          # Utility functions
```

## Usage

1. Fill out the sidebar form with:
   - Student ID
   - Degree level
   - Major
   - Course

2. Click "Start PRISM Session"

3. Ask questions about your course material in the chat interface

## Future Enhancements

- Knowledge graph integration
- Database support for session persistence
- Advanced evaluation metrics
- Multi-modal support

## License

© PRISM Adaptive Learning System 2025 (UNT Dissertation POC)
