# PRISM - Agentic RAG-Based Adaptive Learning System

**PRISM: Personalized Retrieval-Integrated System for Multimodal Adaptive Learning**

PRISM is an adaptive learning application designed for students, leveraging a **7-agent LangGraph pipeline** with retrieval-augmented generation (RAG) to answer questions based on course materials. When course content is insufficient, it falls back to internet search. Responses are personalized to each student's academic background, mathematically evaluated for quality, and refined in a loop until they meet a quality threshold.

## Architecture Overview

PRISM uses a **LangGraph state graph** to orchestrate 7 specialized agents that communicate via an **Agent-to-Agent (A2A) protocol**. Each agent handles a distinct stage of the pipeline:

```
User Query
    │
    ▼
┌─────────────────────┐
│ 1. Query Refinement  │──── vague? ───▶ Ask follow-up ──▶ User
│    Agent             │
└─────────┬───────────┘
          │ clear
          ▼
┌─────────────────────┐
│ 2. Relevance Agent   │──── irrelevant? ──▶ Reject ──▶ User
└─────────┬───────────┘
          │ relevant
          ▼
┌─────────────────────┐
│ 3. Course RAG Agent  │──── content found? ──────────────┐
└─────────┬───────────┘                                    │
          │ not found                                      │
          ▼                                                ▼
┌─────────────────────┐                    ┌──────────────────────┐
│ 4. Web Search Agent  │───────────────────▶│ 5. Personalization   │
│    (Tavily API)      │                    │    Agent             │
└──────────────────────┘                    └──────────┬───────────┘
                                                       │
                                                       ▼
                                            ┌──────────────────────┐
                                      ┌────▶│ 6. Evaluation Agent   │
                                      │     └──────────┬───────────┘
                                      │                │
                                      │     pass (≥0.70)│    fail (<0.70)
                                      │                │         │
                                      │                ▼         ▼
                                      │            ✅ User   ┌──────────────┐
                                      │                      │ 7. Refinement │
                                      │                      │    Agent      │
                                      │                      └──────┬───────┘
                                      │                             │
                                      └─────────── loop (max 3) ───┘
```

### The 7 Agents

| # | Agent | File | Purpose |
|---|-------|------|---------|
| 1 | **Query Refinement** | `core/nodes/query_refinement.py` | Detects vague/ambiguous queries and asks clarifying follow-up questions. Checks conversation history for context resolution. |
| 2 | **Relevance** | `core/nodes/relevance.py` | Determines if a question is relevant to the selected course using the course description and conversation history. |
| 3 | **Course RAG** | `core/nodes/course_rag.py` | Retrieves content from Pinecone vector store, checks answerability. Routes to web search if course materials don't answer the question. |
| 4 | **Web Search** | `core/nodes/web_search.py` | Searches the internet via Tavily API when course content is insufficient. Supports date-aware queries for current information. |
| 5 | **Personalization** | `core/nodes/personalization.py` | Generates responses tailored to the student's degree level and major. Handles inline citation formatting. |
| 6 | **Evaluation** | `core/nodes/evaluation.py` | Mathematically evaluates response quality using embedding-based metrics (relevance, readability, coherence, coverage) plus trust metrics for web sources. |
| 7 | **Refinement** | `core/nodes/refinement.py` | Improves responses that fail the 0.70 quality threshold. Loops up to 3 times before returning the best attempt with a disclaimer. |

### A2A Communication

Agents communicate through a shared state via the **A2AManager** (`core/a2a/__init__.py`). Each message includes sender, receiver, type, content, and timestamp. This enables inter-agent coordination and full traceability of the pipeline.

### Supplementary Tools

| Tool | File | Description |
|------|------|-------------|
| **Flashcard Generator** | `core/flashcard_generator.py` | Generates Q&A flashcards from course content using RAG |
| **Podcast Generator** | `core/podcast_generator.py` | Creates conversational podcasts from course content using OpenAI TTS, with MCP server fallback (`@mcai/podcast-tts-mcp`) |

## Features

- **7-Agent LangGraph Pipeline**: Orchestrated flow from query refinement to evaluation with automatic refinement loops
- **Agent-to-Agent Communication**: Structured A2A messaging protocol for inter-agent coordination
- **Course-Specific RAG**: Pinecone vector store with course-name filtering for isolated retrieval per course
- **Internet Search Fallback**: Tavily API for questions related to course topics but not covered in materials
- **Mathematical Evaluation**: Embedding-based scoring (relevance, readability, coherence, coverage) with a 0.70 quality threshold
- **Personalization**: Adapts response complexity and language to student's degree level and major
- **Conversation Memory**: LangGraph checkpointing preserves multi-turn context within a session
- **Flashcard Generation**: Create study flashcards from course content on any topic
- **Podcast Generation**: Generate conversational-style podcasts (NotebookLM-like) from course content via OpenAI TTS + MCP fallback
- **Evaluation Framework**: Ablation study runner with DeepEval, RAGAS, LLM-judge, readability, and safety metrics

## Supported Courses

| Course | Type | Content |
|--------|------|---------|
| **INFO 4100** - Introduction to Information Sciences | Undergraduate | 11 modules with PDFs, VTT transcripts, and combined PowerPoint |
| **INFO 6945** - Trends and Issues in Information Science | Doctoral seminar | Weekly readings (PDFs), PowerPoint lectures, assignments |
| **LTEC 4510** - Communications in Business, Education and Industry | Undergraduate | 11 chapter PowerPoints (BCOM textbook), 8 module PDFs |

All course content is stored as embeddings in Pinecone. When a student selects a course, queries are filtered to only retrieve vectors for that course.

## Setup

### 1. Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm (required for podcast TTS via MCP server)

```bash
node --version
npm --version
```

If not installed, download from [nodejs.org](https://nodejs.org/)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For PowerPoint ingestion support:
```bash
pip install "unstructured[pptx]"
```

### 3. Configure Environment Variables

Create a `.env` file with the following keys:

```env
# Required
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=prism-course-materials

# Internet search
TAVILY_API_KEY=your-tavily-api-key

# Optional - MongoDB logging
MONGODB_URI=your-mongodb-atlas-uri
```

### 4. Ingest Course Documents

Course content (PDFs, PPTs, VTTs) must be placed in the `courses/` directory and ingested into the Pinecone vector store:

```bash
# Ingest all courses
python scripts/ingest_documents.py

# Ingest only specific new courses
python scripts/ingest_new_courses.py
```

The ingestion pipeline supports:
- **PDFs**: Text extraction with pdfplumber, table/figure detection, multimodal content via unstructured
- **PPT/PPTX**: Slide-level extraction via unstructured with per-slide chunking
- **VTT transcripts**: Timestamp-based chunking from lecture transcripts

Each chunk is embedded using OpenAI `text-embedding-3-large` (3072 dimensions) and stored in Pinecone with metadata: `course_name`, `document_name`, `module_name`, `page_number`/`timestamp`, `type`.

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
PRISM/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── config/
│   ├── settings.py                     # Environment variables and paths
│   ├── prompts.yaml                    # System prompts and course descriptions
│   └── mcp_client.py                   # MCP client for podcast TTS fallback
├── core/
│   ├── agent.py                        # PRISMAgent orchestrator (LangGraph runner)
│   ├── graph.py                        # LangGraph state graph definition
│   ├── state.py                        # AgentState TypedDict
│   ├── a2a/
│   │   └── __init__.py                 # A2A messaging framework
│   ├── nodes/
│   │   ├── query_refinement.py         # Agent 1: Query refinement
│   │   ├── relevance.py                # Agent 2: Relevance classification
│   │   ├── course_rag.py               # Agent 3: Course content retrieval
│   │   ├── web_search.py               # Agent 4: Web search (Tavily)
│   │   ├── personalization.py          # Agent 5: Response personalization
│   │   ├── evaluation.py               # Agent 6: Mathematical evaluation
│   │   └── refinement.py               # Agent 7: Response refinement
│   ├── flashcard_generator.py          # Flashcard generation tool
│   └── podcast_generator.py            # Podcast generation tool (OpenAI TTS + MCP)
├── retrieval/
│   ├── vector_store.py                 # Pinecone vector store integration
│   ├── retriever.py                    # Course retriever with formatting
│   ├── document_loader.py              # Multimodal PDF loader
│   ├── ppt_loader.py                   # PPT/PPTX slide loader
│   └── vtt_loader.py                   # VTT transcript loader
├── search/
│   └── internet_search.py              # Tavily API internet search agent
├── generation/
│   └── response_generator.py           # LLM response generation
├── eval_runner/                        # Evaluation framework
│   ├── run_variants.py                 # Ablation study variant runner
│   ├── prism_wrapper.py                # PRISM agent wrapper for eval
│   ├── schemas.py                      # Evaluation data schemas
│   ├── config.py                       # Eval configuration
│   ├── load_dataset.py                 # Dataset loader
│   ├── metrics/
│   │   ├── compute.py                  # Metric computation orchestrator
│   │   ├── router.py                   # Metric routing by category
│   │   ├── deepeval_metrics.py         # DeepEval metrics
│   │   ├── ragas_metrics.py            # RAGAS metrics
│   │   ├── judge_metrics.py            # LLM-as-judge metrics
│   │   ├── readability_metrics.py      # Readability scoring
│   │   └── safety_metrics.py           # Toxicity and bias detection
│   └── reporting/
│       ├── aggregate.py                # Score aggregation and summary
│       ├── plots.py                    # Visualization generation
│       └── tables.py                   # CSV/LaTeX table generation
├── evaluation/                         # Evaluation dataset construction
│   ├── build_dataset.py                # Dataset builder
│   ├── extract_keypoints.py            # Keypoint extraction from content
│   ├── retrieve_candidates.py          # Candidate chunk retrieval
│   ├── select_gold_chunks.py           # Gold standard chunk selection
│   └── course_eval_dataset.jsonl       # Pre-built evaluation dataset
├── scripts/
│   ├── ingest_documents.py             # Ingest all courses into Pinecone
│   ├── ingest_new_courses.py           # Ingest specific new courses
│   ├── check_vector_store.py           # Inspect vector store contents
│   ├── reset_vector_store.py           # Reset the vector store
│   ├── run_eval.py                     # Run evaluation pipeline
│   └── test_mongo_connection.py        # Test MongoDB connectivity
├── ui/
│   ├── styling.py                      # Theme and CSS styling
│   ├── sidebar.py                      # Sidebar session setup
│   ├── chat.py                         # Chat interface
│   ├── agent_ui.py                     # Agent pipeline dashboard
│   └── session.py                      # Session management
├── prism_logging/
│   └── mongo_logger.py                 # MongoDB Atlas interaction logging
└── courses/                            # Course content (gitignored)
    ├── INFO 4100-.../                  # PDFs, VTTs, PPTXs per module
    ├── INFO 6945-.../                  # PDFs and PPTXs organized by week
    └── LTEC 4510-.../                  # Chapter PPTXs and module PDFs
```

## Usage

### Basic Chat

1. Fill out the sidebar form with Student ID, Degree, Major, and Course
2. Click **Start PRISM Session**
3. Ask questions about your course material in the chat interface
4. The system will route through the 7-agent pipeline and return a personalized, evaluated response

### Generate Flashcards

1. Click the **+** button next to the input field
2. Check **Generate Flashcards**
3. Enter a topic and press Enter
4. View generated flashcards with source citations
5. Click **Generate 5 More** for additional flashcards

### Generate Podcasts

1. Click the **+** button next to the input field
2. Check **Generate Podcast**
3. Enter a topic and press Enter
4. Wait for generation (1-2 minutes)
5. Use the audio player to listen; click **View Transcript** for the script

## Evaluation Framework

PRISM includes a comprehensive evaluation framework for ablation studies:

### Variants
- **full_system**: All agents enabled (baseline)
- **no_personalization**: Personalization agent disabled
- **no_internal_eval**: Internal evaluation/refinement loop skipped
- **no_rag**: Course RAG disabled, web search only
- **no_web_search**: Web search disabled, course content only
- **no_query_refinement**: Query refinement agent disabled

### Metrics
- **DeepEval**: Correctness, context precision/recall/relevancy
- **RAGAS**: Groundedness, task completeness
- **LLM-as-Judge**: Clarification quality, refusal correctness, tool correctness
- **Readability**: Flesch-Kincaid grade level matching
- **Safety**: Toxicity and bias detection

### Running Evaluations

```bash
python scripts/run_eval.py
```

Results are saved to `results/eval/` with plots, tables, and raw scored data.

## Adding New Courses

1. Create a folder in `courses/` with the course name (this becomes the course identifier)
2. Add PDFs, PPTXs, and/or VTT files (optionally organized in module subfolders)
3. Add a course description in `config/prompts.yaml` under `course_descriptions`
4. Run ingestion:
   ```bash
   python scripts/ingest_documents.py
   ```
5. The course will automatically appear in the UI dropdown

## MongoDB Logging

PRISM optionally logs Q&A interactions to MongoDB Atlas (`prism.interactions`). Logging is only enabled for regular query flows -- flashcards and podcasts are not logged.

**Logged fields:** `student_id`, `degree`, `major`, `course`, `source_type`, `question`, `response_1`-`response_3`, `score_1`-`score_3`, `created_at`

If MongoDB is unavailable, PRISM continues to function normally.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph (state graph + checkpointing) |
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-large (3072d) |
| Vector Store | Pinecone (serverless, cosine similarity) |
| Web Search | Tavily API |
| TTS | OpenAI TTS-1 + MCP fallback |
| Frontend | Streamlit |
| Logging | MongoDB Atlas |
| Evaluation | DeepEval, RAGAS, Anthropic Claude (judge) |

## License

PRISM Adaptive Learning System - UNT Dissertation (2025-2026)
