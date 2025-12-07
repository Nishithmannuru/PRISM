# LangGraph Agentic Flow Documentation

## Overview

PRISM now uses a LangGraph-based agentic flow with multiple specialized agents that communicate through a structured state system. The system includes session-based memory for conversation context.

## Agent Flow

```
User Query
    ↓
[Query Refinement Agent]
    ├─→ Vague? → Ask Follow-up Questions → Wait for User Response
    └─→ Clear → [Relevance Agent]
                    ├─→ Not Relevant → Polite Decline → End
                    └─→ Relevant → [Course RAG Agent]
                                      ├─→ Found in Course → [Personalization Agent] → Response
                                      └─→ Not Found → [Web Search Agent] → [Personalization Agent] → Response
```

## Agents

### 1. Query Refinement Agent
- **Purpose**: Detects vague queries and asks clarifying questions
- **Location**: `core/nodes/query_refinement.py`
- **Functionality**:
  - Analyzes query clarity
  - Generates follow-up questions if needed
  - Refines queries based on user answers

### 2. Relevance Agent
- **Purpose**: Determines if a question is relevant to the course
- **Location**: `core/nodes/relevance.py`
- **Functionality**:
  - Uses course description from `prompts.yaml`
  - LLM-based classification
  - Returns relevance decision with reason

### 3. Course RAG Agent
- **Purpose**: Retrieves course content from Pinecone
- **Location**: `core/nodes/course_rag.py`
- **Functionality**:
  - Searches course vectors
  - Checks if content answers the question
  - Returns context and citations

### 4. Web Search Agent
- **Purpose**: Performs internet search when course content not found
- **Location**: `core/nodes/web_search.py`
- **Functionality**:
  - Uses Tavily API for internet search
  - Only called if: `relevant AND NOT found_in_course`
  - Returns search results and citations

### 5. Personalization Agent
- **Purpose**: Tailors response to student's background
- **Location**: `core/nodes/personalization.py`
- **Functionality**:
  - Adapts complexity based on degree level
  - Uses examples relevant to student's major
  - Generates final personalized response

## State Management

The system uses `AgentState` (TypedDict) to manage flow:
- Conversation history (LangChain messages)
- Query information
- Agent decisions (vague, relevant, found, etc.)
- Course content and citations
- Web search results
- User context
- Final response

## Memory

- **Type**: LangGraph in-memory checkpointing
- **Scope**: Session-based (per Streamlit session)
- **Thread ID**: Based on student ID
- **Persistence**: Cleared when session ends

## Configuration

### Course Descriptions
Add course descriptions in `config/prompts.yaml`:
```yaml
course_descriptions:
  Neuroquest: |
    Description of what the course covers...
```

### Agent Prompts
All agent prompts are configurable in `config/prompts.yaml`:
- Query refinement prompts
- Relevance detection prompts
- Personalization settings

## Usage

The system is automatically used when you run:
```bash
streamlit run app.py
```

The flow handles:
- Vague queries → Follow-up questions
- Irrelevant queries → Polite decline
- Relevant + Found → Course RAG → Personalization
- Relevant + Not Found → Web Search → Personalization

## A2A Communication

Agents communicate through:
1. **State-based**: Primary method - agents read/write to shared state
2. **Direct calls**: Agents can call other agents' functions when needed
3. **Message passing**: Through LangChain messages in state

## Future Enhancements

- PostgreSQL checkpointing for production
- More sophisticated relevance detection
- Agent-to-agent direct messaging
- Response evaluation and regeneration
- Knowledge graph integration

