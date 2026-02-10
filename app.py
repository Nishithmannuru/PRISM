import streamlit as st
import logging
from pathlib import Path

# Import UI components
from ui import styling, sidebar, chat, session
from core.agent import PRISMAgent, get_prism_agent
from prism_logging.mongo_logger import log_interaction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_available_courses():
    """Get list of available courses.
    
    Reads from course_descriptions in prompts.yaml so the dropdown works
    on Streamlit Cloud (where the courses/ folder is gitignored).
    Falls back to scanning the local courses/ directory if config is missing.
    """
    import yaml
    
    # Courses to hide from the dropdown (still in vector store but not shown)
    HIDDEN_COURSES = {"Neuroquest"}
    
    courses = ["Select Course..."]
    
    # Primary: read from prompts.yaml (works on Streamlit Cloud)
    try:
        config_path = Path(__file__).parent / "config" / "prompts.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        course_descriptions = config.get("course_descriptions", {})
        for course_name in course_descriptions:
            if course_name not in HIDDEN_COURSES:
                courses.append(course_name)
    except Exception as e:
        logger.warning(f"Could not read course list from prompts.yaml: {e}")
    
    # Fallback: scan local courses/ directory
    if len(courses) <= 1:
        from config.settings import COURSES_PATH
        courses_dir = Path(COURSES_PATH)
        if courses_dir.exists():
            for course_folder in courses_dir.iterdir():
                if course_folder.is_dir() and course_folder.name not in HIDDEN_COURSES:
                    courses.append(course_folder.name)
    
    return courses if len(courses) > 1 else ["Select Course..."]


def generate_response(user_query):
    """
    Generate response using LangGraph agentic system.
    
    Args:
        user_query: User's question
        
    Returns:
        Formatted response with citations or follow-up questions
    """
    try:
        agent = get_prism_agent()
        if agent is None:
            return "Error: PRISM agent not available. Please check your configuration."
        
        course_name = st.session_state.user_context.get('course')
        user_context = st.session_state.user_context
        
        if not course_name or course_name == "Select Course...":
            return "Please select a course to ask questions."
        
        # Handle simple greetings directly
        query_lower = user_query.lower().strip()
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if query_lower in greetings or any(query_lower.startswith(g) for g in greetings):
            return f"Hello! I'm PRISM, your adaptive learning assistant for {course_name}. How can I help you with your course today?"
        
        # Get conversation history for context
        # Exclude the last message (current query) since it's passed separately as 'query'
        conversation_history = None
        if 'chat_history' in st.session_state and len(st.session_state.chat_history) > 0:
            # Get all messages except the last one (which is the current query we just added)
            history_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.chat_history[:-1]  # Exclude last message (current query)
                if msg["role"] in ["user", "assistant"] and msg.get("content")  # Filter out None content
            ]
            # Only set conversation_history if we have messages (not just the current one)
            if history_messages:
                conversation_history = history_messages
        
        # Generate thread ID from session (for memory)
        thread_id = f"session_{st.session_state.user_context.get('student_id', 'default')}"
        
        # Process query through agentic flow
        result = agent.process_query(
            query=user_query,
            course_name=course_name,
            user_context=user_context,
            conversation_history=conversation_history,
            thread_id=thread_id
        )
        
        # Store web search flag in session state for UI to check
        st.session_state._last_query_used_web_search = result.get("used_web_search", False)
        
        # Handle follow-up questions (one at a time)
        if result.get("needs_follow_up"):
            follow_up_questions = result.get("follow_up_questions", [])
            if follow_up_questions:
                # Ask only the first question (one at a time)
                follow_up_question = follow_up_questions[0] if follow_up_questions else "Could you please provide more details?"
                response = f"I need a bit more information to help you better. {follow_up_question}"
                
                # Store follow-up state
                st.session_state.follow_up_needed = True
                st.session_state.follow_up_questions = [follow_up_question]  # Store single question
                st.session_state.original_query = user_query
                
                return response
        
        # Clear follow-up state if we got here
        if 'follow_up_needed' in st.session_state:
            st.session_state.follow_up_needed = False
        
        # Log to MongoDB (only for completed regular queries, not follow-ups)
        # Only log if we have response history and it's a regular query (not flashcards/podcasts)
        response_history = result.get("response_history", [])
        source_type = result.get("source_type")
        
        if response_history and source_type and not result.get("needs_follow_up"):
            try:
                # Extract response and score data
                response_1 = response_history[0].get("response", "") if len(response_history) > 0 else ""
                score_1 = response_history[0].get("score", 0.0) if len(response_history) > 0 else 0.0
                
                response_2 = response_history[1].get("response", "") if len(response_history) > 1 else None
                score_2 = response_history[1].get("score", 0.0) if len(response_history) > 1 else None
                
                response_3 = response_history[2].get("response", "") if len(response_history) > 2 else None
                score_3 = response_history[2].get("score", 0.0) if len(response_history) > 2 else None
                
                # Build logging payload
                log_payload = {
                    "student_id": str(user_context.get("student_id", "")),
                    "degree": str(user_context.get("degree", "")),
                    "major": str(user_context.get("major", "")),
                    "course": str(course_name),
                    "source_type": source_type,  # "course" or "web"
                    "question": user_query,
                    "response_1": response_1,
                    "score_1": float(score_1),
                }
                
                # Add optional response_2 and score_2
                if response_2 is not None:
                    log_payload["response_2"] = response_2
                    log_payload["score_2"] = float(score_2)
                
                # Add optional response_3 and score_3
                if response_3 is not None:
                    log_payload["response_3"] = response_3
                    log_payload["score_3"] = float(score_3)
                
                # Log to MongoDB (non-blocking - errors are handled internally)
                log_interaction(log_payload)
                
            except Exception as e:
                # Don't break the app if logging fails
                logger.warning(f"Failed to log interaction to MongoDB: {e}")
        
        # Return response
        return result.get("response", "No response generated.")
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"I encountered an error while processing your question. Please try again. Error: {str(e)}"


# --- Main Application Layout ---
def main():
    # Initialize styling
    styling.set_streamlit_config()
    
    # Initialize session state
    session.initialize_session_state()
    
    # Get available courses from courses directory
    COURSE_OPTIONS = get_available_courses()
    
    DEGREE_OPTIONS = [
        "Select Degree...",
        "Bachelor of Science",
        "Master of Science",
        "Doctor of Philosophy"
    ]
    
    # Render sidebar
    sidebar.render_sidebar(
        COURSE_OPTIONS,
        DEGREE_OPTIONS,
        session.handle_start_session
    )
    
    # Render main chat interface
    chat.render_chat_interface(generate_response)
    
    # --- Copyright Footer ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="footer">© PRISM Adaptive Learning System 2025 (UNT Dissertation POC)</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
