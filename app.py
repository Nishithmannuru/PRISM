import streamlit as st
import time

# Import UI components
from ui import styling, sidebar, chat, session

# Placeholder response generator - will be replaced with agentic RAG system
def generate_response(user_query):
    """
    Placeholder function for generating responses.
    This will be replaced with the actual agentic RAG system.
    """
    time.sleep(1)  # Simulate processing time
    
    # Placeholder response - will be replaced with actual agentic RAG
    response = f"Thank you for your question: '{user_query}'. The agentic RAG system will process this query based on your course materials and context."
    response += f"\n\n**Your Context:** {st.session_state.user_context['course']} | {st.session_state.user_context['degree']} | {st.session_state.user_context['major']}"
    response += "\n\n*(This is a placeholder. The full agentic RAG system will be integrated next.)*"
    
    return response


# --- Main Application Layout ---
def main():
    # Initialize styling
    styling.set_streamlit_config()
    
    # Initialize session state
    session.initialize_session_state()
    
    # Define mock data for dropdowns
    COURSE_OPTIONS = [
        "Select Course...",
        "CSCE 5310: Database Systems",
        "CSCE 5320: Software Engineering",
        "CSCE 5410: Artificial Intelligence",
        "CSCE 5520: Machine Learning"
    ]
    
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
    st.markdown(
        '<div class="footer">© PRISM Adaptive Learning System 2025 (UNT Dissertation POC)</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
