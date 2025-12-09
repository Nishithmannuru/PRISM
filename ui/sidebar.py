"""Sidebar UI components for PRISM."""

import streamlit as st


def reset_session():
    """Resets the session to initial state for a new chat."""
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome to PRISM! Please fill out the form on the left to start your adaptive learning session."}
    ]
    st.session_state.user_context = {
        'student_id': None,
        'course': None,
        'major': None,
        'degree': None,
        'is_ready': False
    }
    # Clear flashcards
    if 'flashcards' in st.session_state:
        st.session_state.flashcards = []
    if 'flashcard_topic' in st.session_state:
        st.session_state.flashcard_topic = None
    # Clear follow-up state
    if 'follow_up_needed' in st.session_state:
        st.session_state.follow_up_needed = False
    if 'original_query' in st.session_state:
        del st.session_state.original_query
    # Clear input fields
    if 'student_id_input' in st.session_state:
        st.session_state.student_id_input = ""
    if 'major_input' in st.session_state:
        st.session_state.major_input = ""
    if 'course_dropdown' in st.session_state:
        st.session_state.course_dropdown = "Select Course..."
    if 'degree_dropdown' in st.session_state:
        st.session_state.degree_dropdown = "Select Degree..."


def render_new_chat_button():
    """Renders the New Chat button at the top of the sidebar."""
    if st.button("+ New Chat", key="new_chat_button", use_container_width=True):
        reset_session()
        st.rerun()


def render_sidebar(course_options, degree_options, handle_start_session):
    """Renders the complete sidebar with user context and session setup."""
    with st.sidebar:
        # Chatbot name and branding at the top - very compact
        st.markdown(
            '<div style="display: flex; align-items: center; gap: 6px; margin-bottom: 6px; padding: 2px 0;">'
            '<span style="font-size: 1.3em;">🧠</span>'
            '<div>'
            '<h1 style="margin: 0; color: #00853C; font-size: 1.3em; font-weight: 700; line-height: 1.1;">PRISM</h1>'
            '<p style="margin: 0; color: #666; font-size: 0.6em; font-style: italic; line-height: 1.1;">Adaptive Learning System</p>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")
        
        # New Chat button
        render_new_chat_button()
        
        st.subheader("Session Setup")
        
        # Check if session is already active
        if st.session_state.user_context['is_ready']:
            st.success("✅ Session Active")
            st.markdown("**Current Session:**")
            st.markdown(f"**Course:** {st.session_state.user_context['course']}")
            st.markdown(f"**Degree:** {st.session_state.user_context['degree']}")
            st.markdown(f"**Major:** {st.session_state.user_context['major']}")
            st.markdown("---")
            
            if st.button("End Session", use_container_width=True):
                reset_session()
                st.rerun()
        else:
            # Input Form - fields are enabled when session is not active
            # They will be disabled once session starts (handled by session state)
            st.text_input(
                "Student ID",
                key="student_id_input",
                placeholder="e.g., 10005578",
                disabled=st.session_state.user_context['is_ready'],
                help="Unique identifier for the student"
            )
            
            st.selectbox(
                "Degree",
                options=degree_options,
                key="degree_dropdown",
                disabled=st.session_state.user_context['is_ready']
            )
            
            st.text_input(
                "Major",
                key="major_input",
                placeholder="e.g., Computer Science",
                disabled=st.session_state.user_context['is_ready']
            )
            
            st.selectbox(
                "Course",
                options=course_options,
                key="course_dropdown",
                disabled=st.session_state.user_context['is_ready']
            )
            
            # Start Session Button - no separator before it
            if st.button("Start PRISM Session", use_container_width=True):
                handle_start_session(course_options, degree_options)

