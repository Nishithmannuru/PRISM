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
        # New Chat button at the top
        render_new_chat_button()
        
        st.subheader("User Context & Session Setup")
        
        # Check if session is already active
        if st.session_state.user_context['is_ready']:
            st.success("Session Active. Ready for Chat.")
            st.info(f"**Course:** {st.session_state.user_context['course']}")
            st.info(f"**Degree:** {st.session_state.user_context['degree']}")
            st.info(f"**Major:** {st.session_state.user_context['major']}")
            st.markdown("---")
            
            if st.button("End Session"):
                reset_session()
                st.rerun()
        else:
            # Input Form - fields are enabled when session is not active
            # They will be disabled once session starts (handled by session state)
            st.text_input(
                "Student ID (Unique Identifier)",
                key="student_id_input",
                placeholder="e.g., 10005578",
                disabled=st.session_state.user_context['is_ready']
            )
            
            st.selectbox(
                "Degree",
                options=degree_options,
                key="degree_dropdown",
                disabled=st.session_state.user_context['is_ready']
            )
            
            st.text_input(
                "Major (Type In)",
                key="major_input",
                placeholder="e.g., Computer Science",
                disabled=st.session_state.user_context['is_ready']
            )
            
            st.selectbox(
                "Course Code & Name",
                options=course_options,
                key="course_dropdown",
                disabled=st.session_state.user_context['is_ready']
            )
            
            st.markdown("---")
            
            # Start Session Button
            if st.button("Start PRISM Session"):
                handle_start_session(course_options, degree_options)

