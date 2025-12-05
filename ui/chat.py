"""Chat UI components for PRISM."""

import streamlit as st
import time


def display_chat_history():
    """Renders the chat history from session state."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input(user_query, generate_response):
    """
    Handles user input and generates response.
    
    Args:
        user_query: The user's question/input
        generate_response: Function that generates the response based on query
    """
    if not user_query:
        return
    
    # Store User Query in State
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner(f"PRISM Agent (Course: {st.session_state.user_context['course']}) is thinking..."):
            response = generate_response(user_query)
    
    # Store Agent Response in State
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.rerun()


def render_chat_interface(generate_response):
    """Renders the main chat interface."""
    st.markdown('<div class="header-title">PRISM</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="header-subtitle">Personalized Retrieval-Integrated System for Multimodal Adaptive Learning</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.subheader("System Status")
        if st.session_state.user_context['is_ready']:
            st.markdown("**:green[Agent Status:]** Online")
            st.markdown(f"**:green[Course Data:]** Ready")
            st.markdown("**:green[System:]** Operational")
        else:
            st.markdown("**:red[Agent Status:]** Offline")
            st.markdown("**:red[Course Data:]** Unloaded")
            st.markdown("**:orange[System:]** Waiting for Session")
    
    with col2:
        st.subheader("Adaptive Chat Interaction")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        display_chat_history()
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.user_context['is_ready']:
            user_input = st.chat_input(
                "Ask your questions here...",
                disabled=False
            )
            if user_input:
                handle_user_input(user_input, generate_response)
        else:
            st.chat_input("Enter details on the left to activate the chat.", disabled=True)

