"""Chat UI components for PRISM."""

import streamlit as st


def display_chat_history():
    """Renders the chat history from session state."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input(user_query, generate_response):
    """
    Handles user input and generates response.
    Supports follow-up questions for vague queries.
    
    Args:
        user_query: The user's question/input
        generate_response: Function that generates the response based on query
    """
    if not user_query:
        return
    
    # Check if this is a follow-up answer
    if st.session_state.get('follow_up_needed', False):
        # This is an answer to a follow-up question
        from core.agent import PRISMAgent
        
        agent = PRISMAgent()
        course_name = st.session_state.user_context.get('course')
        user_context = st.session_state.user_context
        thread_id = f"session_{st.session_state.user_context.get('student_id', 'default')}"
        
        # Refine and process
        result = agent.refine_query_with_follow_up(
            original_query=st.session_state.original_query,
            follow_up_answer=user_query,
            course_name=course_name,
            user_context=user_context,
            thread_id=thread_id
        )
        
        # Store follow-up answer
        st.session_state.chat_history.append({"role": "user", "content": f"Follow-up: {user_query}"})
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner(f"PRISM Agent (Course: {course_name}) is processing..."):
                response = result.get("response", "Processing your refined question...")
        
        # Clear follow-up state
        st.session_state.follow_up_needed = False
        if 'follow_up_questions' in st.session_state:
            del st.session_state.follow_up_questions
        if 'original_query' in st.session_state:
            del st.session_state.original_query
    else:
        # Regular query
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
        display_chat_history()
        
        if st.session_state.user_context['is_ready']:
            user_input = st.chat_input(
                "Ask your questions here...",
                disabled=False
            )
            if user_input:
                handle_user_input(user_input, generate_response)
        else:
            st.chat_input("Enter details on the left to activate the chat.", disabled=True)

