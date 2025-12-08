"""Chat UI components for PRISM."""

import streamlit as st


def display_chat_history():
    """Renders the chat history from session state with user messages on right and AI on left."""
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        # User messages appear on the right with person icon
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        # Assistant messages appear on the left with brain icon
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(content)


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
        
        # Store follow-up answer (without "Follow-up:" prefix for cleaner conversation)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Check if still needs follow-up (conversational flow)
        if result.get("needs_follow_up"):
            # Still vague, ask another follow-up question
            follow_up_questions = result.get("follow_up_questions", [])
            if follow_up_questions:
                follow_up_question = follow_up_questions[0]
                response = f"I need a bit more information. {follow_up_question}"
                # Keep follow-up state active for next question
                st.session_state.original_query = st.session_state.original_query + " " + user_query  # Accumulate context
                st.session_state.follow_up_questions = [follow_up_question]
            else:
                response = result.get("response", "Processing your refined question...")
                # Clear follow-up state
                st.session_state.follow_up_needed = False
                if 'follow_up_questions' in st.session_state:
                    del st.session_state.follow_up_questions
                if 'original_query' in st.session_state:
                    del st.session_state.original_query
        else:
            # Query is now clear, show response
            response = result.get("response", "Processing your refined question...")
            # Clear follow-up state
            st.session_state.follow_up_needed = False
            if 'follow_up_questions' in st.session_state:
                del st.session_state.follow_up_questions
            if 'original_query' in st.session_state:
                del st.session_state.original_query
        
        # Display response
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(response)
    else:
        # Regular query
        # Store User Query in State
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Generate and display response
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner(f"PRISM Agent (Course: {st.session_state.user_context['course']}) is thinking..."):
                response = generate_response(user_query)
    
    # Store Agent Response in State
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.rerun()


def render_chat_interface(generate_response):
    """Renders the main chat interface."""
    # Main chat area - no header, just chat
    display_chat_history()
    
    # Chat input at the bottom
    if st.session_state.user_context['is_ready']:
        user_input = st.chat_input(
            "Ask your questions here...",
            disabled=False
        )
        if user_input:
            handle_user_input(user_input, generate_response)
    else:
        st.chat_input("Enter details on the left to activate the chat.", disabled=True)

