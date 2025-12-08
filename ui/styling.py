"""Styling and theme configuration for PRISM UI."""

import streamlit as st


def set_streamlit_config():
    """Sets up custom styling and page configuration."""
    st.set_page_config(
        page_title="PRISM Adaptive Learning",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Reduce sidebar width and make it more compact
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            min-width: 240px !important;
            max-width: 280px !important;
        }
        
        /* Make sidebar scrollable if needed but reduce all spacing */
        [data-testid="stSidebar"] .element-container {
            margin-bottom: 0.2rem !important;
            padding: 0.1rem 0 !important;
        }
        
        /* Reduce label sizes */
        [data-testid="stSidebar"] label {
            font-size: 0.85em !important;
            margin-bottom: 0.2rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Inject custom CSS for UNT Green theme and professional look
    st.markdown(
        """
        <style>
        /* Main Theme Colors (UNT Green) */
        .stButton>button {
            background-color: #00853C; /* Dark Green */
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #00662D; /* Darker Green on Hover */
        }
        
        /* New Chat Button Styling */
        .new-chat-button {
            background-color: #00853C;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
            width: 100%;
            margin-bottom: 15px;
        }
        
        /* Chat message styling */
        .stChatMessage {
            padding: 0.75rem 1rem;
            margin-bottom: 1rem;
        }
        
        /* User message styling (right side) */
        .stChatMessage[data-testid="user"] {
            background-color: #e3f2fd;
            border-radius: 18px 18px 4px 18px;
            margin-left: auto;
            max-width: 75%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 12px 16px;
        }
        
        /* Assistant message styling (left side) */
        .stChatMessage[data-testid="assistant"] {
            background-color: #fce4ec;
            border-radius: 18px 18px 18px 4px;
            margin-right: auto;
            max-width: 75%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 12px 16px;
        }
        
        /* Avatar styling */
        .stChatMessage .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
        }
        
        /* Main content area */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Sidebar improvements */
        .stSidebar {
            background-color: #fafafa;
        }
        
        /* Reduce sidebar element spacing - extremely compact */
        .stSidebar .element-container {
            margin-bottom: 0.2rem !important;
            padding: 0.1rem 0 !important;
        }
        
        .stSidebar .stSubheader {
            margin-top: 0.2rem !important;
            margin-bottom: 0.2rem !important;
            font-size: 1em !important;
            padding-top: 0.2rem !important;
        }
        
        .stSidebar .stSuccess {
            padding: 0.25rem 0.4rem !important;
            margin-bottom: 0.2rem !important;
            font-size: 0.8em !important;
        }
        
        .stSidebar .stMarkdown {
            margin-bottom: 0.1rem !important;
            font-size: 0.8em !important;
            line-height: 1.2 !important;
        }
        
        .stSidebar .stTextInput > div > div > input {
            padding: 0.3rem 0.4rem !important;
            font-size: 0.85em !important;
            margin-bottom: 0.1rem !important;
        }
        
        .stSidebar .stSelectbox > div > div > select {
            padding: 0.3rem 0.4rem !important;
            font-size: 0.85em !important;
            margin-bottom: 0.1rem !important;
        }
        
        .stSidebar hr {
            margin: 0.3rem 0 !important;
        }
        
        /* Reduce button padding */
        .stSidebar .stButton > button {
            padding: 0.4rem 0.8rem !important;
            font-size: 0.85em !important;
            margin-bottom: 0.2rem !important;
        }
        
        /* Reduce label font size and spacing */
        .stSidebar label {
            font-size: 0.8em !important;
            margin-bottom: 0.15rem !important;
        }
        
        /* Input field styling */
        .stChatInputContainer {
            position: sticky;
            bottom: 0;
            background-color: white;
            padding: 1rem 0;
            z-index: 100;
        }
        
        /* Chat Input Placeholder Styling */
        .stChatInputContainer textarea::placeholder {
            color: #999999;
            opacity: 0.7;
            font-style: italic;
        }
        
        /* Sidebar Styling for clean look */
        .stSidebar .stSelectbox, .stSidebar .stTextInput {
            padding: 5px;
            border-radius: 6px;
        }
        
        /* Chat history container background - removed min-height to fix white box */
        .chat-container {
            border: none;
            padding: 0;
        }
        
        /* Footer / Copyright */
        .footer {
            font-size: 0.8em;
            color: #888888;
            text-align: center;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

