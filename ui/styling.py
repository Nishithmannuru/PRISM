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
        
        /* Header Styling */
        .header-title {
            color: #00853C;
            font-size: 4.5em;
            font-weight: 900;
            margin-top: -0.5em;
            margin-bottom: -0.3em;
            letter-spacing: 2px;
            padding-top: 0.5em;
        }
        
        /* Subtitle Styling */
        .header-subtitle {
            color: #666666;
            font-size: 0.95em;
            font-weight: 500;
            margin-top: -0.2em;
            margin-bottom: 0.5em;
            font-style: italic;
            letter-spacing: 0.5px;
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
        
        /* Chat history container background */
        .chat-container {
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            min-height: 500px;
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

