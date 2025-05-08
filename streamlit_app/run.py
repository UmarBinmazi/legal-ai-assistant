import streamlit.web.bootstrap as bootstrap
import os
import sys
import argparse

# Add the parent directory to the path so we can import modules properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_args():
    """Parse command line arguments for the Streamlit app."""
    parser = argparse.ArgumentParser(description="Legal AI Assistant UI")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", 
                      help="URL of the API server")
    return parser.parse_args()

# Run the Streamlit app
if __name__ == "__main__":
    args = parse_args()
    
    # Set environment variables for the UI to use
    os.environ["LEGAL_ASSISTANT_API_URL"] = args.api_url
    
    # Run the Streamlit app
    bootstrap.run("streamlit_app/ui.py", "", [], flag_options={}) 