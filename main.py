# main.py
"""
Main entry point for the Legal Assistant MVP application.

This script:
- Initializes the FastAPI application
- Sets up logging and configuration
- Starts the web server
- Handles command-line arguments
"""

import os
import sys
import argparse
import logging
import uvicorn
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("legal-assistant")

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Legal AI Assistant Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to (use 127.0.0.1 for local testing)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", help="Logging level")
    parser.add_argument("--no-samples", action="store_true", help="Don't include sample data in a new index")
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_args()
    
    # Log startup message
    logger.info("Starting Legal AI Assistant Server")
    logger.info(f"Host: {args.host}, Port: {args.port}, Workers: {args.workers}")
    
    # Check for required directories - use absolute paths
    project_root = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, "data")
    uploads_dir = os.path.join(data_dir, "uploads")
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Data directory: {data_dir}")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Set environment variables for API configuration
    if args.no_samples:
        logger.info("Sample data will be excluded from new index creation")
        os.environ["LEGAL_ASSISTANT_NO_SAMPLES"] = "1"
    
    # Make sure the current directory is in the Python path
    # This ensures the app module can be imported properly
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Start the server using the import string format
    uvicorn.run(
        "app.api:app",  # Use the import string format
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,  # Workers must be 1 when reload=True
        log_level=args.log_level
    )

if __name__ == "__main__":
    main() 