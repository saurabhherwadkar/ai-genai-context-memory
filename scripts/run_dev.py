"""Development server launcher script."""

import os  # Environment variable setup
import sys  # System path manipulation
from pathlib import Path  # Path handling

# Add the src directory to the Python path for imports
project_root = Path(__file__).resolve().parent.parent  # Navigate to project root
sys.path.insert(0, str(project_root / "src"))  # Add src to import path


def main() -> None:
    """Launch the development server with hot reload enabled."""
    import uvicorn  # ASGI server

    # Set environment to development
    os.environ.setdefault("APP_ENV", "dev")  # Default to dev environment

    # Run the server with development settings
    uvicorn.run(
        "memory_graph.main:app",  # Application import path
        host="127.0.0.1",  # Localhost only for development
        port=8000,  # Default port
        reload=True,  # Enable hot reload on code changes
        reload_dirs=[str(project_root / "src")],  # Watch src directory
        log_level="debug",  # Verbose logging for development
    )


if __name__ == "__main__":
    main()  # Run when executed directly
