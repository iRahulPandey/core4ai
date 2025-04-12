"""
Core4AI: Contextual Optimization and Refinement Engine for AI
-------------------------------------------------------------

A package for transforming basic user queries into optimized LLM prompts
using MLflow Prompt Registry.

This is a placeholder release. Full functionality coming soon!
"""

# Import the public API
from .engine import describe_project

__version__ = "0.1.3"


def coming_soon():
    """Display a message about the upcoming full release."""
    message = """
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║                  📣 Core4AI Coming Soon! 📣                     ║
    ║                                                                ║
    ║  Thank you for your interest in Core4AI!                       ║
    ║                                                                ║
    ║  This package is currently a placeholder to reserve the name.  ║
    ║  We're actively developing the full functionality and will     ║
    ║  release soon.                                                 ║
    ║                                                                ║
    ║  Stay tuned for updates at:                                    ║
    ║  https://github.com/iRahulPandey/core4ai                       ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """
    print(message)
    return None


# Make describe_project available at the package level
__all__ = ["coming_soon", "describe_project"]

# Automatically show the coming soon message when the package is imported
coming_soon()