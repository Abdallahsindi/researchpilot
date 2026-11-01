"""Document ingestion, retrieval, and evidence-grounded question answering.

Shared by the FastAPI app and the Kaggle notebook. Nothing in this package imports the API
configuration or heavy ML libraries at import time; models load lazily on first use.
"""
