"""Professor-pathway import for the durable SQLite feedback store."""
from .sqlite_store import SQLiteStore

FeedbackStore = SQLiteStore

__all__ = ['FeedbackStore', 'SQLiteStore']
