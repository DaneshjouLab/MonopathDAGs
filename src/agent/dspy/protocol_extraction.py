from typing import Protocol, List, Optional, Dict
from dataclasses import dataclass

@dataclass
class ExtractedDocument:
    """Structured representation of the extracted article."""
    article_id: str
    title: Optional[str]
    text: str  # Full clean body text (excluding irrelevant web/PDF noise)
    sections: Optional[Dict[str, str]] = None  # Optional: keyed by section headings
    images: Optional[List[bytes]] = None       # Raw image content (or paths/URLs if preferred)
    metadata: Optional[Dict[str, str]] = None  # Optional metadata (e.g., journal, date)

class ArticleExtractionProtocol(Protocol):
    def extract_article(self, file_path: str) -> ExtractedDocument:
        """
        Extracts clean clinical content from a case report PDF or XML.

        Parameters:
            file_path (str): Local path to the article file (PDF/XML).

        Returns:
            ExtractedDocument: Clean, structured representation of article contents.
        """
        ...
