from typing import List, Dict, Any
from src.python.insight_rag.ingestion.document_loader import Document

class RecursiveChunker:
    """Recursively splits document text into semantic chunks with sliding overlaps."""
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 60):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits list of loaded documents into overlapping chunks."""
        chunks = []
        for doc in documents:
            text = doc.text
            # Identify if table data exists to avoid severing table lines
            is_table = "Extracted Table Data" in text
            
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = start + self.chunk_size
                
                # Check for word boundary on ending index
                if end < len(text):
                    # Adjust to space
                    space_pos = text.rfind(" ", start, end)
                    if space_pos > start:
                        end = space_pos
                        
                chunk_text = text[start:end].strip()
                if chunk_text:
                    meta = doc.metadata.copy()
                    meta["chunk_id"] = f"{meta['source']}_c{chunk_idx}"
                    meta["is_table"] = is_table
                    chunks.append(Document(
                        text=chunk_text,
                        metadata=meta
                    ))
                    
                chunk_idx += 1
                start += self.chunk_size - self.chunk_overlap
                
                # Prevent infinite loops
                if self.chunk_size <= self.chunk_overlap:
                    break
        return chunks
