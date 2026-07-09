import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

@dataclass
class Document:
    text: str
    metadata: Dict[str, Any]

class DocumentLoader:
    """Loads and normalizes multi-format documents (PDF, MD, TXT, images)."""
    
    def load(self, filepath: str) -> List[Document]:
        """Loads target file based on extension types."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Target file not found at: {filepath}")
            
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == ".pdf":
            return self._load_pdf(filepath)
        elif ext in [".txt", ".md"]:
            return self._load_text(filepath)
        elif ext in [".png", ".jpg", ".jpeg"]:
            return self._load_image_ocr(filepath)
        else:
            return self._load_text(filepath)

    def _load_pdf(self, filepath: str) -> List[Document]:
        documents = []
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                for idx, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    # Table extraction support
                    tables = page.extract_tables()
                    table_str = ""
                    if tables:
                        table_str = "\n\nExtracted Table Data:\n"
                        for table in tables:
                            table_str += "\n".join([str(row) for row in table])
                            
                    full_text = text + table_str
                    if full_text.strip():
                        documents.append(Document(
                            text=full_text,
                            metadata={
                                "source": os.path.basename(filepath),
                                "page": idx,
                                "type": "pdf"
                            }
                        ))
        except Exception as e:
            print(f"Error reading PDF {filepath}: {e}. Falling back to plain-text reader.")
            return self._load_text(filepath)
        return documents

    def _load_text(self, filepath: str) -> List[Document]:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return [Document(
                text=content,
                metadata={
                    "source": os.path.basename(filepath),
                    "page": 1,
                    "type": "text"
                }
            )]
        except Exception as e:
            print(f"Error reading text file: {e}")
            return []

    def _load_image_ocr(self, filepath: str) -> List[Document]:
        """Simulates image OCR extraction and auto-generates semantic descriptors."""
        filename = os.path.basename(filepath)
        # Mock OCR output based on filename trigger
        ocr_text = f"[OCR Image Extraction: {filename}]\n"
        ocr_text += "Visual flow shows entity relationships between user query router and dense FAISS indexes. "
        ocr_text += "Subgraphs indicate Community Detection communities with PageRank coefficients matching forecast profiles."
        
        return [Document(
            text=ocr_text,
            metadata={
                "source": filename,
                "page": 1,
                "type": "ocr_image"
            }
        )]
