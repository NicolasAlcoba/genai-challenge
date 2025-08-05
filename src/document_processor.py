import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Union

import pdfplumber

# Import all available chunkers
from .base_chunk import BaseChunker
from .sentence_chunk import SentenceChunker
from .semantic_chunk import SemanticChunker
from .structure_chunk import StructuralChunker
from .hybrid_chunk import HybridChunker
from .paragraph_chunk import ParagraphChunker
from .sliding_window_chunk import SlidingWindowChunker

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Document processor that can use different chunking strategies"""
    
    # Available chunking strategies
    CHUNKING_STRATEGIES = {
        'simple': None,  # Use built-in simple chunking
        'sentence': SentenceChunker,
        'semantic': SemanticChunker,
        'structural': StructuralChunker,
        'hybrid': HybridChunker,
        'paragraph': ParagraphChunker,
        'sliding_window': SlidingWindowChunker,
    }

    def __init__(self, 
                 chunk_size: int = 500, 
                 overlap: int = 50,
                 chunking_strategy: str = 'simple',
                 chunker_kwargs: Optional[Dict] = None):
        """
        Initialize DocumentProcessor with specified chunking strategy
        
        Args:
            chunk_size: Size of chunks (for simple chunking or as parameter for other chunkers)
            overlap: Overlap between chunks (for simple chunking)
            chunking_strategy: Name of chunking strategy to use
            chunker_kwargs: Additional arguments to pass to the chunker
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunking_strategy = chunking_strategy
        
        # Initialize the chunker if using advanced chunking
        if chunking_strategy != 'simple':
            if chunking_strategy not in self.CHUNKING_STRATEGIES:
                raise ValueError(f"Unknown chunking strategy: {chunking_strategy}. Available: {list(self.CHUNKING_STRATEGIES.keys())}")
            
            ChunkerClass = self.CHUNKING_STRATEGIES[chunking_strategy]
            chunker_kwargs = chunker_kwargs or {}
            
            # Pass chunk_size to chunker if not already specified
            if 'chunk_size' not in chunker_kwargs:
                chunker_kwargs['chunk_size'] = chunk_size
            
            self.chunker = ChunkerClass(**chunker_kwargs)
        else:
            self.chunker = None

    def process_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """Process PDF using the configured chunking strategy"""
        
        # If using advanced chunking, delegate to the chunker
        if self.chunker:
            logger.info(f"Using {self.chunking_strategy} chunking strategy")
            return self.chunker.process_pdf(pdf_path)
        
        # Otherwise, use simple chunking
        logger.info("Using simple character-based chunking")
        
        # Handle URL downloads
        if pdf_path.startswith(("http://", "https://")):
            logger.info(f"Downloading PDF from URL: {pdf_path}")
            try:
                # Download PDF to temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    urllib.request.urlretrieve(pdf_path, tmp_file.name)
                    local_pdf_path = tmp_file.name

                logger.info(f"Downloaded PDF to: {local_pdf_path}")

                try:
                    result = self._process_local_pdf(local_pdf_path, pdf_path)
                    return result
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(local_pdf_path)
                    except:
                        pass

            except Exception as e:
                logger.error(f"Failed to download PDF from URL: {e}")
                raise FileNotFoundError(f"Could not download PDF from URL: {pdf_path}")
        else:
            # Handle local file
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            return self._process_local_pdf(pdf_path, pdf_path)

    def _process_local_pdf(
        self, local_path: str, original_source: str
    ) -> List[Dict[str, str]]:
        logger.info(f"Processing PDF: {local_path}")
        all_chunks = []
        chunk_id = 0

        try:
            with pdfplumber.open(local_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text from the page
                    text = page.extract_text()

                    if text and text.strip():
                        text = self._clean_text(text)

                        page_chunks = self.chunk_text(text, page_num)

                        for chunk in page_chunks:
                            chunk["chunk_id"] = str(chunk_id)
                            chunk["source"] = (
                                os.path.basename(original_source)
                                if not original_source.startswith("http")
                                else "FAA_Airplane_Handbook.pdf"
                            )
                            all_chunks.append(chunk)
                            chunk_id += 1

        except Exception as e:
            raise RuntimeError(f"Error processing PDF {local_path}: {e}")

        if not all_chunks:
            raise ValueError(f"No text could be extracted from PDF: {local_path}")

        logger.info(f"Successfully extracted {len(all_chunks)} chunks from PDF")
        return all_chunks

    def chunk_text(self, text: str, page_num: int) -> List[Dict[str, str]]:
        if not text or not text.strip():
            return []

        chunks = []
        text = text.strip()

        # Simple character-based chunking with overlap
        start = 0
        local_chunk_id = 0

        while start < len(text):
            # Position
            end = start + self.chunk_size

            # If this isn't the last chunk, try to break at word boundary
            if end < len(text):
                # Look for last space within the chunk to avoid breaking words
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            # Extract chunk
            chunk_text = text[start:end].strip()

            if chunk_text:  # Only add non-empty chunks
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": str(page_num),
                        "local_chunk_id": str(local_chunk_id),
                        "chunking_method": "simple"  # Add method identifier
                    }
                )
                local_chunk_id += 1

            # Move start position with overlap
            start = end - self.overlap
            if start <= 0:
                start = end

            if start >= len(text):
                break

        return chunks

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and formatting
        """
        import re

        # Replace multiple whitespaces with single space
        text = re.sub(r"\s+", " ", text)

        # Remove extra newlines but preserve paragraph breaks
        text = re.sub(r"\n\s*\n", "\n\n", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        return {
            "filename": Path(pdf_path).name,
            "path": pdf_path,
            "status": "metadata extraction not implemented",
            "chunking_strategy": self.chunking_strategy  # Add strategy info
        }
        
    def get_chunker_stats(self, chunks: List[Dict[str, str]]) -> Dict[str, any]:
        """Get statistics about the chunking results"""
        if self.chunker and hasattr(self.chunker, 'get_stats'):
            return self.chunker.get_stats(chunks)
        
        # Basic stats for simple chunking
        if not chunks:
            return {}
            
        chunk_lengths = [len(c['text']) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunks),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "chunking_method": "simple"
        }
