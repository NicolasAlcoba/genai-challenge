import logging
import os
import re
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Represents a section of the document"""
    title: str
    content: str
    page_num: int
    section_type: str  # 'header', 'paragraph', 'list', etc.
    level: int  # Header level (1-6)


class BaseChunker:
    """Base class for all chunking strategies"""
    
    def __init__(
        self, 
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def process_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """Process PDF with smart chunking"""
        if pdf_path.startswith(("http://", "https://")):
            logger.info(f"Downloading PDF from URL: {pdf_path}")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    urllib.request.urlretrieve(pdf_path, tmp_file.name)
                    local_pdf_path = tmp_file.name

                logger.info(f"Downloaded PDF to: {local_pdf_path}")
                try:
                    result = self._process_local_pdf(local_pdf_path, pdf_path)
                    return result
                finally:
                    try:
                        os.unlink(local_pdf_path)
                    except:
                        pass

            except Exception as e:
                logger.error(f"Failed to download PDF from URL: {e}")
                raise FileNotFoundError(f"Could not download PDF from URL: {pdf_path}")
        else:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            return self._process_local_pdf(pdf_path, pdf_path)

    def _process_local_pdf(self, local_path: str, original_source: str) -> List[Dict[str, str]]:
        """Process local PDF with enhanced extraction"""
        logger.info(f"Processing PDF: {local_path} with {self.__class__.__name__}")
        all_sections = []
        
        try:
            with pdfplumber.open(local_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text with layout preservation
                    text = page.extract_text(layout=True)
                    
                    if text and text.strip():
                        # Extract structured sections from the page
                        sections = self._extract_sections(text, page_num)
                        all_sections.extend(sections)
        
        except Exception as e:
            raise RuntimeError(f"Error processing PDF {local_path}: {e}")

        if not all_sections:
            raise ValueError(f"No text could be extracted from PDF: {local_path}")

        # Convert sections to chunks using the specific strategy
        all_chunks = self.chunk_sections(all_sections, original_source)
        
        logger.info(f"Successfully extracted {len(all_chunks)} chunks from PDF")
        return all_chunks

    def chunk_sections(self, sections: List[DocumentSection], source: str) -> List[Dict[str, str]]:
        """Override this method in subclasses"""
        raise NotImplementedError("Subclasses must implement chunk_sections")

    def _extract_sections(self, text: str, page_num: int) -> List[DocumentSection]:
        """Extract structured sections from page text"""
        sections = []
        lines = text.split('\n')
        
        current_section = []
        current_title = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect headers
            if self._is_header(line):
                # Save previous section if exists
                if current_section:
                    content = '\n'.join(current_section).strip()
                    if content:
                        sections.append(DocumentSection(
                            title=current_title,
                            content=content,
                            page_num=page_num,
                            section_type="paragraph",
                            level=self._get_header_level(current_title)
                        ))
                
                current_title = line
                current_section = []
            else:
                current_section.append(line)
        
        # Don't forget the last section
        if current_section:
            content = '\n'.join(current_section).strip()
            if content:
                sections.append(DocumentSection(
                    title=current_title,
                    content=content,
                    page_num=page_num,
                    section_type="paragraph",
                    level=self._get_header_level(current_title)
                ))
        
        return sections

    def _is_header(self, line: str) -> bool:
        """Detect if a line is likely a header"""
        if not line.strip():
            return False
        
        # Common header patterns in aviation documents
        patterns = [
            r'^[A-Z][A-Z\s]+$',  # All caps
            r'^\d+\.\d+\.?\s+',  # Numbered sections (1.1, 1.2.3, etc.)
            r'^Chapter\s+\d+',   # Chapter headings
            r'^Section\s+\d+',   # Section headings
            r'^Figure\s+\d+',    # Figure captions
            r'^Table\s+\d+',     # Table captions
        ]
        
        for pattern in patterns:
            if re.match(pattern, line):
                return True
        
        # Also check if line is short and looks like a title
        if len(line) < 60 and line[0].isupper() and ':' not in line:
            # Additional checks to avoid false positives
            if not line.endswith('.') and len(word_tokenize(line)) < 10:
                return True
        
        return False

    def _get_header_level(self, header: str) -> int:
        """Determine header level based on pattern"""
        if not header:
            return 0
        
        # Chapter level
        if re.match(r'^Chapter\s+\d+', header, re.IGNORECASE):
            return 1
        
        # Section level (1.x)
        if re.match(r'^\d+\.\s+', header):
            return 2
        
        # Subsection level (1.1.x)
        if re.match(r'^\d+\.\d+\.?\s+', header):
            return 3
        
        # All caps headers
        if header.isupper():
            return 2
        
        return 3  # Default level

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+-\d+', '', text)  # Remove page ranges
        
        return text.strip()

    def _format_section(self, section: DocumentSection) -> str:
        """Format a single section for output"""
        if section.title:
            return f"[{section.title}]\n\n{section.content}"
        return section.content

    def _are_sections_related(self, section1: DocumentSection, section2: DocumentSection) -> bool:
        """Check if two sections are semantically related"""
        if section1.title and section2.title:
            # Check for common keywords
            words1 = set(word_tokenize(section1.title.lower()))
            words2 = set(word_tokenize(section2.title.lower()))
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
            words1 = words1 - stop_words
            words2 = words2 - stop_words
            
            common_words = words1 & words2
            
            # If significant overlap in titles
            if len(common_words) >= 2 or (len(common_words) >= 1 and len(words1) <= 3):
                return True
        
        # Check if sequential sections
        if self._are_sequential_sections(section1.title, section2.title):
            return True
        
        return False

    def _are_sequential_sections(self, title1: str, title2: str) -> bool:
        """Check if sections are sequential (e.g., 1.1 and 1.2)"""
        if not title1 or not title2:
            return False
        
        # Extract section numbers
        pattern = r'^(\d+(?:\.\d+)*)'
        match1 = re.match(pattern, title1)
        match2 = re.match(pattern, title2)
        
        if match1 and match2:
            num1 = match1.group(1)
            num2 = match2.group(1)
            
            # Check if they share the same parent section
            parts1 = num1.split('.')
            parts2 = num2.split('.')
            
            if len(parts1) == len(parts2) and parts1[:-1] == parts2[:-1]:
                # Same level and parent
                try:
                    return int(parts2[-1]) - int(parts1[-1]) == 1
                except ValueError:
                    pass
        
        return False