import os
import re
from typing import Dict, List

from .base_chunk import BaseChunker, DocumentSection


class ParagraphChunker(BaseChunker):
    """Chunks documents by complete paragraphs to maintain context"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_paragraph_length = 50  # Minimum length to consider text as a paragraph
    
    def chunk_sections(self, sections: List[DocumentSection], source: str) -> List[Dict[str, str]]:
        """Chunk by complete paragraphs"""
        chunks = []
        chunk_id = 0
        
        for section in sections:
            if not section.content.strip():
                continue
            
            # Split into paragraphs (double newline or significant indentation)
            paragraphs = self._split_into_paragraphs(section.content)
            
            # If section is very short, keep it as one chunk
            if len(section.content) <= self.min_chunk_size:
                chunk_text = self._format_section(section)
                chunks.append({
                    "text": chunk_text,
                    "page": str(section.page_num),
                    "chunk_id": str(chunk_id),
                    "source": os.path.basename(source),
                    "title": section.title,
                    "chunking_method": "paragraph"
                })
                chunk_id += 1
                continue
            
            # Process longer sections
            current_chunk = []
            current_size = 0
            
            # Add section title to first chunk if exists
            if section.title:
                current_chunk.append(f"[{section.title}]")
                current_size += len(section.title)
            
            for i, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if not paragraph or len(paragraph) < self.min_paragraph_length:
                    continue
                
                paragraph_size = len(paragraph)
                
                # Check if adding this paragraph would exceed chunk size
                if current_size + paragraph_size > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "page": str(section.page_num),
                        "chunk_id": str(chunk_id),
                        "source": os.path.basename(source),
                        "title": section.title,
                        "chunking_method": "paragraph",
                        "paragraph_count": len([p for p in current_chunk if not p.startswith('[')])
                    })
                    chunk_id += 1
                    
                    # Start new chunk with title for context
                    current_chunk = []
                    if section.title:
                        current_chunk.append(f"[{section.title}]")
                        current_size = len(section.title)
                    else:
                        current_size = 0
                    
                    # Add the paragraph that didn't fit
                    current_chunk.append(paragraph)
                    current_size += paragraph_size
                else:
                    current_chunk.append(paragraph)
                    current_size += paragraph_size
            
            # Save remaining chunk
            if current_chunk and len([p for p in current_chunk if not p.startswith('[')]) > 0:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "page": str(section.page_num),
                    "chunk_id": str(chunk_id),
                    "source": os.path.basename(source),
                    "title": section.title,
                    "chunking_method": "paragraph",
                    "paragraph_count": len([p for p in current_chunk if not p.startswith('[')])
                })
                chunk_id += 1
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs using various heuristics"""
        # First try splitting by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # If that doesn't work well, try other patterns
        if len(paragraphs) <= 1:
            # Try splitting by indented lines
            paragraphs = re.split(r'\n\s{4,}', text)
        
        # Further split very long paragraphs by sentence groups
        final_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(para) > self.max_chunk_size:
                # Split long paragraphs at sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_para = []
                current_length = 0
                
                for sentence in sentences:
                    if current_length + len(sentence) > self.chunk_size:
                        if current_para:
                            final_paragraphs.append(' '.join(current_para))
                            current_para = [sentence]
                            current_length = len(sentence)
                        else:
                            final_paragraphs.append(sentence)
                    else:
                        current_para.append(sentence)
                        current_length += len(sentence)
                
                if current_para:
                    final_paragraphs.append(' '.join(current_para))
            else:
                final_paragraphs.append(para)
        
        return final_paragraphs
    
    def get_stats(self, chunks: List[Dict[str, str]]) -> Dict[str, any]:
        """Get statistics about the chunking results"""
        if not chunks:
            return {}
        
        paragraph_counts = [c.get('paragraph_count', 0) for c in chunks]
        chunk_lengths = [len(c['text']) for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunks),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "avg_paragraphs_per_chunk": sum(paragraph_counts) / len(chunks) if paragraph_counts else 0,
            "chunks_with_title": sum(1 for c in chunks if c.get('title'))
        }