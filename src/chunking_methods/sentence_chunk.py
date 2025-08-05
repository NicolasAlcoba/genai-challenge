import os
from typing import Dict, List
from nltk.tokenize import sent_tokenize

from .base_chunk import BaseChunker, DocumentSection


class SentenceChunker(BaseChunker):
    """Chunks documents by complete sentences with smart overlap"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sentence_overlap = 2  # Number of sentences to overlap between chunks
    
    def chunk_sections(self, sections: List[DocumentSection], source: str) -> List[Dict[str, str]]:
        """Chunk by complete sentences"""
        chunks = []
        chunk_id = 0
        
        for section in sections:
            # Skip empty sections
            if not section.content.strip():
                continue
            
            # Split into sentences
            sentences = sent_tokenize(section.content)
            
            # If section is very short, keep it as one chunk
            if len(section.content) <= self.min_chunk_size:
                chunk_text = self._format_section(section)
                chunks.append({
                    "text": chunk_text,
                    "page": str(section.page_num),
                    "chunk_id": str(chunk_id),
                    "source": os.path.basename(source),
                    "title": section.title,
                    "chunking_method": "sentence"
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
            
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_size = len(sentence)
                
                # Check if adding this sentence would exceed chunk size
                if current_size + sentence_size > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "page": str(section.page_num),
                        "chunk_id": str(chunk_id),
                        "source": os.path.basename(source),
                        "title": section.title,
                        "chunking_method": "sentence",
                        "sentence_count": len([s for s in current_chunk if not s.startswith('[')])
                    })
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    overlap_sentences = []
                    
                    # Always include title in new chunks for context
                    if section.title:
                        overlap_sentences.append(f"[{section.title}]")
                    
                    # Add overlapping sentences from previous chunk
                    prev_sentences = [s for s in current_chunk if not s.startswith('[')]
                    if len(prev_sentences) >= self.sentence_overlap:
                        overlap_sentences.extend(prev_sentences[-self.sentence_overlap:])
                    else:
                        overlap_sentences.extend(prev_sentences)
                    
                    overlap_sentences.append(sentence)
                    current_chunk = overlap_sentences
                    current_size = sum(len(s) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size
            
            # Save remaining chunk
            if current_chunk and len([s for s in current_chunk if not s.startswith('[')]) > 0:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "page": str(section.page_num),
                    "chunk_id": str(chunk_id),
                    "source": os.path.basename(source),
                    "title": section.title,
                    "chunking_method": "sentence",
                    "sentence_count": len([s for s in current_chunk if not s.startswith('[')])
                })
                chunk_id += 1
        
        return chunks
    
    def get_stats(self, chunks: List[Dict[str, str]]) -> Dict[str, any]:
        """Get statistics about the chunking results"""
        if not chunks:
            return {}
        
        sentence_counts = [c.get('sentence_count', 0) for c in chunks]
        chunk_lengths = [len(c['text']) for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunks),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "avg_sentences_per_chunk": sum(sentence_counts) / len(chunks) if sentence_counts else 0,
            "chunks_with_title": sum(1 for c in chunks if c.get('title'))
        }