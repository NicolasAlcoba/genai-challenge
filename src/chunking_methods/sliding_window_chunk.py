import os
from typing import Dict, List
from nltk.tokenize import word_tokenize, sent_tokenize

from .base_chunk import BaseChunker, DocumentSection


class SlidingWindowChunker(BaseChunker):
    """Chunks documents using sliding windows with configurable overlap"""
    
    def __init__(self, window_size: int = 1000, step_size: int = 500, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.step_size = step_size  # How much to move the window each time
        self.overlap_ratio = 1 - (step_size / window_size)  # Calculate overlap ratio
    
    def chunk_sections(self, sections: List[DocumentSection], source: str) -> List[Dict[str, str]]:
        """Chunk using sliding window approach"""
        chunks = []
        chunk_id = 0
        
        # Combine all sections into a single text with section markers
        full_text = []
        section_boundaries = []  # Track where each section starts/ends
        
        for section in sections:
            if not section.content.strip():
                continue
                
            start_pos = len(' '.join(full_text))
            
            # Add section with title marker
            if section.title:
                full_text.append(f"\n[SECTION: {section.title}]\n")
            
            full_text.append(section.content)
            
            end_pos = len(' '.join(full_text))
            section_boundaries.append({
                'start': start_pos,
                'end': end_pos,
                'section': section
            })
        
        # Join all text
        combined_text = ' '.join(full_text)
        
        # Tokenize for better boundary detection
        tokens = word_tokenize(combined_text)
        
        # Create chunks using sliding window
        start_idx = 0
        
        while start_idx < len(tokens):
            # Calculate end of window
            end_idx = min(start_idx + self._calculate_window_size_in_tokens(tokens, start_idx), len(tokens))
            
            # Adjust window to end at sentence boundary if possible
            window_tokens = tokens[start_idx:end_idx]
            window_text = ' '.join(window_tokens)
            
            # Try to find a good breaking point
            sentences = sent_tokenize(window_text)
            if len(sentences) > 1 and end_idx < len(tokens):
                # Remove the last incomplete sentence
                complete_sentences = sentences[:-1]
                window_text = ' '.join(complete_sentences)
                
                # Adjust end_idx to match the actual end
                actual_token_count = len(word_tokenize(window_text))
                end_idx = start_idx + actual_token_count
            
            # Find which section(s) this chunk belongs to
            chunk_start_char = len(' '.join(tokens[:start_idx]))
            chunk_end_char = len(' '.join(tokens[:end_idx]))
            
            relevant_sections = []
            for boundary in section_boundaries:
                if (boundary['start'] <= chunk_end_char and 
                    boundary['end'] >= chunk_start_char):
                    relevant_sections.append(boundary['section'])
            
            # Clean up the window text
            window_text = self._clean_window_text(window_text)
            
            if window_text.strip():
                # Determine primary page number (from first relevant section)
                page_num = relevant_sections[0].page_num if relevant_sections else 1
                
                # Collect all relevant titles
                titles = list(set(s.title for s in relevant_sections if s.title))
                
                chunks.append({
                    "text": window_text,
                    "page": str(page_num),
                    "chunk_id": str(chunk_id),
                    "source": os.path.basename(source),
                    "title": " | ".join(titles) if titles else "",
                    "chunking_method": "sliding_window",
                    "window_size": self.window_size,
                    "overlap_ratio": self.overlap_ratio,
                    "token_count": end_idx - start_idx
                })
                chunk_id += 1
            
            # Move the window
            start_idx += self._calculate_step_size_in_tokens(tokens, start_idx)
            
            # Ensure we don't create tiny final chunks
            if len(tokens) - start_idx < self.min_chunk_size // 4:  # Rough approximation
                break
        
        return chunks
    
    def _calculate_window_size_in_tokens(self, tokens: List[str], start_idx: int) -> int:
        """Calculate window size in tokens based on character limit"""
        char_count = 0
        token_count = 0
        
        for i in range(start_idx, len(tokens)):
            char_count += len(tokens[i]) + 1  # +1 for space
            token_count += 1
            
            if char_count >= self.window_size:
                break
        
        return token_count
    
    def _calculate_step_size_in_tokens(self, tokens: List[str], start_idx: int) -> int:
        """Calculate step size in tokens based on character limit"""
        char_count = 0
        token_count = 0
        
        for i in range(start_idx, len(tokens)):
            char_count += len(tokens[i]) + 1  # +1 for space
            token_count += 1
            
            if char_count >= self.step_size:
                break
        
        return max(1, token_count)  # Ensure we always move at least 1 token
    
    def _clean_window_text(self, text: str) -> str:
        """Clean up window text removing section markers and extra whitespace"""
        import re
        
        # Remove section markers but keep the title information
        text = re.sub(r'\[SECTION: ([^\]]+)\]', r'[\1]', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def get_stats(self, chunks: List[Dict[str, str]]) -> Dict[str, any]:
        """Get statistics about the chunking results"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(c['text']) for c in chunks]
        token_counts = [c.get('token_count', 0) for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunks),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "avg_tokens_per_chunk": sum(token_counts) / len(chunks) if token_counts else 0,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "overlap_ratio": self.overlap_ratio,
            "chunks_with_title": sum(1 for c in chunks if c.get('title'))
        }