import logging
import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import numpy as np
from collections import Counter

from transformers.pipelines import pipeline as hf_pipeline
from transformers import AutoTokenizer
import torch

from src.config import EARLY_STOPPING, GEN_MODEL, MAX_NEW_TOKENS, NUM_BEAMS, TOP_K

logger = logging.getLogger(__name__)


class BaseRAGPipeline(ABC):
    """Abstract base class for RAG pipelines."""

    def __init__(self, docs: List[str], top_k: int = TOP_K):
        self.docs = docs
        self.top_k = top_k

    @abstractmethod
    def retrieve(self, question: str) -> List[str]:
        """Retrieve relevant context for the question."""
        pass

    @abstractmethod
    def run(self, question: str) -> Tuple[List[str], List[int]]:
        """Run the retrieval, rank and return (snippets, source_indices)."""
        pass


class ImprovedVectorStoreRAGPipeline(BaseRAGPipeline):
    """Optimized RAG pipeline for small models with enhanced retrieval and generation"""

    def __init__(self, vector_store, top_k: int = TOP_K, 
                 rerank_top_k: int = None,
                 max_context_length: int = 1024,
                 temperature: float = 0.7,
                 repetition_penalty: float = 1.1):
        super().__init__(docs=[], top_k=top_k)
        self.vector_store = vector_store
        self.rerank_top_k = rerank_top_k or max(3, top_k // 2)
        self.max_context_length = max_context_length
        
        # Initialize generation pipeline with optimizations for small models
        self._gen = hf_pipeline(
            "text-generation",
            model=GEN_MODEL,
            tokenizer=GEN_MODEL,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            early_stopping=EARLY_STOPPING,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            top_p=0.9,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self._tokenizer = self._gen.tokenizer
        
        # Cache for embeddings to avoid recomputation
        self.query_cache = {}

    def retrieve(self, question: str) -> Tuple[List[str], List[float], List[int]]:
        """Enhanced retrieval with relevance scores and source tracking"""
        if not question.strip():
            logger.debug("Empty question provided, returning fallback")
            return ["I do not know"], [0.0], [-1]

        # Check cache first
        cache_key = question.lower().strip()
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        logger.debug(f"Searching vector store for: {question}")
        # Your VectorStore returns (texts, scores, metadata), not indices
        texts, scores, metadata = self.vector_store.search(question, k=self.top_k)

        if texts:
            logger.debug(f"Found {len(texts)} relevant documents")
            # Generate indices from metadata or use sequential indices
            indices = list(range(len(texts)))
            # Cache the results
            self.query_cache[cache_key] = (texts, scores, indices)
            return texts, scores, indices

        logger.debug("No relevant documents found")
        return ["I do not know"], [0.0], [-1]

    def _rerank_snippets(self, question: str, snippets: List[str], 
                        scores: List[float]) -> Tuple[List[str], List[float]]:
        """Rerank snippets based on keyword overlap and semantic scores"""
        if len(snippets) <= self.rerank_top_k:
            return snippets, scores
        
        # Extract keywords from question
        question_tokens = set(re.findall(r'\b\w+\b', question.lower()))
        
        # Calculate combined scores
        combined_scores = []
        for snippet, score in zip(snippets, scores):
            snippet_tokens = set(re.findall(r'\b\w+\b', snippet.lower()))
            
            # Keyword overlap score
            overlap = len(question_tokens & snippet_tokens) / max(len(question_tokens), 1)
            
            # Length penalty (prefer concise snippets for small models)
            length_penalty = 1.0 / (1.0 + len(snippet.split()) / 100)
            
            # Combined score: 60% semantic, 30% keyword overlap, 10% length
            combined = 0.6 * score + 0.3 * overlap + 0.1 * length_penalty
            combined_scores.append(combined)
        
        # Sort by combined score and take top k
        ranked_indices = np.argsort(combined_scores)[::-1][:self.rerank_top_k]
        
        return ([snippets[i] for i in ranked_indices], 
                [combined_scores[i] for i in ranked_indices])

    def _chunk_text(self, text: str, max_tokens: int) -> List[str]:
        """Smart chunking that preserves sentence boundaries"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = len(self._tokenizer.encode(sentence, add_special_tokens=False))
            
            if current_length + sentence_tokens > max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _build_optimized_prompt(self, question: str, context: str) -> str:
        """Build an optimized prompt for small models"""
        if not self._tokenizer:
            raise ValueError("LLM not initialized. Cannot build prompt.")

        # Estimate token usage
        system_tokens = 50  # Approximate
        question_tokens = len(self._tokenizer.encode(question, add_special_tokens=False))
        available_tokens = self.max_context_length - system_tokens - question_tokens - 50  # Buffer
        
        # Chunk context if needed
        context_chunks = self._chunk_text(context, available_tokens)
        
        if context_chunks:
            context = context_chunks[0]  # Use the most relevant chunk
        
        # Concise system prompt optimized for small models
        system_prompt = (
            "Answer based on the context. Be concise and direct. "
            "If unsure, say 'I don't have enough information'."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        return prompt if isinstance(prompt, str) else str(prompt)

    def _extract_answer_patterns(self, text: str) -> str:
        """Extract and clean the answer from generated text"""
        # Remove common artifacts
        patterns_to_remove = [
            r'Based on the context[,:]?\s*',
            r'According to the information[,:]?\s*',
            r'The context (?:states|mentions|indicates)[,:]?\s*',
            r'Answer:\s*',
        ]
        
        cleaned = text
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove redundant line breaks
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()

    def _generate_answer(self, prompt: str) -> str:
        """Generate answer with error handling and post-processing"""
        if not self._gen:
            raise ValueError("LLM not initialized. Cannot generate answer.")

        try:
            # Generate with controlled randomness
            out = self._gen(
                prompt,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
            
            if isinstance(out, list) and out and "generated_text" in out[0]:
                raw_answer = out[0]["generated_text"]
                
                # Extract only the new generated content
                if prompt in raw_answer:
                    raw_answer = raw_answer.replace(prompt, "").strip()
                
                # Clean and format the answer
                answer = self._extract_answer_patterns(raw_answer)
                
                return answer
            
            raise ValueError("Unexpected output format from the generation pipeline.")
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return "I apologize, but I encountered an error generating the response."

    def generate_answer(self, question: str, snippets: List[str], 
                       scores: List[float]) -> str:
        """Generate answer with context optimization for small models"""
        logger.debug(f"Generating answer for question: {question}")
        
        # Filter out low-confidence snippets
        confidence_threshold = 0.3
        filtered_snippets = [
            s for s, score in zip(snippets, scores) 
            if score > confidence_threshold
        ]
        
        if not filtered_snippets or filtered_snippets[0] == "I do not know":
            return "I don't have enough information to answer this question."
        
        # Rerank snippets for better relevance
        reranked_snippets, _ = self._rerank_snippets(
            question, filtered_snippets[:self.top_k], scores[:len(filtered_snippets)]
        )
        
        # Build context with snippet separation
        context_parts = []
        for i, snippet in enumerate(reranked_snippets[:3], 1):  # Limit to top 3 for small models
            context_parts.append(f"[{i}] {snippet}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = self._build_optimized_prompt(question, context)
        answer = self._generate_answer(prompt)
        
        # Validate answer quality
        if len(answer.split()) < 5 and "don't" not in answer.lower():
            # Too short, might be truncated
            logger.warning("Generated answer seems too short, using fallback")
            return f"Based on the available information: {reranked_snippets[0][:200]}..."
        
        return answer

    def run(self, question: str) -> Tuple[List[str], List[int]]:
        """Run the complete RAG pipeline"""
        texts, scores, indices = self.retrieve(question)
        
        if texts[0] == "I do not know":
            return ["I don't have enough information to answer this question."], []
        
        answer = self.generate_answer(question, texts, scores)
        
        # Return answer and source indices
        return [answer], indices[:self.rerank_top_k]

    def clear_cache(self):
        """Clear the query cache to free memory"""
        self.query_cache.clear()
        logger.debug("Query cache cleared")

class HybridRAGPipeline(ImprovedVectorStoreRAGPipeline):
    """Hybrid search combining semantic and keyword matching"""
    
    def __init__(self, vector_store, keyword_index=None, **kwargs):
        super().__init__(vector_store, **kwargs)
        self.keyword_index = keyword_index  # BM25 or similar
        
    def retrieve(self, question: str) -> Tuple[List[str], List[float], List[int]]:
        """Combine vector and keyword search results"""
        # Vector search
        vec_texts, vec_scores, vec_indices = super().retrieve(question)
        
        if self.keyword_index is None:
            return vec_texts, vec_scores, vec_indices
        
        # Keyword search (assuming it returns same format)
        kw_texts, kw_scores, kw_metadata = self.keyword_index.search(question, k=self.top_k)
        
        # Merge results with weighted scores
        all_results = {}
        
        # Add vector results (weight: 0.7)
        for i, (text, score) in enumerate(zip(vec_texts, vec_scores)):
            all_results[i] = {
                'text': text,
                'score': 0.7 * score,
                'source': 'vector'
            }
        
        # Add keyword results (weight: 0.3)
        start_idx = len(vec_texts)
        for i, (text, score) in enumerate(zip(kw_texts, kw_scores)):
            idx = start_idx + i
            if text not in vec_texts:  # Avoid duplicates
                all_results[idx] = {
                    'text': text,
                    'score': 0.3 * score,
                    'source': 'keyword'
                }
        
        # Sort by combined score
        sorted_results = sorted(
            all_results.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )[:self.top_k]
        
        texts = [r[1]['text'] for r in sorted_results]
        scores = [r[1]['score'] for r in sorted_results]
        indices = list(range(len(texts)))
        
        return texts, scores, indices