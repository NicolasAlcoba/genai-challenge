import logging
import re
from typing import List, Tuple, Dict, Optional
import numpy as np

from transformers.pipelines import pipeline as hf_pipeline
import torch

from src.config import EARLY_STOPPING, GEN_MODEL, MAX_NEW_TOKENS, NUM_BEAMS, TOP_K
from src.rag import VectorStoreRAGPipeline

logger = logging.getLogger(__name__)


class BalancedVectorStoreRAGPipeline:
    """Balanced RAG pipeline that combines speed with quality"""

    def __init__(self, vector_store, top_k: int = TOP_K):
        self.vector_store = vector_store
        self.top_k = top_k
        
        # Initialize generation pipeline - keep original settings for compatibility
        self._gen = hf_pipeline(
            "text-generation",
            model=GEN_MODEL,
            tokenizer=GEN_MODEL,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            early_stopping=EARLY_STOPPING,
            # Use original temperature settings for consistency
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self._tokenizer = self._gen.tokenizer

    def retrieve(self, question: str) -> List[str]:
        """Simple, fast retrieval without complex reranking"""
        if not question.strip():
            logger.debug("Empty question provided, returning fallback")
            return ["I do not know"]

        logger.debug(f"Searching vector store for: {question}")
        
        # Use the original search method
        texts, scores, metadata = self.vector_store.search(question, k=self.top_k)

        if texts:
            logger.debug(f"Found {len(texts)} relevant documents")
            logger.debug(f"Best score: {scores[0]:.3f}")
            
            # Simple filtering - only remove very low confidence results
            filtered_texts = []
            for text, score in zip(texts, scores):
                if score > 0.1:  # Very low threshold to keep most results
                    filtered_texts.append(text)
            
            if filtered_texts:
                return filtered_texts
            
        logger.debug("No relevant documents found")
        return ["I do not know"]

    def _build_prompt(self, content: str) -> str:
        """Build prompt similar to original but slightly optimized"""
        if not self._tokenizer:
            raise ValueError("LLM not initialized. Cannot build prompt.")

        # Keep system prompt similar to original for consistency
        system_prompt = (
            "You are an expert assistant. Answer the user's question based on the provided context. "
            "If the context doesn't contain enough information to answer the question, "
            "say 'I don't have enough information to answer this question.'"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not isinstance(prompt, str):
            prompt = str(prompt)
        return prompt

    def _generate_answer(self, prompt: str) -> str:
        """Generate answer similar to original approach"""
        if not self._gen:
            raise ValueError("LLM not initialized. Cannot generate answer.")

        try:
            out = self._gen(prompt)
            if isinstance(out, list) and out and "generated_text" in out[0]:
                return out[0]["generated_text"].strip()
            raise ValueError("Unexpected output format from the generation pipeline.")
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_answer(self, question: str, snippets: List[str]) -> str:
        """Generate answer with balanced approach"""
        logger.debug(f"Generating answer for question: {question}")
        
        # Handle no context case
        if snippets and snippets[0] != "I do not know":
            # Use more snippets for better context (like original)
            # But limit total context to avoid overwhelming small models
            context_snippets = snippets[:5]  # Use up to 5 snippets
            context = "\n\n".join(context_snippets)
            content = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            # If no context, still try to generate (like original behavior)
            content = f"Question: {question}\n\nAnswer:"

        logger.debug("Generating LLM response")
        prompt = self._build_prompt(content)
        answer = self._generate_answer(prompt)

        # Clean up answer
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        logger.debug(f"Generated answer length: {len(answer)}")
        return answer

    def run(self, question: str) -> Tuple[List[str], List[int]]:
        """Run the RAG pipeline"""
        snippets = self.retrieve(question)
        sources = list(range(len(snippets))) if snippets[0] != "I do not know" else []
        answer = self.generate_answer(question, snippets)
        return [answer], sources


class OptimizedVectorStoreRAGPipeline(BalancedVectorStoreRAGPipeline):
    """Optimized version with smart caching and better retrieval"""
    
    def __init__(self, vector_store, top_k: int = TOP_K, 
                 score_threshold: float = 0.2,
                 use_reranking: bool = False):
        super().__init__(vector_store, top_k)
        self.score_threshold = score_threshold
        self.use_reranking = use_reranking
        self.query_cache = {}
    
    def retrieve(self, question: str) -> List[str]:
        """Enhanced retrieval with optional reranking"""
        if not question.strip():
            return ["I do not know"]
        
        # Check cache
        cache_key = question.lower().strip()
        if cache_key in self.query_cache:
            logger.debug("Using cached results")
            return self.query_cache[cache_key]
        
        # Get more results initially
        search_k = min(self.top_k * 2, 10)  # Get double but cap at 10
        texts, scores, metadata = self.vector_store.search(question, k=search_k)
        
        if not texts:
            return ["I do not know"]
        
        # Filter by score threshold
        filtered = [(t, s) for t, s in zip(texts, scores) if s >= self.score_threshold]
        
        if not filtered:
            # If nothing passes threshold, take best result anyway
            filtered = [(texts[0], scores[0])]
        
        # Optional simple reranking
        if self.use_reranking and len(filtered) > 1:
            filtered = self._simple_rerank(question, filtered)
        
        # Take top k results
        result_texts = [t for t, s in filtered[:self.top_k]]
        
        # Cache results
        self.query_cache[cache_key] = result_texts
        
        return result_texts
    
    def _simple_rerank(self, question: str, 
                      results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Simple keyword-based reranking without heavy computation"""
        # Extract key terms from question
        question_terms = set(re.findall(r'\b\w+\b', question.lower()))
        
        # Score each result
        reranked = []
        for text, score in results:
            text_terms = set(re.findall(r'\b\w+\b', text.lower()))
            
            # Simple overlap score
            overlap = len(question_terms & text_terms) / max(len(question_terms), 1)
            
            # Combine with original score (70% semantic, 30% keyword)
            combined_score = 0.7 * score + 0.3 * overlap
            reranked.append((text, combined_score))
        
        # Sort by combined score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked


class SimpleImprovedRAG(VectorStoreRAGPipeline):
    def __init__(self, vector_store, top_k: int = 5):
        super().__init__(vector_store, top_k)
        # Solo agrega GPU support
        if torch.cuda.is_available():
            self._gen.model = self._gen.model.to('cuda')
            self._gen.device = 0
    
    def retrieve(self, question: str) -> List[str]:
        # Usa el método original pero con logging
        results = super().retrieve(question)
        logger.debug(f"Retrieved {len(results)} documents for: {question}")
        return results


# Configuration recommendations for evaluation
def get_recommended_config():
    """Get recommended configuration for different use cases"""
    return {
        "balanced": {
            "class": BalancedVectorStoreRAGPipeline,
            "params": {
                "top_k": 5  # Use 5 snippets like original
            }
        },
        "optimized": {
            "class": OptimizedVectorStoreRAGPipeline,
            "params": {
                "top_k": 5,
                "score_threshold": 0.15,  # Low threshold to avoid filtering good results
                "use_reranking": True
            }
        },
        "fast": {
            "class": BalancedVectorStoreRAGPipeline,
            "params": {
                "top_k": 3  # Fewer snippets for speed
            }
        }
    }