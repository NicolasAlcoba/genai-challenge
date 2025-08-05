import logging
import re
from typing import List, Tuple
import numpy as np

from transformers.pipelines import pipeline as hf_pipeline
import torch

from src.config import EARLY_STOPPING, GEN_MODEL, MAX_NEW_TOKENS, NUM_BEAMS, TOP_K

logger = logging.getLogger(__name__)


class DebugVectorStoreRAGPipeline:
    """Debug version to diagnose vector store issues"""

    def __init__(self, vector_store, top_k: int = TOP_K):
        self.vector_store = vector_store
        self.top_k = top_k
        
        # Check vector store status
        logger.info(f"Initializing RAG with vector_store: {type(vector_store)}")
        
        # Try to get info about the vector store
        if hasattr(vector_store, 'index'):
            logger.info(f"Vector store index size: {vector_store.index.ntotal if hasattr(vector_store.index, 'ntotal') else 'unknown'}")
        
        # Initialize generation pipeline
        self._gen = hf_pipeline(
            "text-generation",
            model=GEN_MODEL,
            tokenizer=GEN_MODEL,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            early_stopping=EARLY_STOPPING,
            temperature=0.7,
            repetition_penalty=1.1,
            do_sample=True,
            top_p=0.9,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self._tokenizer = self._gen.tokenizer

    def retrieve(self, question: str) -> Tuple[List[str], List[float], List[int]]:
        """Debug version of retrieve with detailed logging"""
        if not question.strip():
            logger.debug("Empty question provided")
            return ["I do not know"], [0.0], [-1]

        logger.info(f"=== DEBUG RETRIEVE ===")
        logger.info(f"Question: {question}")
        
        # Check if vector store has documents
        if hasattr(self.vector_store, 'texts'):
            logger.info(f"Vector store has {len(self.vector_store.texts)} texts")
            if len(self.vector_store.texts) > 0:
                logger.info(f"Sample text: {self.vector_store.texts[0][:100]}...")
        
        # Try search
        try:
            result = self.vector_store.search(question, k=self.top_k)
            logger.info(f"Search returned: {type(result)}, length: {len(result) if result else 'None'}")
            
            if result and len(result) >= 3:
                texts, scores, indices = result
                logger.info(f"Found {len(texts)} texts")
                logger.info(f"Scores: {scores[:5] if scores else 'None'}")  # Show first 5 scores
                logger.info(f"Indices: {indices[:5] if indices else 'None'}")
                
                if texts and len(texts) > 0:
                    logger.info(f"First text preview: {texts[0][:200]}...")
                    return texts, scores, indices
                else:
                    logger.warning("Search returned empty texts")
            else:
                logger.warning(f"Unexpected search result format: {result}")
                
        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
        
        logger.info("No relevant documents found - returning fallback")
        return ["I do not know"], [0.0], [-1]

    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for generation"""
        # Simple prompt for debugging
        system_prompt = "Answer based on the context. If unsure, say 'I don't have enough information'."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        return prompt if isinstance(prompt, str) else str(prompt)

    def generate_answer(self, question: str, snippets: List[str], scores: List[float]) -> str:
        """Generate answer with fallback to direct generation if no context"""
        logger.info(f"=== GENERATE ANSWER ===")
        logger.info(f"Snippets count: {len(snippets)}")
        logger.info(f"First snippet: {snippets[0][:100] if snippets else 'None'}...")
        
        # If no good context found, try to generate anyway (like old RAG)
        if not snippets or snippets[0] == "I do not know":
            logger.info("No context found - attempting direct generation")
            
            # Direct generation without context (similar to old behavior)
            prompt = self._build_prompt(question, "No specific context available.")
            
            try:
                out = self._gen(prompt)
                if isinstance(out, list) and out and "generated_text" in out[0]:
                    answer = out[0]["generated_text"]
                    # Clean the prompt from answer
                    if prompt in answer:
                        answer = answer.replace(prompt, "").strip()
                    return answer
            except Exception as e:
                logger.error(f"Generation error: {str(e)}")
                
            return "I don't have enough information to answer this question."
        
        # Normal generation with context
        context = "\n\n".join(snippets[:3])
        prompt = self._build_prompt(question, context)
        
        try:
            out = self._gen(prompt)
            if isinstance(out, list) and out and "generated_text" in out[0]:
                answer = out[0]["generated_text"]
                if prompt in answer:
                    answer = answer.replace(prompt, "").strip()
                return answer
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            
        return "Error generating response."

    def run(self, question: str) -> Tuple[List[str], List[int]]:
        """Run the complete RAG pipeline with debugging"""
        logger.info(f"\n{'='*50}")
        logger.info(f"RUNNING RAG PIPELINE FOR: {question}")
        logger.info(f"{'='*50}")
        
        texts, scores, indices = self.retrieve(question)
        answer = self.generate_answer(question, texts, scores)
        
        return [answer], indices[:3] if indices[0] != -1 else []


# Quick test function
def test_vector_store(vector_store):
    """Test function to check vector store status"""
    print("\n=== VECTOR STORE DIAGNOSTIC ===")
    
    # Check basic attributes
    attrs_to_check = ['texts', 'embeddings', 'index', 'encoder']
    for attr in attrs_to_check:
        if hasattr(vector_store, attr):
            val = getattr(vector_store, attr)
            if attr == 'texts' and isinstance(val, list):
                print(f"{attr}: List with {len(val)} items")
                if val:
                    print(f"  First item: {str(val[0])[:100]}...")
            elif attr == 'embeddings' and hasattr(val, 'shape'):
                print(f"{attr}: Shape {val.shape}")
            elif attr == 'index':
                if hasattr(val, 'ntotal'):
                    print(f"{attr}: Contains {val.ntotal} vectors")
                else:
                    print(f"{attr}: {type(val)}")
            else:
                print(f"{attr}: {type(val)}")
        else:
            print(f"{attr}: NOT FOUND")
    
    # Try a test search
    print("\n=== TEST SEARCH ===")
    try:
        test_query = "test query"
        result = vector_store.search(test_query, k=5)
        print(f"Search result type: {type(result)}")
        print(f"Search result: {result}")
    except Exception as e:
        print(f"Search failed: {str(e)}")
    
    print("=== END DIAGNOSTIC ===\n")