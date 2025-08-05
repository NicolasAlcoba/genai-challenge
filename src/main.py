import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict

from src.config import PDF_URL

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    if debug:
        level = logging.DEBUG
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Configure root logger for debug mode
        logging.basicConfig(
            level=level, format=format_str, datefmt="%Y-%m-%d %H:%M:%S", force=True
        )

        # Enable full logging in debug mode
        logging.getLogger("src").setLevel(logging.DEBUG)
        logging.getLogger("__main__").setLevel(logging.DEBUG)
        logging.getLogger("mcp").setLevel(logging.INFO)

    else:
        # NORMAL MODE: Only show user-facing messages, suppress all internal logs
        level = logging.CRITICAL  # Only show critical errors
        format_str = "%(message)s"

        # Configure root logger to suppress almost everything
        logging.basicConfig(level=level, format=format_str, force=True)

        # Suppress all internal logging in normal mode
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger("transformers").setLevel(logging.CRITICAL)
        logging.getLogger("sentence_transformers").setLevel(logging.CRITICAL)
        logging.getLogger("faiss").setLevel(logging.CRITICAL)
        logging.getLogger("torch").setLevel(logging.CRITICAL)
        logging.getLogger("mcp").setLevel(logging.CRITICAL)
        logging.getLogger("mcp.server").setLevel(logging.CRITICAL)
        logging.getLogger("mcp.server.lowlevel").setLevel(logging.CRITICAL)
        logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.CRITICAL)
        logging.getLogger("src").setLevel(logging.CRITICAL)
        logging.getLogger("__main__").setLevel(logging.CRITICAL)

        # Disable progress bars in normal mode
        import os

        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Disable tqdm progress bars
        os.environ["DISABLE_TQDM"] = "true"


def run_evaluation(use_hybrid: bool = False, use_optimized: bool = False):
    try:
        from src.evaluation import RAGEvaluator
        from src.agents import RAGAgent
        from src.vector_store import VectorStore

        # Initialize components
        vector_store = VectorStore()
        vector_store.load()
        
        if use_hybrid:
            from src.rag_methods.hybrid_rag import ImprovedVectorStoreRAGPipeline
            logger.info("Using Hybrid RAG Pipeline")
            rag_pipeline = ImprovedVectorStoreRAGPipeline(vector_store)
        elif use_optimized:
            from src.rag_methods.balanced_rag import OptimizedVectorStoreRAGPipeline, BalancedVectorStoreRAGPipeline, SimpleImprovedRAG
            logger.info("Using Optimized RAG Pipeline")
            # rag_pipeline = OptimizedVectorStoreRAGPipeline(vector_store)
            # rag_pipeline = BalancedVectorStoreRAGPipeline(vector_store)
            rag_pipeline = SimpleImprovedRAG(vector_store)
        else:
            from src.rag_methods.rag import VectorStoreRAGPipeline
            logger.info("Using Standard RAG Pipeline")
            rag_pipeline = VectorStoreRAGPipeline(vector_store)
            
        rag_agent = RAGAgent(rag_pipeline)
        evaluator = RAGEvaluator(rag_agent)
    
        # Run evaluation
        logger.info("Starting RAG evaluation...")
        results = evaluator.run_evaluation()

        # Print summary
        print("\n=== Evaluation Complete ===")
        print(f"Evaluated {results['aggregate_metrics']['total_samples']} samples")
        print(f"Average Answer Relevance: {results['aggregate_metrics']['avg_answer_relevance']:.3f}")
        print(f"Average Latency: {results['aggregate_metrics']['avg_latency_ms']:.1f}ms")
        print(f"\nDetailed results saved to: {evaluator.results_dir}")
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        raise e


def process_pdf(pdf_path: str, vectorstore_path: str = "vector_store", chunking_strategy: str = "simple", chunker_kwargs: Optional[Dict] = None):
    try:
        from src.document_processor import DocumentProcessor
        from src.vector_store import VectorStore

        logger.info(f"Processing PDF document using {chunking_strategy} chunking...")
        processor = DocumentProcessor(chunking_strategy=chunking_strategy, chunker_kwargs=chunker_kwargs)
        chunks = processor.process_pdf(pdf_path)

        logger.info(f"Extracted {len(chunks)} chunks from PDF")
        
        # Get and display chunking statistics
        stats = processor.get_chunker_stats(chunks)
        logger.info(f"Chunking statistics: {stats}")

        logger.info("Building vector store...")
        store = VectorStore(store_path=vectorstore_path)
        store.add_documents(chunks)
        store.save()

        logger.info("Vector store created successfully!")
        logger.info(f"Stats: {store.get_stats()}")
        
        # Save metadata about chunking strategy
        import json
        metadata_path = os.path.join(vectorstore_path, "chunking_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "chunking_strategy": chunking_strategy,
                "chunker_kwargs": chunker_kwargs or {},
                "stats": stats
            }, f, indent=2)
        logger.info(f"Saved chunking metadata to {metadata_path}")

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise e


def run_interactive_chat(vectorstore_path: Optional[str] = None, use_hybrid: bool = False, use_optimized: bool = False):

    async def chat_main():
        try:
            from src.mcp_client import RAGMCPChatUI, RAGMCPClient

            client = RAGMCPClient(vectorstore_path=vectorstore_path, use_hybrid=use_hybrid, use_optimized=use_optimized)

            async with client:
                chat_ui = RAGMCPChatUI(client)
                await chat_ui.run_interactive_chat()
        except KeyboardInterrupt:
            logger.debug("Chat interrupted by user")
        except Exception as e:
            logger.error(f"Error in chat application: {e}")
            raise e

    asyncio.run(chat_main())


def main():
    parser = argparse.ArgumentParser(description="GenAI Challenge CLI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat command
    chat_parser = subparsers.add_parser(
        "chat", help="Start interactive chat with AdvancedRAGAgent"
    )
    chat_parser.add_argument(
        "--vectorstore",
        dest="vectorstore_path",
        help="Path to vector store directory",
        default="vector_store",
    )
    chat_parser.add_argument(
        "--use-hybrid",
        action="store_true",
        help="Use hybrid RAG pipeline instead of standard RAG",
    )
    chat_parser.add_argument(
        "--use-optimized",
        action="store_true", 
        help="Use optimized RAG pipeline instead of standard RAG",
    )

    # Evaluation command
    eval_parser = subparsers.add_parser("eval", help="Run the evaluation suite")
    eval_parser.add_argument(
        "--use-hybrid",
        action="store_true",
        help="Use hybrid RAG pipeline instead of standard RAG",
    )
    eval_parser.add_argument(
        "--use-optimized",
        action="store_true",
        help="Use optimized RAG pipeline instead of standard RAG",
    )

    # PDF processing commands
    pdf_parser = subparsers.add_parser(
        "process-pdf", help="Process PDF and create vector store"
    )
    pdf_parser.add_argument(
        "pdf_path",
        nargs="?",
        default=PDF_URL,
        help="Path to the PDF file to process (default: value from config)",
    )
    pdf_parser.add_argument(
        "-o",
        "--output",
        dest="vectorstore_path",
        default="vector_store",
        help="Output directory for the vector store (default: vector_store)",
    )
    pdf_parser.add_argument(
        "--chunking",
        dest="chunking_strategy",
        default="simple",
        choices=["simple", "sentence", "semantic", "structural", "hybrid", "paragraph", "sliding_window"],
        help="Chunking strategy to use (default: simple)",
    )
    pdf_parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in characters (default: 500)",
    )
    pdf_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks for simple chunking (default: 50)",
    )
    pdf_parser.add_argument(
        "--window-size",
        type=int,
        help="Window size for sliding window chunking",
    )
    pdf_parser.add_argument(
        "--step-size",
        type=int,
        help="Step size for sliding window chunking",
    )

    args = parser.parse_args()

    # Setup logging based on debug flag
    setup_logging(debug=args.debug)

    if args.debug:
        logger.info("Debug mode enabled")

    if args.command == "chat":
        run_interactive_chat(
            vectorstore_path=getattr(args, "vectorstore_path", None),
            use_hybrid=getattr(args, "use_hybrid", False),
            use_optimized=getattr(args, "use_optimized", False)
        )
    elif args.command == "eval":
        run_evaluation(use_hybrid=getattr(args, "use_hybrid", False), use_optimized=getattr(args, "use_optimized", False))
    elif args.command == "process-pdf":
        # Prepare chunker kwargs based on command line arguments
        chunker_kwargs = {
            "chunk_size": getattr(args, "chunk_size", 500),
        }
        
        # Add strategy-specific parameters
        if args.chunking_strategy == "simple":
            chunker_kwargs["overlap"] = getattr(args, "chunk_overlap", 50)
        elif args.chunking_strategy == "sliding_window":
            if hasattr(args, "window_size") and args.window_size:
                chunker_kwargs["window_size"] = args.window_size
            if hasattr(args, "step_size") and args.step_size:
                chunker_kwargs["step_size"] = args.step_size
        
        process_pdf(
            pdf_path=getattr(args, "pdf_path"),
            vectorstore_path=getattr(args, "vectorstore_path"),
            chunking_strategy=getattr(args, "chunking_strategy", "simple"),
            chunker_kwargs=chunker_kwargs
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
