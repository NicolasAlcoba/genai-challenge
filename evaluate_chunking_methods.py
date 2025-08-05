import os
import sys
import subprocess
import json
from datetime import datetime

# Chunking strategies to test
CHUNKING_STRATEGIES = [
    {
        "name": "simple",
        "args": ["--chunking", "simple", "--chunk-size", "500", "--chunk-overlap", "50"]
    },
    {
        "name": "sentence",
        "args": ["--chunking", "sentence", "--chunk-size", "800"]
    },
    {
        "name": "structural",
        "args": ["--chunking", "structural", "--chunk-size", "1200"]
    },
    {
        "name": "semantic",
        "args": ["--chunking", "semantic", "--chunk-size", "1200"]
    },
    {
        "name": "hybrid",
        "args": ["--chunking", "hybrid", "--chunk-size", "1200"]
    },
    {
        "name": "paragraph",
        "args": ["--chunking", "paragraph", "--chunk-size", "1200"]
    },
    {
        "name": "sliding_window",
        "args": ["--chunking", "sliding_window", "--window-size", "1200", "--step-size", "600"]
    }
]

def run_command(cmd):
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True

def process_pdf_with_strategy(strategy):
    """Process PDF with a specific chunking strategy"""
    vectorstore_path = f"vector_store_{strategy['name']}"
    
    cmd = [
        sys.executable, "-m", "src.main",
        "process-pdf",
        "-o", vectorstore_path
    ] + strategy["args"]
    
    print(f"\n{'='*60}")
    print(f"Processing PDF with {strategy['name']} chunking strategy...")
    print(f"Output: {vectorstore_path}")
    print(f"{'='*60}")
    
    if run_command(cmd):
        # Read and display chunking statistics
        metadata_path = os.path.join(vectorstore_path, "chunking_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print(f"\nChunking Statistics for {strategy['name']}:")
                print(json.dumps(metadata['stats'], indent=2))
        return vectorstore_path
    return None

def run_evaluation_for_strategy(strategy_name, vectorstore_path):
    """Run evaluation for a specific vector store"""
    print(f"\n{'='*60}")
    print(f"Running evaluation for {strategy_name} chunking...")
    print(f"Using vector store: {vectorstore_path}")
    print(f"{'='*60}")
    
    # Create output directory for this strategy's results
    eval_output_dir = f"evaluation_results_{strategy_name}"
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Run evaluation
    cmd = [
        sys.executable, "-m", "src.main",
        "eval"
    ]
    
    # Temporarily change to use the specific vector store
    # This would need modification in the evaluation code to accept vectorstore path
    # For now, we'll copy the vector store to the default location
    import shutil
    
    # Backup existing vector store if it exists
    if os.path.exists("vector_store"):
        shutil.move("vector_store", "vector_store_backup")
    
    # Copy strategy-specific vector store to default location
    shutil.copytree(vectorstore_path, "vector_store")
    
    # Run evaluation
    run_command(cmd)
    
    # Move results to strategy-specific directory
    for file in os.listdir("evaluation_results"):
        if file.endswith(".json") or file.endswith(".md"):
            src = os.path.join("evaluation_results", file)
            dst = os.path.join(eval_output_dir, file)
            if os.path.exists(src):
                shutil.move(src, dst)
    
    # Restore original vector store
    shutil.rmtree("vector_store")
    if os.path.exists("vector_store_backup"):
        shutil.move("vector_store_backup", "vector_store")
    
    return eval_output_dir

def compare_results(results_dirs):
    """Compare evaluation results across different chunking strategies"""
    print(f"\n{'='*60}")
    print("COMPARISON OF CHUNKING STRATEGIES")
    print(f"{'='*60}\n")
    
    comparison_data = {}
    
    for strategy_name, results_dir in results_dirs.items():
        # Find the most recent evaluation result
        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json') and f.startswith('evaluation_results_')]
        if json_files:
            json_files.sort()
            latest_file = json_files[-1]
            
            with open(os.path.join(results_dir, latest_file), 'r') as f:
                data = json.load(f)
                
                comparison_data[strategy_name] = {
                    "avg_answer_relevance": data['aggregate_metrics']['avg_answer_relevance'],
                    "avg_latency_ms": data['aggregate_metrics']['avg_latency_ms'],
                    "total_samples": data['aggregate_metrics']['total_samples']
                }
    
    # Print comparison table
    print(f"{'Strategy':<20} {'Avg Relevance':<15} {'Avg Latency (ms)':<20} {'Samples':<10}")
    print("-" * 70)
    
    for strategy, metrics in comparison_data.items():
        print(f"{strategy:<20} {metrics['avg_answer_relevance']:<15.3f} {metrics['avg_latency_ms']:<20.1f} {metrics['total_samples']:<10}")
    
    # Find best performing strategy
    best_relevance = max(comparison_data.items(), key=lambda x: x[1]['avg_answer_relevance'])
    best_latency = min(comparison_data.items(), key=lambda x: x[1]['avg_latency_ms'])
    
    print(f"\n{'='*60}")
    print(f"Best Answer Relevance: {best_relevance[0]} ({best_relevance[1]['avg_answer_relevance']:.3f})")
    print(f"Best Latency: {best_latency[0]} ({best_latency[1]['avg_latency_ms']:.1f} ms)")
    print(f"{'='*60}")

def main():
    """Main execution function"""
    print("Evaluating Different Chunking Methods")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results_dirs = {}
    
    # Process PDF with each chunking strategy
    for strategy in CHUNKING_STRATEGIES:
        vectorstore_path = process_pdf_with_strategy(strategy)
        if vectorstore_path:
            # Run evaluation
            eval_dir = run_evaluation_for_strategy(strategy['name'], vectorstore_path)
            results_dirs[strategy['name']] = eval_dir
    
    # Compare results
    if results_dirs:
        compare_results(results_dirs)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 