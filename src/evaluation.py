import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import numpy as np
from pathlib import Path
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from src.agents import RAGAgent
from src.vector_store import VectorStore
from src.rag import BaseRAGPipeline
from src.rag import VectorStoreRAGPipeline

logger = logging.getLogger(__name__)

@dataclass
class EvaluationSample:
    """Represents a single evaluation sample"""
    question: str
    expected_answer: str
    category: str
    difficulty: str
    
    
@dataclass
class EvaluationResult:
    """Stores evaluation results for a single sample"""
    question: str
    generated_answer: str
    expected_answer: str
    answer_relevance_score: float
    latency_ms: float
    timestamp: str

class RAGEvaluator:

    def __init__(self, rag_agent: RAGAgent, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.rag_agent = rag_agent
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)

    def load_evaluation_dataset(self, dataset_path: str = None) -> List[EvaluationSample]:
        """Load or create evaluation dataset"""
        if dataset_path and Path(dataset_path).exists():
            with open(dataset_path, 'r') as f:
                data = json.load(f)
                return [EvaluationSample(**sample) for sample in data['samples']]
        
        # Default evaluation dataset for aviation domain
        return [
            EvaluationSample(
                question="What is a lazy 8 maneuver?",
                expected_answer="A lazy 8 is a flight training maneuver designed to develop proper coordination of flight controls across a wide range of airspeeds and attitudes. It involves a figure-8 pattern in the sky with constantly changing control pressures.",
                category="flight_maneuvers",
                difficulty="basic"
            ),
            EvaluationSample(
                question="What is taxiing in aviation?",
                expected_answer="Taxiing is the movement of an aircraft on the ground under its own power, excluding takeoff and landing. It includes moving the aircraft from parking areas to runways, between runways, and to/from gates or tie-down areas. Proper taxiing involves maintaining directional control using rudder pedals, appropriate speed control, and following ground control instructions and airport signage.",
                category="ground_operations",
                difficulty="basic"
            ),
            EvaluationSample(
                question="What are the common errors in the performance of intentional stalls?",
                expected_answer="Common errors in intentional stalls include: failure to adequately clear the area, improper pitch control during stall entry, failure to recognize the first indications of a stall, failure to achieve a full stall, improper rudder control causing uncoordinated flight, failure to maintain proper recovery technique (not reducing angle of attack first), excessive forward elevator pressure during recovery causing negative G-forces, secondary stall during recovery, excessive altitude loss, and failure to return to the specified heading.",
                category="flight_maneuvers",
                difficulty="intermediate"
            ),
            EvaluationSample(
                question="What are the cloud clearance requirements for VFR flight in Class E airspace below 10,000 feet MSL?",
                expected_answer="In Class E airspace below 10,000 feet MSL, VFR cloud clearance requirements are: 500 feet below clouds, 1,000 feet above clouds, and 2,000 feet horizontal distance from clouds. Flight visibility must be at least 3 statute miles.",
                category="regulations",
                difficulty="intermediate"
            ),
            EvaluationSample(
                question="What are the required documents for an aircraft to be legally airworthy?",
                expected_answer="Required Certificates and Documents: Certificate of Airworthiness, Certificate of Registration, Aircraft Logbooks (airframe, engine, propeller), Technical Documentation (Weight and Balance Data, Equipment List, Flight Manual/Pilot's Operating Handbook, Required Placards and Markings), Current Inspections (Annual Inspection (all aircraft), 100-Hour Inspection (if used for hire/instruction), Progressive Inspection (alternative to annual)), Compliance Records - Airworthiness Directives (ADs),Form 337s (major repairs/alterations)",
                category="regulations",
                difficulty="basic"
            ),
            # EvaluationSample(
            #     question="What is the proper procedure for recovering from a power-off stall?",
            #     expected_answer="Immediately reduce angle of attack by releasing back pressure on the elevator control, apply maximum allowable power, level the wings with coordinated use of rudder and ailerons, and return to the desired flight path. The key is to reduce angle of attack first before adding power.",
            #     category="flight_maneuvers",
            #     difficulty="basic"
            # ),
            EvaluationSample(
                question="What are the performance standards for a steep turn maneuver?",
                expected_answer="Steep turns should maintain a bank angle of 45° (±5°), altitude within 100 feet of entry altitude, airspeed within 10 knots of entry airspeed, and complete a 360° turn with rollout on entry heading (±10°). The turn should be smooth and coordinated throughout.",
                category="flight_maneuvers", 
                difficulty="intermediate"
            ),
            EvaluationSample(
                question="What factors affect takeoff distance and how do they influence performance?",
                expected_answer="Takeoff distance is affected by: aircraft weight (heavier = longer), density altitude (higher = longer), wind (headwind decreases, tailwind increases), runway surface (soft/rough = longer), flap setting, and aircraft configuration. Each 1000 feet of density altitude can increase takeoff distance by approximately 10%.",
                category="performance",
                difficulty="intermediate"
            ),
            EvaluationSample(
                question="What is the difference between a chandelle and a lazy eight maneuver?",
                expected_answer="A chandelle is a maximum performance climbing turn through 180° that gains the most altitude possible for a given bank angle and power setting. A lazy eight is a maneuver with continuous changing bank and pitch attitudes in a figure-eight pattern that emphasizes smooth control coordination and altitude/airspeed management.",
                category="flight_maneuvers",
                difficulty="advanced"
            ),
            EvaluationSample(
                question="What are the typical indications of an engine fire during flight?",
                expected_answer="Engine fire indications include: engine temperature and pressure abnormalities, smoke in the cockpit, visible flames or smoke from the engine compartment, unusual engine sounds, vibration, and possible loss of engine power. Immediate action is required following emergency procedures.",
                category="emergency_procedures",
                difficulty="basic"
            ),
            EvaluationSample(
                question="How does ground effect influence aircraft performance during landing?",
                expected_answer="Ground effect occurs within one wingspan of the ground and reduces induced drag while creating a cushioning effect. This can cause the aircraft to float during landing, require less power to maintain flight, and may create a slight nose-down pitching moment. Pilots must be prepared for reduced elevator effectiveness.",
                category="aerodynamics",
                difficulty="intermediate"
            ),
        ]

    def calculate_answer_relevance(self, generated: str, expected: str) -> float:
        """
        Calculate semantic similarity between generated and expected answers
        using sentence embeddings
        """
        gen_embedding = self.embedding_model.encode([generated])
        exp_embedding = self.embedding_model.encode([expected])
        similarity = cosine_similarity(gen_embedding, exp_embedding)[0][0]
        return float(similarity)

    def evaluate_single_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate a single question-answer pair"""
        start_time = time.time()
        
        # Get RAG response
        self.rag_agent.observe(sample.question)
        response = self.rag_agent.act()
        
        latency_ms = (time.time() - start_time) * 1000

        # Calculate metrics
        relevance_score = self.calculate_answer_relevance(
            response, 
            sample.expected_answer
        )
        
        return EvaluationResult(
            question=sample.question,
            generated_answer=response,
            expected_answer=sample.expected_answer,
            answer_relevance_score=relevance_score,
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat()
        )

    def run_evaluation(self, samples: List[EvaluationSample] = None, 
                      save_results: bool = True) -> Dict[str, Any]:
        """Run complete evaluation pipeline"""
        if samples is None:
            samples = self.load_evaluation_dataset()
        
        logger.info(f"Starting evaluation with {len(samples)} samples")
        
        results = []
        for i, sample in enumerate(samples):
            logger.info(f"Evaluating sample {i+1}/{len(samples)}: {sample.question[:50]}...")
            try:
                result = self.evaluate_single_sample(sample)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating sample {i+1}: {str(e)}")
                continue
        
        # Calculate aggregate metrics
        metrics = self.calculate_aggregate_metrics(results)
        
        # Save results
        if save_results:
            self.save_results(results, metrics)
        
        return {
            'individual_results': results,
            'aggregate_metrics': metrics
        }

    def calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate aggregate metrics across all evaluation results"""
        if not results:
            return {}
        
        metrics = {
            'avg_answer_relevance': np.mean([r.answer_relevance_score for r in results]),
            'avg_latency_ms': np.mean([r.latency_ms for r in results]),
            'p95_latency_ms': np.percentile([r.latency_ms for r in results], 95),
            'total_samples': len(results)
        }
        
        return metrics

    def save_results(self, results: List[EvaluationResult], 
                    metrics: Dict[str, float]) -> None:
        """Save evaluation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'results': [asdict(r) for r in results],
                'aggregate_metrics': metrics
            }, f, indent=2)
        
        # Save summary report
        report_file = self.results_dir / f"evaluation_report_{timestamp}.md"
        self.generate_report(results, metrics, report_file)
        
        logger.info(f"Results saved to {results_file} and {report_file}")

    def generate_report(self, results: List[EvaluationResult], 
                       metrics: Dict[str, float], 
                       output_path: Path) -> None:
        """Generate human-readable evaluation report"""
        report = f"""# RAG Evaluation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Metrics

| Metric | Value |
|--------|-------|
| Average Answer Relevance | {metrics.get('avg_answer_relevance', 0):.3f} |
| Average Latency (ms) | {metrics.get('avg_latency_ms', 0):.1f} |
| P95 Latency (ms) | {metrics.get('p95_latency_ms', 0):.1f} |
| Total Samples | {metrics.get('total_samples', 0)} |

## Detailed Results

        """
        for i, result in enumerate(results):
            report += f"""
### Sample {i+1}: {result.question}

**Expected Answer:** {result.expected_answer}

**Generated Answer:** {result.generated_answer}

**Metrics:**
- Answer Relevance: {result.answer_relevance_score:.3f}
- Latency: {result.latency_ms:.1f}ms

---
            """
        
        with open(output_path, 'w') as f:
            f.write(report)

def main():
    """Main evaluation entry point"""
    # Initialize components
    vector_store = VectorStore()
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

if __name__ == "__main__":
    main()