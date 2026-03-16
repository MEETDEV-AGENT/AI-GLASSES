import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class EvaluationMetrics:
    """Stores evaluation metrics for a single response"""
    query: str
    response: str
    length_score: float
    relevance_score: float
    coherence_score: float
    fluency_score: float
    overall_score: float
    timestamp: str


class ResponseEvaluator:
    """Evaluates chatbot responses using various metrics"""
    
    def __init__(self):
        self.results: List[EvaluationMetrics] = []
    
    def calculate_length_score(self, response: str) -> float:
        """Score based on response length (ideal: 50-300 tokens)"""
        words = len(response.split())
        
        if words < 10:
            return 0.3
        elif words < 30:
            return 0.7
        elif words <= 300:
            return 1.0
        elif words <= 500:
            return 0.8
        else:
            return 0.5
    
    def calculate_coherence_score(self, response: str) -> float:
        """Score based on sentence structure and coherence"""
        sentences = re.split(r'[.!?]+', response.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Ideal average sentence length: 8-20 words
        if avg_sentence_length < 3:
            return 0.4
        elif avg_sentence_length <= 20:
            return 1.0
        elif avg_sentence_length <= 30:
            return 0.8
        else:
            return 0.5
    
    def calculate_fluency_score(self, response: str) -> float:
        """Score based on language fluency (no excessive punctuation/errors)"""
        score = 1.0
        
        # Check for excessive punctuation
        excessive_marks = response.count('!!') + response.count('??')
        score -= excessive_marks * 0.1
        
        # Check for common contractions and proper grammar
        if re.search(r'\b(is|are|was|were|be)\b', response):
            score += 0.1
        
        # Check for proper capitalization
        if response[0].isupper():
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def calculate_relevance_score(self, query: str, response: str) -> float:
        """Score based on response relevance to query (simple keyword matching)"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        query_words = query_words - common_words
        response_words = response_words - common_words
        
        if not query_words:
            return 0.5
        
        overlap = query_words.intersection(response_words)
        relevance = len(overlap) / len(query_words)
        
        return min(1.0, relevance)
    
    def evaluate_response(self, query: str, response: str) -> EvaluationMetrics:
        """Evaluate a single query-response pair"""
        length_score = self.calculate_length_score(response)
        coherence_score = self.calculate_coherence_score(response)
        fluency_score = self.calculate_fluency_score(response)
        relevance_score = self.calculate_relevance_score(query, response)
        
        # Weighted average: relevance (40%), length (20%), coherence (20%), fluency (20%)
        overall_score = (
            relevance_score * 0.4 +
            length_score * 0.2 +
            coherence_score * 0.2 +
            fluency_score * 0.2
        )
        
        metrics = EvaluationMetrics(
            query=query,
            response=response,
            length_score=round(length_score, 2),
            relevance_score=round(relevance_score, 2),
            coherence_score=round(coherence_score, 2),
            fluency_score=round(fluency_score, 2),
            overall_score=round(overall_score, 2),
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(metrics)
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all evaluations"""
        if not self.results:
            return {}
        
        scores = {
            'overall': [m.overall_score for m in self.results],
            'relevance': [m.relevance_score for m in self.results],
            'length': [m.length_score for m in self.results],
            'coherence': [m.coherence_score for m in self.results],
            'fluency': [m.fluency_score for m in self.results],
        }
        
        summary = {}
        for metric_name, metric_values in scores.items():
            summary[metric_name] = {
                'mean': round(sum(metric_values) / len(metric_values), 2),
                'min': round(min(metric_values), 2),
                'max': round(max(metric_values), 2),
                'total_samples': len(metric_values)
            }
        
        return summary
    
    def save_results(self, filepath: str):
        """Save evaluation results to JSON file"""
        results_dict = [asdict(m) for m in self.results]
        summary = self.get_summary()
        
        output = {
            'summary': summary,
            'detailed_results': results_dict,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Evaluation results saved to {filepath}")


class EvaluationRunner:
    """Main runner for evaluation framework"""
    
    @staticmethod
    def run_evaluation(queries_file: str = "queries.json", 
                       responses_file: str = "responses.json",
                       output_file: str = "evaluation_results.json"):
        """Run complete evaluation pipeline"""
        
        # Load data
        with open(queries_file, 'r') as f:
            queries = json.load(f)
        
        with open(responses_file, 'r') as f:
            responses = json.load(f)
        
        # Create evaluator
        evaluator = ResponseEvaluator()
        
        # Evaluate each query-response pair
        print(f"Evaluating {len(responses)} responses...\n")
        for item in responses:
            query = item.get('user_input', '')
            response = item.get('response', '')
            
            if query and response:
                metrics = evaluator.evaluate_response(query, response)
                print(f"Query: {query[:50]}...")
                print(f"  Overall Score: {metrics.overall_score}/1.0")
                print()
        
        # Save results
        evaluator.save_results(output_file)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        summary = evaluator.get_summary()
        
        for metric, stats in summary.items():
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {stats['mean']}")
            print(f"  Min: {stats['min']}")
            print(f"  Max: {stats['max']}")
            print(f"  Samples: {stats['total_samples']}")
        
        return evaluator


if __name__ == "__main__":
    evaluator = EvaluationRunner.run_evaluation()
