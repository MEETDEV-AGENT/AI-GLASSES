"""Standalone evaluation script - evaluates existing responses without loading the model"""

from evaluation_framework import EvaluationRunner

if __name__ == "__main__":
    print("Running Evaluation Framework (Evaluation Only Mode)")
    print("="*60 + "\n")
    
    EvaluationRunner.run_evaluation(
        queries_file="queries.json",
        responses_file="responses.json",
        output_file="evaluation_results.json"
    )
