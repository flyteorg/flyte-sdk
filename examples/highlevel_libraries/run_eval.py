import asyncio
from typing import Dict, List, Optional

from evaluator import Dataset, Evaluator, ReferenceDataset


# Define a simple word count evaluation function
async def word_count_eval(actual: Dataset, reference: Optional[ReferenceDataset] = None) -> Dict[str, str]:
    """
    Evaluate the response by counting words and checking against reference if provided.

    Args:
        actual: Dataset containing input, context, and response
        reference: Optional reference dataset with expected output

    Returns:
        Dictionary with word count and accuracy metrics
    """
    word_count = len(actual.response.split())

    result = {
        "word_count": str(word_count),
        "response_length": str(len(actual.response)),
    }

    # If reference is provided, check if response matches
    if reference is not None:
        matches = actual.response.strip() == reference.output.strip()
        result["matches_reference"] = str(matches)
        result["reference_word_count"] = str(len(reference.output.split()))

    return result


# Define an aggregation function that computes statistics
def compute_statistics(eval_results: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Aggregate evaluation results by computing mean word count and accuracy.

    Args:
        eval_results: List of individual evaluation results

    Returns:
        Dictionary with aggregated statistics
    """
    if not eval_results:
        return {"error": "No evaluation results"}

    # Compute average word count
    word_counts = [int(result["word_count"]) for result in eval_results]
    avg_word_count = sum(word_counts) / len(word_counts)

    # Compute accuracy if reference matches are available
    accuracy_stats = {}
    if "matches_reference" in eval_results[0]:
        matches = [result["matches_reference"] == "True" for result in eval_results]
        accuracy = sum(matches) / len(matches)
        accuracy_stats["accuracy"] = f"{accuracy:.2f}"
        accuracy_stats["correct_count"] = str(sum(matches))

    return {
        "mean_word_count": f"{avg_word_count:.2f}",
        "min_word_count": str(min(word_counts)),
        "max_word_count": str(max(word_counts)),
        **accuracy_stats,
    }


async def main():
    """Run the evaluation example."""

    # Create sample datasets
    datasets = [
        Dataset(
            input="What is AI?",
            context="Artificial Intelligence",
            response="AI is artificial intelligence used in computers",
        ),
        Dataset(input="Define ML", context="Machine Learning", response="ML is machine learning"),
        Dataset(
            input="What is NLP?",
            context="Natural Language Processing",
            response="NLP stands for natural language processing and is used in text analysis",
        ),
    ]

    # Create reference datasets (optional)
    reference_datasets = [
        ReferenceDataset(input="What is AI?", output="AI is artificial intelligence used in computers"),
        ReferenceDataset(input="Define ML", output="ML is machine learning technology"),
        ReferenceDataset(
            input="What is NLP?", output="NLP stands for natural language processing and is used in text analysis"
        ),
    ]

    # Create evaluator with eval function and aggregator
    evaluator = Evaluator(eval_func=word_count_eval, aggregate_func=compute_statistics)

    print("Running evaluation with reference dataset...")
    results = await evaluator.run(dataset=datasets, reference_dataset=reference_datasets, mode="remote")

    print("\nEvaluation Results:")
    print(f"Number of samples: {results['num_samples']}")

    # Print aggregated results
    print("\nAggregated Metrics:")
    for key, value in results.items():
        if key.startswith("aggregated_"):
            print(f"  {key}: {value}")

    # Example without reference dataset
    print("\n" + "=" * 50)
    print("Running evaluation without reference dataset...")
    evaluator_no_ref = Evaluator(eval_func=word_count_eval, aggregate_func=compute_statistics)

    results_no_ref = await evaluator_no_ref.run(dataset=datasets, mode="remote")

    print("\nEvaluation Results (no reference):")
    print(f"Number of samples: {results_no_ref['num_samples']}")

    print("\nAggregated Metrics:")
    for key, value in results_no_ref.items():
        if key.startswith("aggregated_"):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
