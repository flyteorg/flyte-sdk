# LLM Evaluation Framework

A flexible, high-level evaluation framework for assessing Large Language Model (LLM) outputs using Flyte's distributed execution capabilities.

## Overview

The **Evaluator** is a framework designed to streamline the evaluation of LLM-generated responses. It provides:

- **Parallel Evaluation**: Execute evaluations concurrently across datasets using asyncio
- **Flexible Metrics**: Define custom evaluation functions with or without reference data
- **Aggregation Support**: Combine individual results into aggregate statistics
- **Visual Reports**: Generate interactive HTML dashboards with Plotly visualizations
- **Distributed Execution**: Run evaluations locally or distribute across Flyte clusters

## What is the Evaluator Used For?

The Evaluator framework is ideal for:

1. **Model Quality Assessment**: Evaluate LLM outputs against specific criteria (accuracy, relevance, coherence)
2. **A/B Testing**: Compare different model versions or prompt strategies
3. **Regression Testing**: Ensure model updates don't degrade performance
4. **Dataset Analysis**: Understand model behavior across diverse inputs
5. **Production Monitoring**: Track model performance metrics over time

## Core Components

### 1. Evaluator Class

The main interface for running evaluations:

```python
from evaluator import Evaluator, Dataset

evaluator = Evaluator(
    eval_func=your_evaluation_function,  # Required: How to score each sample
    aggregate_func=your_aggregator       # Optional: How to combine results
)

# Run evaluation
results = await evaluator.run(
    dataset=datasets,
    reference_dataset=reference_data,  # Optional
    mode="remote"  # "local" or "remote"
)
```

### 2. Dataset Models

**Dataset**: Contains the data to evaluate
```python
Dataset(
    input="What is AI?",           # The input/prompt
    context="Artificial Intelligence",  # Additional context
    response="AI is..."             # The LLM's response
)
```

**ReferenceDataset**: Optional ground truth for comparison
```python
ReferenceDataset(
    input="What is AI?",
    output="Expected answer"
)
```

### 3. Evaluation Protocols

**EvaluationMetric**: Async function that scores individual samples
```python
async def my_eval(actual: Dataset, reference: Optional[ReferenceDataset] = None) -> Dict[str, str]:
    # Your evaluation logic
    return {"score": "0.95", "reasoning": "..."}
```

**AggregationMetric**: Function that combines individual results
```python
def my_aggregator(eval_results: List[Dict[str, str]]) -> Dict[str, str]:
    # Compute aggregate statistics
    return {"mean_score": "0.87", "std_dev": "0.12"}
```

## Usage Example: run_eval.py

The `run_eval.py` file demonstrates a **simple word-count evaluation** - a basic example to illustrate the framework's structure.

### What run_eval.py Does

1. **Defines a Simple Evaluation Function** (`word_count_eval`):
   - Counts words in the response
   - Measures response length
   - Optionally checks exact match against reference data

2. **Defines an Aggregation Function** (`compute_statistics`):
   - Computes mean, min, max word counts
   - Calculates accuracy when reference data is provided

3. **Runs Two Evaluation Scenarios**:
   - **With reference dataset**: Compares responses to expected outputs
   - **Without reference dataset**: Analyzes responses without ground truth

### Running the Example


#### Local Mode is also possible install plotly, else you can skip it
```bash
# Install dependencies
pip install plotly

# Run the evaluation
python run_eval.py
```

This will:
- Evaluate 3 sample datasets
- Execute evaluations in parallel on Flyte
- Generate an interactive HTML report with visualizations
- Print aggregated metrics to console

### Sample Output

```
Running evaluation with reference dataset...
Evaluation results: https://your-flyte-cluster/executions/...

Evaluation Results:
Number of samples: 3

Aggregated Metrics:
  aggregated_mean_word_count: 10.33
  aggregated_min_word_count: 4
  aggregated_max_word_count: 14
  aggregated_accuracy: 0.67
  aggregated_correct_count: 2
```

### Interactive Report

The framework generates a beautiful HTML report featuring:
- **Distribution plots**: Box plots showing metric distributions across samples
- **Aggregated metrics**: High-level statistics in card format
- **Detailed table**: Per-sample results for drill-down analysis

## Advanced Evaluation Methods

While `run_eval.py` demonstrates a simple word-counting approach, production LLM evaluation typically requires more sophisticated methods:

### 1. LLM-as-a-Judge

Use a powerful LLM to evaluate responses from another model:

```python
async def llm_judge_eval(actual: Dataset, reference: Optional[ReferenceDataset] = None) -> Dict[str, str]:
    """Use GPT-4 or Claude to score response quality."""
    prompt = f"""
    Evaluate this response on a scale of 1-10:
    Input: {actual.input}
    Response: {actual.response}

    Criteria: Accuracy, Relevance, Coherence
    """

    judge_response = await call_llm(prompt)  # Your LLM API call
    return {
        "score": extract_score(judge_response),
        "reasoning": judge_response
    }
```

**Use Cases**: Subjective quality metrics, multi-dimensional scoring, nuanced evaluation

### 2. Semantic Similarity

Measure semantic equivalence using embeddings:

```python
async def semantic_similarity_eval(actual: Dataset, reference: ReferenceDataset) -> Dict[str, str]:
    """Compute cosine similarity between embeddings."""
    actual_embedding = await get_embedding(actual.response)
    ref_embedding = await get_embedding(reference.output)

    similarity = cosine_similarity(actual_embedding, ref_embedding)
    return {"semantic_similarity": f"{similarity:.3f}"}
```

**Libraries**: sentence-transformers, OpenAI embeddings, Cohere
**Use Cases**: Paraphrase detection, answer equivalence

### 3. Traditional NLP Metrics

Classical metrics for text comparison:

```python
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

async def nlp_metrics_eval(actual: Dataset, reference: ReferenceDataset) -> Dict[str, str]:
    """Compute BLEU, ROUGE, and METEOR scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    rouge_scores = scorer.score(reference.output, actual.response)
    bleu_score = sentence_bleu([reference.output.split()], actual.response.split())

    return {
        "bleu": f"{bleu_score:.3f}",
        "rouge1": f"{rouge_scores['rouge1'].fmeasure:.3f}",
        "rougeL": f"{rouge_scores['rougeL'].fmeasure:.3f}"
    }
```

**Use Cases**: Machine translation, summarization, question answering

### 4. Factual Consistency Checking

Verify factual accuracy against source material:

```python
async def factual_consistency_eval(actual: Dataset, reference: Optional[ReferenceDataset] = None) -> Dict[str, str]:
    """Check for hallucinations and factual errors."""
    # Use fact-checking models or knowledge bases
    claims = extract_claims(actual.response)
    verified_claims = await verify_claims(claims, actual.context)

    accuracy = sum(verified_claims) / len(claims)
    return {
        "factual_accuracy": f"{accuracy:.2f}",
        "verified_claims": str(sum(verified_claims)),
        "total_claims": str(len(claims))
    }
```

**Tools**: FActScore, SummaC, retrieval-based verification
**Use Cases**: RAG systems, fact-based QA, content generation

### 5. Human Evaluation

Incorporate human judgment into the evaluation loop:

```python
async def human_eval(actual: Dataset, reference: Optional[ReferenceDataset] = None) -> Dict[str, str]:
    """Collect human ratings for response quality."""
    # Could integrate with labeling platforms (Label Studio, Scale AI)
    rating = await request_human_rating(actual)

    return {
        "human_rating": str(rating.score),
        "annotator_id": rating.annotator,
        "confidence": str(rating.confidence)
    }
```

**Use Cases**: Gold standard evaluation, edge case analysis, model calibration

### 6. Task-Specific Metrics

Custom metrics for specific applications:

- **Code Generation**: Execution success, test coverage, cyclomatic complexity
- **Translation**: chrF++, TER, human adequacy/fluency ratings
- **Summarization**: Compression ratio, coverage, faithfulness
- **Dialogue**: Coherence, engagement, appropriateness
- **Reasoning**: Logical correctness, step-by-step accuracy

## Architecture Benefits

### Parallel Execution

All evaluations run concurrently using `asyncio.gather`:

```python
tasks = [_run_single_eval(eval_func, item) for item in dataset]
results = await asyncio.gather(*tasks)  # Parallel execution
```

### Distributed Computing

Run evaluations on Flyte clusters for large-scale assessments:

- **Local mode**: Quick testing in current process
- **Remote mode**: Distributed execution with resource isolation
- **Scalability**: Handle thousands of samples with configurable concurrency

### Reusability

Tasks use Flyte's reuse policy to avoid redundant computation:

```python
env = flyte.TaskEnvironment(
    "evaluator",
    reusable=flyte.ReusePolicy(replicas=1, concurrency=10),  # Cache results
    resources=flyte.Resources(cpu=1, memory="1Gi")
)
```

## Customization Guide

### Creating Custom Evaluators

1. **Define your evaluation function**:
```python
async def my_custom_eval(actual: Dataset, reference: Optional[ReferenceDataset] = None) -> Dict[str, str]:
    # Your evaluation logic
    return {"metric1": "value1", "metric2": "value2"}
```

2. **Optionally define an aggregator**:
```python
def my_aggregator(results: List[Dict[str, str]]) -> Dict[str, str]:
    # Combine results
    return {"overall_score": "..."}
```

3. **Create and run the evaluator**:
```python
evaluator = Evaluator(eval_func=my_custom_eval, aggregate_func=my_aggregator)
results = await evaluator.run(dataset=my_data, mode="remote")
```

## Best Practices

1. **Keep Metrics Interpretable**: Return human-readable scores and explanations
2. **Use Reference Data Wisely**: Not all evaluations need ground truth
3. **Combine Multiple Metrics**: Use different evaluation functions for comprehensive assessment
4. **Monitor Execution**: Check Flyte UI for task status and debugging
5. **Version Your Evaluators**: Track evaluation logic alongside model versions

## Files Overview

- **`evaluator.py`**: Core evaluation framework and Flyte tasks
- **`eval_reporting.py`**: HTML report generation with Plotly visualizations
- **`run_eval.py`**: Simple word-count evaluation example
- **`README.md`**: This documentation

## Next Steps

- Explore LLM-as-a-judge implementations
- Integrate with your model serving pipeline
- Add custom metrics for your specific use case
- Scale evaluations across larger datasets using Flyte clusters
- Build automated evaluation workflows with Flyte's scheduling capabilities
