# Working with Multiple Environments

In many real-world machine learning applications, different components of your workflow may require different dependencies or configurations. Flyte enables you to manage this complexity by allowing multiple environments within a single project.

## Why Use Multiple Environments?

Multiple environments are useful when:
- Different parts of your workflow need different dependencies
- Some tasks require specific CPU/GPU configurations
- You're integrating specialized tools that have conflicting requirements

## The AlphaFold Example

Our AlphaFold example demonstrates this concept perfectly. The workflow involves:

1. A preprocessing task that handles data preparation using standard Python libraries
2. A prediction task that uses the specialized AlphaFold model requiring specific dependencies
3. A post-processing task that analyzes results and generates visualizations

Each of these steps requires a different set of dependencies:
- The preprocessing environment needs basic data handling libraries
- The AlphaFold environment needs specialized scientific packages and possibly GPU support
- The visualization environment needs graphing libraries

## Creating Environment Dependencies

To establish relationships between environments, use the `depends_on` parameter in the task environment configuration:

```python
preprocessing_env = flyte.TaskEnvironment(name="preprocessing")
alphafold_env = flyte.TaskEnvironment(name="alphafold", depends_on=[preprocessing_env])
viz_env = flyte.TaskEnvironment(name="visualization", depends_on=[alphafold_env])
```

This ensures that environments are built in the correct order and that deployment happens in the right sequence.

## Important Considerations

**Parent Task Dependencies:** When a task invokes other tasks with different environments, the parent task's environment must include all dependencies from the child tasks. This is necessary because the parent loads the child tasks during execution.

For example, in our AlphaFold workflow, the orchestrating task that calls the preprocessing, prediction, and visualization tasks needs access to all their dependencies:

```python
@env.task
def alphafold_workflow(sequence_data):
    preprocessed = preprocess_data(sequence_data)
    prediction = run_alphafold(preprocessed)
    return visualize_results(prediction)
```

**Alternative Approach:** If including all dependencies in the parent task becomes problematic, consider using `reference_tasks`. This approach allows you to reference tasks without requiring their dependencies in the parent environment. See the `reference_tasks` example for more details.

By properly structuring your environments in the AlphaFold example, you can seamlessly integrate different computational components with varying requirements into a unified workflow.
