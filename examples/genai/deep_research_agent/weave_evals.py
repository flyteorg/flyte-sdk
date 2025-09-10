# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==0.2.0b7",
#    "flyteidl>=1.15.4b0",
#    "weave==0.51.51",
#    "datasets==3.6.0",
#    "huggingface-hub==0.32.6",
#    "litellm==1.72.2",
#    "tavily-python==0.7.5",
# ]
# ///

import os

import weave
from agent import research_topic
from datasets import load_dataset
from huggingface_hub import login
from libs.utils.log import AgentLogger
from litellm import completion

import flyte

logging = AgentLogger()


weave.init(project_name="deep-research-agent")

env = flyte.TaskEnvironment(name="deep-research-agent-eval")


@weave.op
def llm_as_a_judge_scoring(answer: str, output: str, question: str) -> bool:
    prompt = f"""
    Given the following question and answer, evaluate the answer against the correct answer:

    <question>
    {question}
    </question>

    <agent_answer>
    {output}
    </agent_answer>

    <correct_answer>
    {answer}
    </correct_answer>

    Note that the agent answer might be a long text containing a lot of information or it might be a short answer.

    You should read the entire text and think if the agent answers the question somewhere
    in the text. You should try to be flexible with the answer but careful.

    For example, answering with names instead of name and surname is fine.

    The important thing is that the answer of the agent either contains the correct answer or is equal to
    the correct answer.

    <reasoning>
    The agent answer is correct because I can read that ....
    </reasoning>

    <answer>
    1
    </answer>

    Otherwise, return

    <reasoning>
    The agent answer is incorrect because there is ...
    </reasoning>

    <answer>
    0
    </answer>

    """

    messages = [
        {
            "role": "system",
            "content": "You are an helpful assistant that returns a number between 0 and 1.",
        },
        {"role": "user", "content": prompt},
    ]
    answer = (
        completion(
            model="together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.0,
        )
        .choices[0]  # type: ignore
        .message["content"]  # type: ignore
    )

    return bool(int(answer.split("<answer>")[1].split("</answer>")[0].strip()))


def authenticate_huggingface():
    """Authenticate with Hugging Face Hub using token from environment variable."""
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError(
            "HUGGINGFACE_TOKEN environment variable not set. "
            "Please set it with your token from https://huggingface.co/settings/tokens"
        )

    try:
        login(token=token)
        print("Successfully authenticated with Hugging Face Hub")
    except Exception as e:
        raise RuntimeError(f"Failed to authenticate with Hugging Face Hub: {e!s}")


@env.task
async def load_questions(
    dataset_names: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Load questions from the specified Hugging Face dataset configurations.

    Args:
        dataset_names: List of dataset configurations to load
                      Options:
                          "smolagents:simpleqa",
                          "hotpotqa",
                          "simpleqa",
                          "together-search-bench"
                      If None, all available configurations except hotpotqa will be loaded

    Returns:
        List of question-answer pairs
    """
    if dataset_names is None:
        dataset_names = ["smolagents:simpleqa"]

    all_questions = []

    # Authenticate with Hugging Face Hub (once and for all)
    authenticate_huggingface()

    for dataset_name in dataset_names:
        print(f"Loading dataset: {dataset_name}")

        try:
            if dataset_name == "together-search-bench":
                # Load Together-Search-Bench dataset
                dataset_path = "togethercomputer/together-search-bench"
                ds = load_dataset(dataset_path)
                if "test" in ds:
                    split_data = ds["test"]
                else:
                    print(f"No 'test' split found in dataset at {dataset_path}")
                    continue

                for i in range(len(split_data)):
                    item = split_data[i]
                    question_data = {
                        "question": item["question"],
                        "answer": item["answer"],
                        "dataset": item.get("dataset", "together-search-bench"),
                    }
                    all_questions.append(question_data)

                print(f"Loaded {len(split_data)} questions from together-search-bench dataset")
                continue

            elif dataset_name == "hotpotqa":
                # Load HotpotQA dataset (using distractor version for validation)
                ds = load_dataset("hotpotqa/hotpot_qa", "distractor", trust_remote_code=True)
                split_name = "validation"
            elif dataset_name == "simpleqa":
                ds = load_dataset("basicv8vc/SimpleQA")
                split_name = "test"
            else:
                # Strip "smolagents:" prefix when loading the dataset
                actual_dataset = dataset_name.split(":")[-1]
                ds = load_dataset("smolagents/benchmark-v1", actual_dataset)
                split_name = "test"

        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e!s}")
            continue  # Skip this dataset if it fails to load

        print(f"Dataset structure for {dataset_name}: {ds}")
        print(f"Available splits: {list(ds)}")

        split_data = ds[split_name]  # type: ignore

        for i in range(len(split_data)):
            item = split_data[i]

            if dataset_name == "hotpotqa":
                # we remove questions that are easy or medium (if any) just to reduce the number of questions
                if item["level"] != "hard":
                    continue

                question_data = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "dataset": dataset_name,
                }
            elif dataset_name == "simpleqa":
                # Handle SimpleQA dataset format
                question_data = {
                    "question": item["problem"],
                    "answer": item["answer"],
                    "dataset": dataset_name,
                }
            else:
                question_data = {
                    "question": item["question"],
                    "answer": item["true_answer"],
                    "dataset": dataset_name,
                }

            all_questions.append(question_data)

    print(f"Loaded {len(all_questions)} questions in total")
    return all_questions


@weave.op
async def predict(question: str):
    return await research_topic(topic=str(question))


@env.task
async def main(datasets: list[str] = ["together-search-bench"], limit: int | None = 1):
    questions = await load_questions(datasets)

    if limit is not None:
        questions = questions[:limit]
        print(f"Limited to {len(questions)} question(s)")

    evaluation = weave.Evaluation(dataset=questions, scorers=[llm_as_a_judge_scoring])
    await evaluation.evaluate(predict)


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.with_runcontext(raw_data_path="data").run(main)
