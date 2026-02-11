"""
DataFrame Report Example

Generates a sample DataFrame and renders it as a two-tab report:
  - DataFrame tab: styled HTML table via pandas
  - Profile tab: full statistical profile via ydata-profiling
"""

import random

import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="dataframe_report",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "pandas", "ydata-profiling", "setuptools"
    ),
)


@env.task(report=True)
async def generate_dataframe_report():
    import pandas as pd
    from ydata_profiling import ProfileReport

    # Generate a sample dataset
    random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "department": [random.choice(["Engineering", "Marketing", "Sales", "Finance", "HR"]) for _ in range(n)],
            "city": [random.choice(["San Francisco", "New York", "Austin", "Seattle", "Chicago"]) for _ in range(n)],
            "salary": [round(random.gauss(85000, 20000), 2) for _ in range(n)],
            "years_experience": [max(0, round(random.gauss(7, 4), 1)) for _ in range(n)],
            "satisfaction_score": [round(random.uniform(1.0, 5.0), 2) for _ in range(n)],
            "projects_completed": [random.randint(0, 30) for _ in range(n)],
            "is_remote": [random.choice([True, False]) for _ in range(n)],
        }
    )

    # Tab 1: Render the raw DataFrame as a styled HTML table
    table_tab = flyte.report.get_tab("DataFrame")
    table_tab.log(df.to_html(index=False, border=0, classes="dataframe"))

    # Tab 2: Render a full profile report using ydata-profiling
    profile = ProfileReport(df, title="Employee Dataset Profile", minimal=True)
    profile_tab = flyte.report.get_tab("Profile")
    profile_tab.log(profile.to_html())

    await flyte.report.flush.aio()


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(generate_dataframe_report)
    print(run.name)
    print(run.url)
