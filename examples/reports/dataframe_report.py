"""
DataFrame Report Example

Generates a sample DataFrame and renders it as a two-tab report:
  - DataFrame tab: styled HTML table via pandas
  - Charts tab: salary distribution and department breakdown via matplotlib
"""

import random

import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="dataframe_report",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "pandas", "matplotlib"
    ),
)


@env.task(report=True)
async def generate_dataframe_report():
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

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

    # Tab 2: Render matplotlib charts as embedded images
    charts_tab = flyte.report.get_tab("Charts")

    # Helper to convert a matplotlib figure to an HTML <img> tag
    def fig_to_html(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return f'<img src="data:image/png;base64,{b64}" />'

    # Chart 1: Salary distribution histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["salary"], bins=25, edgecolor="white", color="#7652a2")
    ax.set_title("Salary Distribution")
    ax.set_xlabel("Salary ($)")
    ax.set_ylabel("Count")
    charts_tab.log(fig_to_html(fig))

    # Chart 2: Average salary by department (bar chart)
    fig, ax = plt.subplots(figsize=(8, 4))
    dept_salary = df.groupby("department")["salary"].mean().sort_values()
    dept_salary.plot.barh(ax=ax, color="#7652a2", edgecolor="white")
    ax.set_title("Average Salary by Department")
    ax.set_xlabel("Average Salary ($)")
    charts_tab.log(fig_to_html(fig))

    await flyte.report.flush.aio()


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(generate_dataframe_report)
    print(run.name)
    print(run.url)
