# /// script
# requires-python = "==3.13"
# dependencies = [
#    "gql>=3.4.1",
#    "flyte>=0.2.0b20",
# ]
# ///

"""
GraphQL Data Processing Example

This example demonstrates how to use Flyte tasks to call a public GraphQL server
using the `gql` library. The example queries the Countries GraphQL API to fetch
country data and process it.

To run this example:
1. Install the required dependencies: pip install gql aiohttp
2. Run: flyte run graphql_sync.py fetch_countries_workflow
"""

from typing import List

from gql import Client, gql
from gql.transport.httpx import HTTPXTransport

import flyte

env = flyte.TaskEnvironment(
    name="graphql_sync",
    image=flyte.Image.from_uv_script(__file__, name="graphql_sync", pre=True),
)


@env.task
def fetch_countries() -> List[dict]:
    """
    Fetch all countries from the Countries GraphQL API.

    Returns:
        List of country dictionaries containing code, name, and continent
    """
    # Define the GraphQL query
    query = gql("""
        query GetCountries {
            countries {
                code
                name
                continent {
                    name
                }
            }
        }
    """)

    # Create the transport and client
    transport = HTTPXTransport(url="https://countries.trevorblades.com/")
    client = Client(transport=transport, fetch_schema_from_transport=True)

    result = client.execute(query)

    # Extract and format the countries data
    countries = []
    for country in result["countries"]:
        countries.append({"code": country["code"], "name": country["name"], "continent": country["continent"]["name"]})

    return countries


@env.task
def analyze_countries(countries: List[dict]) -> dict:
    """
    Analyze the countries data and return statistics.

    Args:
        countries: List of country dictionaries

    Returns:
        Dictionary containing analysis results
    """
    continent_counts = {}
    total_countries = len(countries)

    for country in countries:
        continent = country["continent"]
        continent_counts[continent] = continent_counts.get(continent, 0) + 1

    return {"total_countries": total_countries, "continent_distribution": continent_counts}


@env.task
def fetch_countries_workflow() -> dict:
    """
    Main workflow that fetches countries and analyzes them.

    Returns:
        Analysis results dictionary
    """
    # Fetch all countries
    countries = fetch_countries()

    # Analyze the data
    analysis = analyze_countries(countries)
    return analysis


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(fetch_countries_workflow)
    print(run.url)
