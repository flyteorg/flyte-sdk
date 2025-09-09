# /// script
# requires-python = "==3.13"
# dependencies = [
#    "gql>=3.4.1",
#    "nest-asyncio>=1.6.0",
#    "aiohttp>=3.8.0",
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

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List


import flyte
import nest_asyncio
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.httpx import HTTPXTransport


nest_asyncio.apply()


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

    # ❌ This raises an error:
    # Cannot run client.execute(query) if an asyncio loop is running. Use 'await client.execute_async(query)' instead.
    transport = AIOHTTPTransport(url="https://countries.trevorblades.com/")

    # ✅ This works:
    # Using httpx transport instead of aiohttp fixes the issue
    # transport = HTTPXTransport(url="https://countries.trevorblades.com/")
    client = Client(transport=transport, fetch_schema_from_transport=True)

    result = client.execute(query)
    
    # Extract and format the countries data
    countries = []
    for country in result["countries"]:
        countries.append({
            "code": country["code"],
            "name": country["name"],
            "continent": country["continent"]["name"]
        })
    
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
    
    return {
        "total_countries": total_countries,
        "continent_distribution": continent_counts
    }


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
    # import flyte.git

    # flyte.init_from_config(flyte.git.config_from_root())
    flyte.init()
    run = flyte.run(fetch_countries_workflow)
    print(run.url)
