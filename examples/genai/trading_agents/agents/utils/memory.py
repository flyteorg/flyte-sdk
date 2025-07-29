import uuid

import boto3
from openai import OpenAI


class FinancialSituationMemory:
    def __init__(
        self,
        name: str,
        embedding_model: str,
        dimension: int,
        region_name: str,
        bucket_name: str,
    ):
        self.name = name
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.bucket_name = bucket_name
        self.index_name = f"trading-agents-{name}"

        self.openai_client = OpenAI()
        self.client = boto3.client("s3vectors", region_name=region_name)

        self._ensure_vector_bucket_exists()
        self._ensure_index_exists()

    def _ensure_vector_bucket_exists(self):
        try:
            self.client.get_vector_bucket(vectorBucketName=self.bucket_name)
        except self.client.exceptions.NotFoundException:
            self.client.create_vector_bucket(
                vectorBucketName=self.bucket_name,
                encryptionConfiguration={"sseType": "AES256"},
            )

    def _ensure_index_exists(self):
        try:
            self.client.get_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
            )
        except self.client.exceptions.NotFoundException:
            self.client.create_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                dataType="float32",
                dimension=self.dimension,
                distanceMetric="cosine",
                metadataConfiguration={"nonFilterableMetadataKeys": ["recommendation", "source_text"]},
            )

    def get_embedding(self, text: str) -> list[float]:
        """Get OpenAI embedding for a given text"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def add_situations(self, situations_and_advice: list[tuple[str, str]]) -> None:
        """Store situation + recommendation pairs in the vector store."""
        vectors = []
        for situation, recommendation in situations_and_advice:
            vectors.append(
                {
                    "key": f"situation-{uuid.uuid4()}",
                    "data": {"float32": self.get_embedding(situation)},
                    "metadata": {
                        "source_text": situation,
                        "recommendation": recommendation,
                    },
                }
            )

        self.client.put_vectors(
            vectorBucketName=self.bucket_name,
            indexName=self.index_name,
            vectors=vectors,
        )

    def get_memories(self, current_situation: str, n_matches: int = 1) -> list[dict]:
        """Retrieve the top-n most similar situations from memory"""
        try:
            response = self.client.query_vectors(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                queryVector={"float32": self.get_embedding(current_situation)},
                topK=n_matches,
                returnDistance=True,
                returnMetadata=True,
            )
        except self.client.exceptions.NotFoundException:
            return []

        return [
            {
                "matched_situation": vector["metadata"]["source_text"],
                "recommendation": vector["metadata"]["recommendation"],
                "similarity_score": 1 - vector["distance"],
            }
            for vector in response.get("vectors", [])
        ]


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory(
        name="trader",
        embedding_model="text-embedding-3-small",
        dimension=1536,
        region_name="us-east-2",
        bucket_name="text-embeddings",
    )

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. "
            "Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. "
            "Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=10)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {e!s}")
