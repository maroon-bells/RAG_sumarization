import os
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.api_client import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

import constants
from models import Document

from dotenv import load_dotenv

load_dotenv()

class QdrantVectorOutput:
    """
    A sink that writes document embeddings to a Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client to use for writing.
        collection_name (str, optional): The name of the collection to write to.
            Defaults to constants.VECTOR_DB_OUTPUT_COLLECTION_NAME.
    """

    def __init__(
        self,
    ):
        self._client = self.build_qdrant_client()
        self._collection_name = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME
        self._vector_size = constants.EMBEDDING_MODEL_MAX_INPUT_LENGTH
        try:
            self._client.get_collection(collection_name=self._collection_name)
        except (UnexpectedResponse, ValueError):
            self._client.recreate_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
            )

    def build_qdrant_client(url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Builds a QdrantClient object with the given URL and API key.

        Args:
            url (Optional[str]): The URL of the Qdrant server. If not provided,
                it will be read from the QDRANT_URL environment variable.
            api_key (Optional[str]): The API key to use for authentication. If not provided,
                it will be read from the QDRANT_API_KEY environment variable.

        Raises:
            KeyError: If the QDRANT_URL or QDRANT_API_KEY environment variables are not set
                and no values are provided as arguments.

        Returns:
            QdrantClient: A QdrantClient object connected to the specified Qdrant server.
        """
        url = os.getenv("QDRANT_URL")
        api_key =  os.getenv("QDRANT_API_KEY")
        client = QdrantClient(url, api_key=api_key)

        return client

    def write(self, document: Document):
        try:
            ids, payloads = document.to_payloads()
            print(ids,payloads)
            points = [
                PointStruct(id=idx, vector=vector, payload=_payload)
                for idx, vector, _payload in zip(ids, document.embeddings, payloads)
            ]

            self._client.upsert(collection_name=self._collection_name, points=points)
            print("Added sucsessfully !")
        except Exception as e:
            print(e )

test = QdrantVectorOutput()