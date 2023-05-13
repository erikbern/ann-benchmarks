import subprocess
import sys
import uuid

import weaviate
from weaviate.embedded import EmbeddedOptions
from weaviate.util import generate_uuid5

from ..base.module import BaseANN


class Weaviate(BaseANN):
    def __init__(self, metric, max_connections, ef_construction=512):
        self.class_name = "Vector"
        self.client = weaviate.Client(embedded_options=EmbeddedOptions(version="1.19.0-beta.1"))
        self.max_connections = max_connections
        self.ef_construction = ef_construction
        self.distance = {
            "angular": "cosine",
            "euclidean": "l2-squared",
        }[metric]

    def fit(self, X):
        self.client.schema.create(
            {
                "classes": [
                    {
                        "class": self.class_name,
                        "properties": [
                            {
                                "name": "i",
                                "dataType": ["int"],
                            }
                        ],
                        "vectorIndexConfig": {
                            "distance": self.distance,
                            "efConstruction": self.ef_construction,
                            "maxConnections": self.max_connections,
                        },
                    }
                ]
            }
        )
        with self.client.batch as batch:
            batch.batch_size = 100
            for i, x in enumerate(X):
                properties = {"i": i}
                self.client.batch.add_data_object(
                    data_object=properties, class_name=self.class_name, uuid=uuid.UUID(int=i), vector=x
                )

    def set_query_arguments(self, ef):
        self.ef = ef
        schema = self.client.schema.get(self.class_name)
        schema["vectorIndexConfig"]["ef"] = ef
        self.client.schema.update_config(self.class_name, schema)

    def query(self, v, n):
        ret = (
            self.client.query.get(self.class_name, None)
            .with_additional("id")
            .with_near_vector(
                {
                    "vector": v,
                }
            )
            .with_limit(n)
            .do()
        )
        # {'data': {'Get': {'Vector': [{"_additional": {"id": "<uuid>" }}, ...]}}}

        return [uuid.UUID(res["_additional"]["id"]).int for res in ret["data"]["Get"][self.class_name]]

    def __str__(self):
        return f"Weaviate(ef={self.ef}, maxConnections={self.max_connections}, efConstruction={self.ef_construction})"
