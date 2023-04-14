import subprocess
import sys

import weaviate
from weaviate.embedded import EmbeddedOptions
from weaviate.util import generate_uuid5

from .base import BaseANN


class Weaviate(BaseANN):
    def __init__(self, metric, ef_construction):
        self.client = weaviate.Client(
            embedded_options=EmbeddedOptions()
        )
        self.ef_construction = ef_construction
        self.distance = {
            "angular": "cosine",
            "euclidean": "l2-squared",
        }[metric]

    def fit(self, X):
        self.client.schema.create({
            "classes": [
                {
                    "class": "Vector",
                    "properties": [
                        {
                            "name": "i",
                            "dataType": ["int"],
                        }
                    ],
                    "vectorIndexConfig": {
                        "distance": self.distance,
                        "efConstruction": self.ef_construction,
                    },
                }
            ]
        })
        with self.client.batch as batch:
            batch.batch_size = 100
            for i, x in enumerate(X):
                properties = { "i": i }
                uuid = generate_uuid5(properties, "Vector")
                self.client.batch.add_data_object(
                    data_object=properties,
                    class_name="Vector",
                    uuid=uuid,
                    vector=x
                )

    def query(self, v, n):
        ret = (
            self.client.query
            .get("Vector", ["i"])
            .with_near_vector({
                "vector": v,
            })
            .with_limit(n)
            .do()
        )
        # {'data': {'Get': {'Vector': [{'i': 3618}, {'i': 8213}, {'i': 4462}, {'i': 6709}, {'i': 3975}, {'i': 3129}, {'i': 5120}, {'i': 2979}, {'i': 6319}, {'i': 3244}]}}}
        return [d["i"] for d in ret["data"]["Get"]["Vector"]]

    def __str__(self):
        return "Weaviate()"
