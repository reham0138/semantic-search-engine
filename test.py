
import numpy as np
from app.indexing.vector_index import VectorIndex

index = VectorIndex(dimension=3)

documents = np.array([
    [1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0],
    [0.0, 1.0, 0.0]
], dtype=np.float32)

document_ids = ["product_1", "product_2", "product_3"]

index.add(documents, document_ids)

query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

results = index.search(query, top_k=2)

print("Total vectors:", index.total_vectors)
print("Search results:", results)
