import os
from datetime import datetime
from embeddings import EmbeddingModelSingleton
from models import Data
from qdrant import QdrantVectorOutput

model = EmbeddingModelSingleton(cache_dir=None)


files = os.listdir("data/")

for file  in files:
    file_path = "data/" + file
    print(f"Loading {file}")
    article = Data(data_name=file_path,added_at=datetime.now(),post_url="")
    document = article.to_document()
    document.compute_chunks(model)
    document.compute_embeddings(model)
    qdrantoutput = QdrantVectorOutput()
    qdrantoutput.write(document)
    print(f"Completed {file}")
