from ingestion import ingestor
from QdrantIndexer import indexer


payload = ingestor.run_pipeline()

indexer.run_indexer(payload)

query_string =  "what are the most common IT concerns?"

search = indexer.vector_search(query_string)

print(search)