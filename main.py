from ingestion import ingestor
from QdrantIndexer import indexer


payload = ingestor.run_pipeline()

indexer.run_indexer(payload)