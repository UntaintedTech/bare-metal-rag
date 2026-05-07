import os
from typing import Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv()
hf_token = os.environ.get('HF_TOKEN')

class VectorStoreIndexer:

    def __init__(
            self,
            collection_name: str = 'IT_Knowledge_collection',
            top_k: int = 3,
            vector_size: int = 1024,
            embedding_model: str = "BAAI/bge-m3",
            distance_metric = Distance.COSINE,
            client_url: str = "http://localhost:6333",
    ):
        # self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.distance_metric = distance_metric
        self.vector_size = vector_size
        self.sentence_transformer = SentenceTransformer(embedding_model, token=hf_token)
        self.client = QdrantClient(url=client_url)

    def _create_collection(self):

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=self.distance_metric)
        )

    def _embed_text(self, payload_text: str) -> list[float]:
        vector = self.sentence_transformer.encode(payload_text).tolist()
        return vector
    

    def _embed_batch_texts(self, payloads: list[dict]) -> list[list]:
        vectors = []
        payload_texts =[]

        for payload in payloads:
            payload_text = payload["text"]
            payload_texts.append(payload_text)

        embedded_payload = self._embed_text(payload_texts)
        vectors.extend(embedded_payload)
        
        return vectors

    def _build_points(self, payloads, vectors) -> list:
        points = []
        #  change the index for uuid after development pahse
        if len(payloads) == len(vectors):
            for index, (payload, vector) in enumerate(zip(payloads, vectors)):
                point_id = index
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        else:
            print("Payload length and vector length mismatched.")
        
        return points

    def _upsert_points(self, point_struct: list[PointStruct]):

        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=point_struct
            )
        return operation_info.status
    
    def _parse_search_results(self, results: list) -> list[dict]:
        parsed_results = []

        for result in results:
            score = result.score
            text = result.payload["text"]
            source_file = result.payload["source_file"]
            page_start = result.payload["page_start"]
            page_end = result.payload["page_end"]
            chunk_id = result.payload["chunk_id"]

            parsed_result = {
                "score": score,
                "text": text,
                "source_file": source_file,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_id": chunk_id,
            }
            parsed_results.append(parsed_result)
        
        return parsed_results

    def vector_search(self, query_string: str, top_k=None) -> list[dict]:
        search_item = self._embed_text(query_string)

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=search_item,
            with_payload=True,
            limit=top_k or self.top_k
        ).points

        parsed_results = self._parse_search_results(search_result)
        return parsed_results


    def run_indexer(self, payloads: list[dict]):
        if self.client.collection_exists(collection_name=f"{self.collection_name}"): 
            self.client.delete_collection(collection_name=f"{self.collection_name}")
            
        self._create_collection()
        vectors = self._embed_batch_texts(payloads)
        point_struct = self._build_points(payloads, vectors)
        operation_info = self._upsert_points(point_struct)
        print(f"\nUpsert status: {operation_info}\n")

        return operation_info

indexer = VectorStoreIndexer()
