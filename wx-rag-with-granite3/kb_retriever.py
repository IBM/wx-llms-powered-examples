#
# Copyright IBM Corp. 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# Author: Nguyen, Hung (Howie) Sy
#

import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.runnables import chain
from langchain_core.documents import Document
import threading
import os
from dotenv import load_dotenv

import sys
sys.path.append("../common_libs") # not a good pratice but it's ok in this case
from watsonx import WatsonxClient

load_dotenv()

TECH_NOTE_COLLECTION_NAME = "TechNote"

class SingletonKnowledgeBaseRetrieverMeta(type):
    _instances = {} 
    _lock = threading.Lock()  # a lock object to ensure thread safety

    def __call__(cls, *args, **kwargs):
        # double-checked locking
        if cls not in cls._instances:
            with cls._lock:  # ensure only one thread creates the instance
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
    def destroy_instance(cls):
        """
        Destroy the current instance 
        """
        with cls._lock:  # Ensure thread safety during destruction
            if cls in cls._instances:
                del cls._instances[cls]

class KnowledgeBaseRetriever(metaclass=SingletonKnowledgeBaseRetrieverMeta):
    def __init__(self):
        self._weaviate_client = weaviate.connect_to_local(
                host=os.getenv("WEAVIATE_HOSTNAME", "localhost"), 
                port=int(os.getenv("WEAVIATE_PORT", "8080")),
                grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")))
        
        self._embedding_model = WatsonxClient.request_embedding_model()

    def close_weaviate_client(self):
        if self._weaviate_client:
            self._weaviate_client.close()

    def _similarity_search_with_relevance_scores(self, query: str, collection_name, 
                                                key_property, k=3, 
                                                score_threshold=0.6):
        
        vector_store = WeaviateVectorStore(text_key=key_property,
                                                index_name=collection_name,
                                                client=self._weaviate_client,
                                                embedding=self._embedding_model) 

        docs, scores = zip(*vector_store.similarity_search_with_relevance_scores(query, 
                            k=k, score_threshold=score_threshold))
            
        for doc, score in zip(docs, scores):
            doc.metadata["score"] = score

        return docs
        
    @staticmethod
    def cleanup():
        # Close resources before destroying the instance
        singleton_kb_retriever = KnowledgeBaseRetriever()
        singleton_kb_retriever.close_weaviate_client()

        # Call the metaclass destroy_instance method
        SingletonKnowledgeBaseRetrieverMeta.destroy_instance(KnowledgeBaseRetriever)

    @staticmethod
    @chain
    def technote_retriever(query: str, **kwargs) -> list[Document]:
        singleton_kb_retriever = KnowledgeBaseRetriever()

        k = kwargs.get('k', 5)
        results = singleton_kb_retriever._similarity_search_with_relevance_scores(query, 
                                        collection_name=TECH_NOTE_COLLECTION_NAME, 
                                        key_property="note_id",
                                        k=k)
        return results

### For dev/test/demo purposes
if __name__ == "__main__":
    technote_results = KnowledgeBaseRetriever.technote_retriever.invoke("I'm having an issue relating to Websphere MQ", k=1)
    print("\nResults:")
    for result in technote_results:
        del result.metadata['content'] # don't need content as just a HTML version of 'text'
        print(f"\n{result}\n")

    KnowledgeBaseRetriever.cleanup()