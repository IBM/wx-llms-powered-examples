#
# Copyright IBM Corp. 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# Author: Nguyen, Hung (Howie) Sy
#

import weaviate
from weaviate.classes.config import (
    Property,
    DataType,
    Configure,
    VectorDistances
)

from weaviate.collections import Collection
import ijson
import json
import os
from dotenv import load_dotenv

load_dotenv()

TECH_NOTE_COLLECTION_NAME = "TechNote"

weaviate_client = weaviate.connect_to_local(
                host=os.getenv("WEAVIATE_HOSTNAME", "localhost"), 
                port=int(os.getenv("WEAVIATE_PORT", "8080")),
                grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")))

def create_collection(
    collection_name, collection_properties, delete_if_exists=False
) -> Collection | None:
    collection = weaviate_client.collections.get(collection_name)

    if delete_if_exists and collection.exists():
        weaviate_client.collections.delete(collection_name)

    if collection.exists():
        print(f"The {collection_name} has been existed already")
        return collection
    else:
        print(f"Creating the collection {collection_name}...")
        collection = weaviate_client.collections.create(
            collection_name,
            vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
            properties=collection_properties,
        )
        print(f"The collection {collection_name} has been created")
        return collection



def export_technotes_to_file(source_filepath, output_filepath, 
                             string_filter_in_text_field:str = None,
                             ):
    """"An utility"""
    print(f"Extracting data items from file {source_filepath}, and write to file {output_filepath}...")
   
    with open(source_filepath, "rb") as input_file, open(output_filepath, "w") as output_file:
        output_file.write("{\n")

        first_item = True
        for (key, note) in ijson.kvitems(input_file, ""):

            if string_filter_in_text_field is None or (string_filter_in_text_field is not None and string_filter_in_text_field in note['text']):
                if not first_item:
                    output_file.write(",\n")
               
                output_file.write(f'"{key}" : ')
                json.dump(note, output_file, indent=2)

                first_item = False
        
        output_file.write("\n}") 

    print(f"Finished exporting data to {output_filepath}")

def import_tech_note_data(filename, delete_and_recreate_collection=False):
    SCHEMA = [
        Property(name="note_id", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT, skip_vectorization=True ),
        Property(name="title", data_type=DataType.TEXT),
        Property(name="text", data_type=DataType.TEXT),        
        Property(name="note_metadata", data_type=DataType.OBJECT,
                nested_properties=[
                    Property(name="sourceDocumentId", data_type=DataType.TEXT),
                    Property(name="date", data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name="productName", data_type=DataType.TEXT),
                    Property(name="productId", data_type=DataType.TEXT),
                    Property(name="canonicalUrl", data_type=DataType.TEXT, skip_vectorization=True)]
                )
        ]
    
    tech_notes = weaviate_client.collections.get(TECH_NOTE_COLLECTION_NAME)

    if delete_and_recreate_collection or not tech_notes.exists():
        tech_notes = create_collection(collection_name=TECH_NOTE_COLLECTION_NAME,
            collection_properties=SCHEMA,
            delete_if_exists=True,
        )

    print(f"Importing data from file {filename} into {TECH_NOTE_COLLECTION_NAME} (it may take several minutes)...")
    with open(filename, "rb") as f:
        for (key, note) in ijson.kvitems(f, ""):
            try:
                lowercased_key_item = {k.lower(): v for k, v in note.items()}

                # rename id to note_id
                lowercased_key_item['note_id'] = lowercased_key_item.pop('id')
                lowercased_key_item['note_metadata'] = lowercased_key_item.pop('metadata')

                tech_notes.data.insert(lowercased_key_item)
                # print(f"{lowercased_key_item}\n\n")

                print(f"{lowercased_key_item['note_id']}",  end=", ", flush=True)
            except Exception as e:
                print(f"\nError inserting item: {e}")
                break

        print()

    return tech_notes


###
if __name__ == "__main__":
    try:
        if not weaviate_client.is_ready():
            raise Exception("Weaviate is not ready")

        tech_notes = import_tech_note_data("techqa_technote_faq_samples.json",
                                            delete_and_recreate_collection=True)

        tech_notes = weaviate_client.collections.get(TECH_NOTE_COLLECTION_NAME)

        count_result = tech_notes.aggregate.over_all(total_count=True)
        print(f"The current number of objects in {TECH_NOTE_COLLECTION_NAME}: {count_result.total_count}")

        if tech_notes.exists():
            # verify
            query = "How can I change ITAG setting for InfoSphere DataStage?"
            print(f"\n*** Try querying the Weaviate database with the query: {query}")
            result = tech_notes.query.near_text(query=query, limit=1)
            print(f"*** Result:\n\n{result.objects[0].properties['text']}")

    except Exception as error:
        print(f"Error: {error}")
    finally:
        weaviate_client.close()

