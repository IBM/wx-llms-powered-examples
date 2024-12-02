# A workaround to integrate embedding models from watsonx with Weaviate

## Scope

This code is to aim with development or demonstration. In some cases, best practices for development may be simplified or omitted. In real-world applications, additional effort and considerations are often required to build a robust and effective solution.

## Overview

Weaviate is an open-source vector database that supports storing vector embeddings and performing vector searches, enabling semantic search functionality.

Currently, Weaviate integrates with many AI platforms, but Watsonx is not yet directly supported. Specifically, we aim to utilize an embedding model, such as _ibm/slate-30m-english-rtrvr_. Fortunately, there is a way to integrate this model into Weaviate. This code provides a workaround based on this article https://weaviate.io/developers/weaviate/modules/custom-modules#a-replace-parts-of-an-existing-module. It implements an API that wraps client code to use the embedding model on Watsonx, allowing Weaviate to consume it seamlessly.

Additionally, a shell script is provided to assist in setting up a local Weaviate container for demonstration purposes.

## How to run

### Prerequisites

- This example has been tested in a Linux environment. However, if you are using Windows or MacOS, it should work with minor adjustments to settings or command lines as needed.
- Preferred Python version: 3.12. However, it should also work with Python 3.10 and later versions.
- Assumptions: The current working directory is wx-weaviate-embedding-api. A .env file should exist in this directory, containing the following environment variables as an example:
  
```
WATSONX_URL=https://us-south.ml.cloud.ibm.com
IBM_CLOUD_API_KEY=<your API key>
WATSONX_PROJECT_ID=<your watsonx project id>
```

### Set up and start the API

```
# Change the working directory to wx-weaviate-embedding-api

# Create and activate the virtual environment
$ python -m venv .venv
$ source .venv/bin/activate

# Install required packages 
$ pip install -r requirements. txt

# Run the API (Ctrl+C to terminate it when needed)
$ python weavite_text2vec_watsonx_api.py

# Run the test and wait for the result
$ python test.py

# If it is passed the test, then the API is ready to be used...

# Optionally, if you want to run a Weaviate container that uses this API, execute the following command. Before running it, I strongly recommend taking a look at it as you may want to update it to suit your local environment if needed.
$ source install_weaviate_container.sh
```

By default, it runs on the port 5000, so the endpoint will be "http://localhost:5000"  (assume that it is running on local)

On the Weaviate side, you need to specify these environment variables:
`TRANSFORMERS_INFERENCE_API=http://localhost:5000
`ENABLE_MODULES=text2vec-transformers`

And in the  client code, when creating a collection, specifying Vectorizer.text2vec_transformers(), e.g like this:

```
weaviate_client.collections.create(name=collection_name,
	vectorizer_config=Configure.Vectorizer.text2vec_transformers(),	
	vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE),
	properties=collection_properties)
```

## License

Apache-2.0

You may obtain a copy of the License at 
```
http://www.apache.org/licenses/LICENSE-2.0
```

## Author

Nguyen, Hung (Howie) Sy, 
\
https://github.com/howiesnguyen