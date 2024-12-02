#
# Copyright IBM Corp. 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# Author: Nguyen, Hung (Howie) Sy
#

#! /bin/bash

VOLUME_NAME="weaviatedata"
CONTAINER_NAME="myweaviate"
IMAGE="cr.weaviate.io/semitechnologies/weaviate:1.27.6"

check_podman() {
    if ! command -v podman &> /dev/null; then
        echo "Podman is required but not installed. Please install Podman."
        exit 1
    fi
}

# Function to create a new volume
create_weaviate_data_volume() {
    echo "Creating the podman volume: $VOLUME_NAME"
    podman volume create "$VOLUME_NAME"
    if [ $? -eq 0 ]; then
        echo "Volume '$VOLUME_NAME' created successfully."
    else
        echo "Failed to create volume '$VOLUME_NAME'."
        exit 1
    fi
}

create_and_run_weaviate_container() {
    echo "Creating and running the container: $CONTAINER_NAME"

    podman run -dt --name "$CONTAINER_NAME" \
    -p 8082:8080 -p 50051:50051 \
    -v weaviatedata:/data \
    --env PERSISTENCE_DATA_PATH=/data \
    --env QUERY_DEFAULTS_LIMIT=20 \
    --env AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
    --env AUTOSCHEMA_ENABLED=false \
    --env ENABLE_MODULES='text2vec-transformers' \
    --env TRANSFORMERS_INFERENCE_API='http://host.containers.internal:5000' \
    --env DEFAULT_VECTORIZER_MODULE='text2vec-transformers' \
    "$IMAGE"
    if [ $? -eq 0 ]; then
        echo "Container '$CONTAINER_NAME' created successfully."
    else
        echo "Failed to create container '$CONTAINER_NAME'."
        exit 1
    fi
}

### main ###

check_podman

if podman volume exists "$VOLUME_NAME"; then
    echo "Volume '$VOLUME_NAME' already exists."
else
    # Create the data volume if not exist
    echo "Volume '$VOLUME_NAME' does not exist."
    create_weaviate_data_volume
fi

if podman container exists "$CONTAINER_NAME"; then
    echo "Container '$CONTAINER_NAME' already exists."
else
    echo "Container '$CONTAINER_NAME' does not exist."
    create_and_run_weaviate_container
fi

if podman ps --filter "name=$CONTAINER_NAME" --format "{{.Status}}" | grep -q "Up"; then
    echo -e "\nContainer '$CONTAINER_NAME' is running:"
else
    echo -e "\nContainer '$CONTAINER_NAME' is not running:"
fi
podman container ls -a --filter "name=$CONTAINER_NAME"
