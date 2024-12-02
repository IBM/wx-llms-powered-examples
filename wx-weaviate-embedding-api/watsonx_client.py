#
# Copyright IBM Corp. 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# Author: Nguyen, Hung (Howie) Sy
#

import os
from dotenv import load_dotenv
load_dotenv()

class WatsonxClient():

    _initialized = False  # Class-level flag to track initialization

    @classmethod
    def initialize(cls):
        if not cls._initialized:
            # print("Load watsonx Python modules...")
            global EmbedTextParamsMetaNames, EmbeddingTypes, Embeddings, WatsonxLLM, ModelTypes, GenParams, DecodingMethods
            from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes, ModelTypes, DecodingMethods
            from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames, GenTextParamsMetaNames as GenParams
            from ibm_watsonx_ai.foundation_models.embeddings import Embeddings
            from langchain_ibm import WatsonxLLM
            cls._initialized = True

    @staticmethod
    def _initialization(method):
        """A initialization decorator for static methods"""
        def wrapper(*args, **kwargs):
            WatsonxClient.initialize()  
            return method(*args, **kwargs)
        return wrapper
    
    @staticmethod
    def _get_watsonx_url():
        return os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    
    @staticmethod
    def _get_cloud_api_key():
        return os.getenv("IBM_CLOUD_API_KEY")
    
    @staticmethod
    def _get_project_id():
        return os.getenv("WATSONX_PROJECT_ID")
    
    @staticmethod
    @_initialization
    def request_embedding_model(model_id="ibm/slate-30m-english-rtrvr",
                                truncate_input_tokens=512):
            embed_params = {
                EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: truncate_input_tokens,
                EmbedTextParamsMetaNames.RETURN_OPTIONS: {
                    'input_text': False
                }
            }
            cloud_credentials = {
                "url":  WatsonxClient._get_watsonx_url(),
                "apikey": WatsonxClient._get_cloud_api_key()
            }
            embeddings_model = Embeddings(
                model_id = model_id,
                project_id = WatsonxClient._get_project_id(),
                credentials = cloud_credentials,
                params=embed_params
            )

            return embeddings_model

    @staticmethod
    @_initialization
    def connect_llm(model_id="ibm/granite-13b-instruct-v2",
                    decoding_method = "greedy", 
                    temperature = 0.7, top_p = 1.0, top_k = 50, 
                    min_new_tokens = 1, max_new_tokens = 512,
                    repetition_penalty = 1,
                    stop_sequences = None):

        parameters = {
            GenParams.DECODING_METHOD: decoding_method,
            GenParams.TEMPERATURE: temperature,
            GenParams.TOP_P: top_p,
            GenParams.TOP_K: top_k,
            GenParams.MIN_NEW_TOKENS: min_new_tokens,
            GenParams.MAX_NEW_TOKENS: max_new_tokens,
            GenParams.REPETITION_PENALTY: repetition_penalty,
        }
        
        if not stop_sequences:
            parameters[GenParams.STOP_SEQUENCES] = stop_sequences

        granite_model = WatsonxLLM(
            model_id = model_id,
            url = WatsonxClient._get_watsonx_url(),
            apikey = WatsonxClient._get_cloud_api_key(),
            project_id = WatsonxClient._get_project_id(),
            params = parameters
        )
        return granite_model    
