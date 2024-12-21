#
# Copyright IBM Corp. 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# Author: Nguyen, Hung (Howie) Sy
#

from ibm_watsonx_ai.foundation_models.embeddings import Embeddings
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams, EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM
import os

class WatsonxClient:
    @staticmethod
    def _get_watsonx_url():
        return os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

    @staticmethod
    def  _get_cloud_api_key():
        return os.getenv("IBM_CLOUD_API_KEY")

    @staticmethod
    def  _get_project_id():
        return os.getenv("WATSONX_PROJECT_ID")

    @staticmethod
    def request_llm(model_id="ibm/granite-3-8b-instruct",
                                decoding_method = "greedy", 
                                temperature = 0.7, top_p = 1.0, top_k = 50, 
                                min_new_tokens = 1, max_new_tokens = 820,
                                repetition_penalty = 1,
                                stop_sequences=None):
        
        parameters = { # model parameters
            GenParams.DECODING_METHOD: decoding_method,
            GenParams.TEMPERATURE: temperature,
            GenParams.TOP_P: top_p,
            GenParams.TOP_K: top_k,
            GenParams.MIN_NEW_TOKENS: min_new_tokens,
            GenParams.MAX_NEW_TOKENS: max_new_tokens,
            GenParams.REPETITION_PENALTY: repetition_penalty,
        }

        # A workaround to deal with the error `stop_sequences` found in both the input and default params when using LangChain's ReAct
        if stop_sequences:
             parameters[GenParams.STOP_SEQUENCES]=stop_sequences

        granite_model = WatsonxLLM(
            model_id = model_id,
            url = WatsonxClient._get_watsonx_url(),
            apikey = WatsonxClient._get_cloud_api_key(),
            project_id = WatsonxClient._get_project_id(),
            params = parameters
        )

        return granite_model
    
    @staticmethod
    def request_embedding_model(model_id="ibm/slate-30m-english-rtrvr",
                                truncate_input_tokens=512):
            embed_params = {
                EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: truncate_input_tokens,
                EmbedTextParamsMetaNames.RETURN_OPTIONS: {
                    'input_text': False
                }
            }
            cloud_credentials=Credentials(
                url= WatsonxClient._get_watsonx_url(),
                api_key=WatsonxClient._get_cloud_api_key())
            
            embeddings_model = Embeddings(
                model_id = model_id,
                project_id = WatsonxClient._get_project_id(),
                credentials = cloud_credentials,
                params=embed_params
            )

            return embeddings_model
