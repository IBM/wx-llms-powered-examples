#
# Copyright IBM Corp. 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# Author: Nguyen, Hung (Howie) Sy
#

import unittest
import requests
import time
import signal
import os

class TestText2VecWatsonx(unittest.TestCase):
    EMBEDDING_API_URL = "http://localhost:5000"
    
    @classmethod
    def setUpClass(cls):
        signal.signal(signal.SIGINT, cls.signal_handler)

    def signal_handler(signal, frame):
        print('Exited as Ctrl+C pressed')
        os._exit(0)

    def test_weavite_text2vec_watsonx(self):
        print("\nWaiting for the API ready (Ctrl+C to quit if needed)..", end='', flush=True)
        for i in range(15):
            try:
                print(".", end='', flush=True)
                response = requests.get(f"{self.EMBEDDING_API_URL}/.well-known/ready")
                if response.status_code == 204:
                    print("\nIt's ready!")
                    break
                else:
                    time.sleep(5)
            except:
                time.sleep(5)
        
        try:
            input = {"text": "Hello!"}

            VECTOR_API_URL = f"{self.EMBEDDING_API_URL}/vectors"
            print(f"\nRun Test: querying the API {VECTOR_API_URL} ...")
            response = requests.post(VECTOR_API_URL, json=input)

            status_code = response.status_code
            output = response.json()
            text = output["text"]
            embedding = output["vector"]
            
            print("\nTest result: ")
            self.assertEqual(status_code, 200, "Status code 200")
            self.assertEqual(text, input["text"])
            self.assertIsNotNone(embedding, f"The embedding returned: {embedding}")

        except Exception as error:
            self.fail("An exception was raised: ", error)

if __name__ == '__main__':
    unittest.main()