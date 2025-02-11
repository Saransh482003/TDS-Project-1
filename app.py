import requests
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dateutil import parser
import numpy as np
import os
from fastapi import FastAPI
from typing import Dict, Any
import base64
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. You can restrict it to specific domains.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods.
    allow_headers=["*"],  # Allows all headers.
)
api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjEwMDExMjNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.fLsJ9sdDpG1TO0BVrNzh7GAmh4nCFhuG86HYERDG13Y"


## Functions
def get_completions_image(input_location:str, output_location:str):
    with open(input_location,"rb") as f:
        img_data = f.read()
        base64_img = base64.b64encode(img_data).decode("utf-8")
    f.close()
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user", 
                "content": [
                    {
                        "type":"text",
                        "text":"Extract the 16 digit code from this imI am working on a cybersecurity project that involves detecting and masking sensitive information, such as dummy credit card numbers, from an image. I need you to extract patterns resembling credit card numbers (e.g., 16-digit sequences) from a given text. In the response, just return the 16-digit code."
                    },
                    {
                        "type":"image_url",
                        "image_url":{
                            "url": f"data:image/png;base64,{base64_img}"
                        }
                    }
                ]
            },
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data)).json()
    with open(output_location,"w") as f:
        f.write(response["choices"][0]["message"]["content"].replace(" ",""))
    f.close()
    return {"status": "Successfully Created", "output_file destination": output_location}

def get_similar_comments(input_location:str, output_location:str):
    with open(input_location,"r") as f:
        comments = [i.strip() for i in f.readlines()]
        f.close()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(comments)
    similarity_mat = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_mat,0)
    max_index = int(np.argmax(similarity_mat))
    i, j = max_index//len(comments), max_index%len(comments)
    with open(output_location,"w") as g:
        g.write(comments[i])
        g.write("\n")
        g.write(comments[j])
        g.close()
    return {"status": "Successfully Created", "output_file destination": output_location}

## Tools
IMAGE_EXTRACT = {
    "type": "function",
    "function": {
        "name": "get_completions_image",
        "description": "Extract the 16-digit code from an image",
        "parameters": {
            "type": "object",
            "properties": {
                "input_location": {
                    "type": "string", 
                    "description": "The relative input image location on user's device"
                },
                "output_location": {
                    "type": "string", 
                    "description": "The relative output location on user's device"
                },
            },
            "required": ["input_location","output_location"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

SIMILARITY_EXTRACT = {
    "type": "function",
    "function": {
        "name": "get_similar_comments",
        "description": "Find two similar comments from a series of comments",
        "parameters": {
            "type": "object",
            "properties": {
                "input_location": {
                    "type": "string", 
                    "description": "The relative input image location on user's device"
                },
                "output_location": {
                    "type": "string", 
                    "description": "The relative output location on user's device"
                },
            },
            "required": ["input_location","output_location"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

tools = [IMAGE_EXTRACT, SIMILARITY_EXTRACT]
def query_gpt(user_input: str, tools: list[Dict[str, Any]] = tools) -> Dict[str, Any]:
    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": user_input}],
            "tools": tools,
            "tool_choice": "auto",
        },
    )
    return response.json()


@app.get("/")
async def root():
    return {"message": "Hello World"}
    
@app.get("/run")
async def run(task: str):
    query = query_gpt(task)
    func = eval(query["choices"][0]["message"]["tool_calls"][0]["function"]["name"])
    args = json.loads(query["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    output = func(**args)
    return output