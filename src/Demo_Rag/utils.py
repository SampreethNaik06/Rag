import os



import json
import numpy as np
import pandas as pd
from pprint import pprint as original_pprint
from dateutil import parser
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os 
from openai import OpenAI

model_name = os.path.join(os.environ.get('MODEL_PATH', './'), "BAAI/bge-base-en-v1.5")

model = SentenceTransformer("BAAI/bge-base-en-v1.5", cache_folder=os.environ.get('MODEL_PATH', './'))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.joblib") if not os.path.exists("embeddings.joblib") else "embeddings.joblib"

EMBEDDINGS = joblib.load(EMBEDDINGS_PATH)

def get_nvidia_key():
    """
    Get the NVIDIA API key from environment variables or a local .env file.
    """
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if api_key:
        return api_key

    # Check for .env file in project directories
    search_paths = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(BASE_DIR, ".env"),
        os.path.join(BASE_DIR, "..", "..", ".env"),
    ]
    for path in search_paths:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("NVIDIA_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip("'\"")
                        if val:
                            os.environ["NVIDIA_API_KEY"] = val
                            return val
    return ""

def pprint(*args, **kwargs):
    print(json.dumps(*args, indent=2))

def format_date(date_string):
    # Parse the input string into a datetime object
    date_object = parser.parse(date_string)
    # Format the date to "YYYY-MM-DD"
    formatted_date = date_object.strftime("%Y-%m-%d")
    return formatted_date

# Read the CSV without parsing dates

def read_dataframe(path):
    if not os.path.exists(path):
        candidate = os.path.join(BASE_DIR, path)
        if os.path.exists(candidate):
            path = candidate
    df = pd.read_csv(path)

    # Apply the custom date formatting function to the relevant columns
    df['published_at'] = df['published_at'].apply(format_date)
    df['updated_at'] = df['updated_at'].apply(format_date)

    # Convert the DataFrame to dictionary after formatting
    df = df.to_dict(orient='records')
    return df


def generate_with_single_input(prompt: str,
                               role: str = 'user',
                               top_p: float = None,
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str = "meta/llama-3.2-11b-vision-instruct",
                               nvidia_api_key: str = None,
                               **kwargs):

    if nvidia_api_key is None:
        nvidia_api_key = get_nvidia_key()

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=nvidia_api_key
    )

    payload = {
        "model": model,
        "messages": [{'role': role, 'content': prompt}],
        "max_tokens": max_tokens,
        **kwargs
    }

    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    try:
        response = client.chat.completions.create(**payload)
        output_dict = {
            'role': response.choices[0].message.role,
            'content': response.choices[0].message.content
        }
    except Exception as e:
        raise Exception(f"Failed to get correct output from NVIDIA LLM call.\nException: {e}")

    return output_dict


def concatenate_fields(dataset, fields):
    # Initialize the list where the texts will be stored    
    concatenated_data = [] 

    # Iterate over movies
    for data in dataset:
        # Initialize text as an empty string
        text = "" 

        # Iterate over the fields
        for field in fields: 
            # Get the desired field (if the key is missing an empty string should be used)
            context = data.get(field, '') 

            if context:
                # Add the context to the text (add an extra space so fields are separate)
                text += f"{context} " 

        # Strip whitespaces from the text
        text = text.strip()[:493]
        # Append the text with extra context to the list
        concatenated_data.append(text) 
    
    return concatenated_data


NEWS_DATA_PATH = os.path.join(BASE_DIR, "news_data_dedup.csv") if not os.path.exists("./news_data_dedup.csv") else "./news_data_dedup.csv"
NEWS_DATA = pd.read_csv(NEWS_DATA_PATH).to_dict(orient='records')


def retrieve(query, top_k=5):
    query_embedding = model.encode(query)

    similarity_scores = cosine_similarity(query_embedding.reshape(1, -1), EMBEDDINGS)[0]
    
    similarity_indices = np.argsort(-similarity_scores)

    top_k_indices = similarity_indices[:top_k]

    return top_k_indices

