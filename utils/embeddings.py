import requests
import numpy as np
from config import EURON_API_KEY

def generate_embeddings(text):
    """
    Generate embeddings using Euron API
    
    Args:
        text (str): Input text to embed
        
    Returns:
        list: Embedding vector as list, or None if failed
    """
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURON_API_KEY}"
    }
    payload = {
        "input": text,
        "model": "text-embedding-3-small"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'data' not in data or len(data['data']) == 0:
            raise ValueError(f"Unexpected API response structure: {data}")
        
        embedding = np.array(data['data'][0]['embedding'])
        return embedding.tolist()
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except (KeyError, IndexError, ValueError) as e:
        print(f"Failed to parse embedding: {e}")
        return None