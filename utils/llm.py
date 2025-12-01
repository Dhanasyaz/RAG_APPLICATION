import requests
from config import EURON_API_KEY, LLM_MODEL, MAX_TOKENS, TEMPERATURE

def generate_completion(messages, model=None, max_tokens=None, temperature=None):
    """
    Generate chat completion using Euron API
    
    Args:
        messages (list): List of message dicts with 'role' and 'content'
        model (str, optional): Model to use. Defaults to config value.
        max_tokens (int, optional): Max tokens. Defaults to config value.
        temperature (float, optional): Temperature. Defaults to config value.
        
    Returns:
        str: Generated completion text, or None if failed
    """
    url = "https://api.euron.one/api/v1/euri/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURON_API_KEY}"
    }
    payload = {
        "messages": messages,
        "model": model or LLM_MODEL,
        "max_tokens": max_tokens or MAX_TOKENS,
        "temperature": temperature or TEMPERATURE
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if 'choices' not in data or len(data['choices']) == 0:
            raise ValueError(f"Unexpected API response structure: {data}")
        
        return data['choices'][0]['message']['content']
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except (KeyError, IndexError, ValueError) as e:
        print(f"Failed to parse completion: {e}")
        return None


def create_rag_prompt(context, query):
    """
    Create messages for RAG query
    
    Args:
        context (str): Retrieved context from vector store
        query (str): User's question
        
    Returns:
        list: Messages formatted for LLM
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain the answer, say so clearly."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        }
    ]
    return messages