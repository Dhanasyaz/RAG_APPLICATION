from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text, chunk_size=None, chunk_overlap=None):
    """
    Split text into overlapping chunks
    
    Args:
        text (str): Text to split
        chunk_size (int, optional): Size of each chunk. Defaults to config value.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to config value.
        
    Returns:
        list: List of text chunks
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    
    # Split by sentences first (rough split)
    sentences = text.replace('\n', ' ').split('. ')
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Add period back if it was removed
        if not sentence.endswith('.'):
            sentence += '.'
        
        # If adding this sentence exceeds chunk_size, save current chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous chunk
            overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks