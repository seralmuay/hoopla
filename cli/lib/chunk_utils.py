import re

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    """
    Split text into chunks of specified word count.
    
    Args:
        text (str): The text to be chunked
        chunk_size (int): Maximum number of words per chunk (default: 200)
        
    Returns:
        list[str]: List of text chunks
    """
    # Normalize text by replacing newlines with spaces and removing leading/trailing whitespace
    normalized_text = text.replace('\n', ' ').strip()
    total_characters = len(normalized_text)
    word_list = normalized_text.split(' ')
    
    # Initialize tracking variables
    words_in_current_chunk = 0
    current_chunk_text = ""
    text_chunks = []
    
    # Process each word and build chunks
    for word in word_list:
        if words_in_current_chunk < chunk_size:
            # Add word to current chunk
            current_chunk_text += word + " "
            words_in_current_chunk += 1
        else:
            # Current chunk is full, save it and start a new one
            text_chunks.append(current_chunk_text.strip())
            current_chunk_text = word + " "
            words_in_current_chunk = 1  # Start with 1 since we already added the current word
    
    # Add the final chunk if it contains any content
    _add_final_chunk_if_not_empty(current_chunk_text, text_chunks)

    if overlap > 0:
        text_chunks = _overlap_chunks(text_chunks, overlap)
    
    # Display chunking information
    _display_chunking_info(total_characters, text_chunks)
    
    return text_chunks


def _add_final_chunk_if_not_empty(current_chunk: str, chunks: list[str]) -> None:
    """
    Add the final chunk to the list if it contains content.
    
    Args:
        current_chunk (str): The current chunk text that may need to be added
        chunks (list[str]): The list of chunks to add to
    """
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

def _overlap_chunks(chunks: list[str], overlap: int) -> list[str]:
    """
    Apply overlap between consecutive chunks by taking the last 'overlap' words
    from each chunk and adding them to the beginning of the next chunk.
    
    Args:
        chunks (list[str]): List of text chunks
        overlap (int): Number of words to overlap between chunks
        
    Returns:
        list[str]: List of text chunks with overlap applied
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    
    # Create a new list to avoid modifying the original during iteration
    overlapped_chunks = [chunks[0]]  # First chunk remains unchanged
    
    for i in range(1, len(chunks)):
        prev_chunk = chunks[i-1]
        current_chunk = chunks[i]
        
        # Get the last 'overlap' words from the previous chunk
        prev_words = prev_chunk.split()
        overlap_words = prev_words[-overlap:] if len(prev_words) >= overlap else prev_words
        overlap_text = ' '.join(overlap_words)
        
        # Combine overlap words with current chunk
        overlapped_chunk = overlap_text + ' ' + current_chunk
        overlapped_chunks.append(overlapped_chunk)
    
    return overlapped_chunks



def _display_chunking_info(total_chars: int, chunks: list[str]) -> None:
    """
    Display information about the chunking process.
    
    Args:
        total_chars (int): Total number of characters in the original text
        chunks (list[str]): The list of created chunks
    """
    print(f"Chunking {total_chars} characters into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  {i+1}. {chunk}")


def semantic_chunk(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    """
    Split text into chunks based on sentence boundaries to preserve meaning.
    
    Args:
        text (str): The text to be chunked
        max_chunk_size (int): Maximum number of sentences per chunk (default: 4)
        overlap (int): Number of overlapping sentences between chunks (default: 0)
        
    Returns:
        list[str]: List of text chunks
    """
    # Normalize text by replacing newlines with spaces and removing leading/trailing whitespace
    normalized_text = text.replace('\n', ' ').strip()
    total_characters = len(normalized_text)
    
    # Split text into sentences using regex
    sentences = re.split(r"(?<=[.!?])\s+", normalized_text)
    
    # Filter out empty strings that might result from the split
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        print(f"Semantically chunking {total_characters} characters")
        print("  No sentences found in the text.")
        return []
    
    # Create chunks with specified number of sentences
    text_chunks = []
    i = 0
    
    while i < len(sentences):
        # Calculate the end index for this chunk
        end_idx = min(i + max_chunk_size, len(sentences))
        chunk = " ".join(sentences[i:end_idx])
        text_chunks.append(chunk)
        
        # Move to the next chunk, accounting for overlap
        i += max_chunk_size - overlap
        
        # Ensure we always make progress and don't go backward
        if overlap >= max_chunk_size:
            i = end_idx
        
        # If we've already processed all sentences, break to avoid redundant chunks
        if end_idx >= len(sentences):
            break
    
    # Display chunking information
    print(f"Semantically chunking {total_characters} characters")
    for i, chunk in enumerate(text_chunks):
        print(f"{i+1}. {chunk}")
    
    return text_chunks