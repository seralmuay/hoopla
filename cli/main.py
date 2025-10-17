

def add_vectors(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")
    
    return [a + b for a, b in zip(vector1, vector2)]

def substract_vectors(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")
    
    return [a - b for a, b in zip(vector1, vector2)]

def dot_product(vec1,vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")
    return sum(a*b for a,b in zip(vec1,vec2))
    
def euclidean_norm(vec):
    total = 0.0
    for x in vec:
        total += x**2

    return total**0.5

def cosine_similarity(vec1,vec2):
    dot_prod = dot_product(vec1,vec2)
    norm1 = euclidean_norm(vec1)
    norm2 = euclidean_norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_prod / (norm1 * norm2)

def main() -> None: 
    vec1 = [0,0,0]
    vec2 = [1,2,3]

    print("cosine similarity:", cosine_similarity(vec1,vec2))
            
if __name__ == "__main__":
    main()