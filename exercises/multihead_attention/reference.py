"""Head splitting reference implementation using plain loops."""


def split_heads(vector: list[float], num_heads: int) -> list[list[float]]:
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if len(vector) % num_heads != 0:
        raise ValueError("vector length must be divisible by num_heads")

    head_dim = len(vector) // num_heads
    heads = []
    for head_index in range(num_heads):
        head = []
        start = head_index * head_dim
        for dimension in range(head_dim):
            head.append(vector[start + dimension])
        heads.append(head)
    return heads
