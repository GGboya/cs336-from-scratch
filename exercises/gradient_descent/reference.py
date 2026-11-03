"""Reference implementation of one-dimensional gradient descent."""


def gradient_descent(initial_w: float, learning_rate: float, steps: int) -> float:
    w = initial_w
    for _ in range(steps):
        gradient = 2.0 * (w - 3.0)
        w = w - learning_rate * gradient
    return w
