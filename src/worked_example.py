"""Regenerate the worked numerical example in paper_notes/03_attention_math.md.

  python -m src.worked_example
"""

import numpy as np

from src.attention_numpy import attention_backward, attention_forward, causal_allowed

# Three tokens, dk = dv = 2; small integers so the steps can be checked by hand.
Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
V = np.array([[1.0, 2.0], [3.0, 0.0], [0.0, 1.0]])
G = np.ones((3, 2))  # dL/dO for the scalar loss L = sum(O)


def show(name, a):
    print(f"{name} =")
    print(np.array2string(a, precision=4, suppress_small=True))


def main():
    np.set_printoptions(floatmode="fixed")
    for label, allowed in (("no mask", None), ("causal mask", causal_allowed(3))):
        print(f"--- {label} ---")
        O, cache = attention_forward(Q, K, V, allowed)
        show("S = Q K^T / sqrt(2)", (Q @ K.T) / np.sqrt(2))
        show("P = softmax(S)", cache["P"])
        show("O = P V", O)
        dQ, dK, dV = attention_backward(G, cache)
        show("dL/dQ for L = sum(O)", dQ)
        show("dL/dV for L = sum(O)", dV)


if __name__ == "__main__":
    main()
