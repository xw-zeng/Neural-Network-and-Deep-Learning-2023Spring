def momentum(g, Weights, prev_Weights, alpha, beta, it):
    """Update weights with momentum."""
    if it != 0:
        Weights[2] -= alpha * g[2] - beta * (Weights[2] - prev_Weights[2])
        Weights[0] -= alpha * g[0] - beta * (Weights[0] - prev_Weights[0])
        if len(Weights[1]) != 0:
            Weights[1] = [Weights[1][h] - alpha * g[1][h] + beta * (Weights[1][h] - prev_Weights[1][h])
                          for h in range(len(Weights[1]))]
    else:
        Weights[2] -= alpha * g[2]
        Weights[0] -= alpha * g[0]
        if len(Weights[1]) != 0:
            Weights[1] = [Weights[1][h] - alpha * g[1][h] for h in range(len(Weights[1]))]
    return Weights
