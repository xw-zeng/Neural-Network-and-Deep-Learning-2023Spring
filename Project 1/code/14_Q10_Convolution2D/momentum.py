def momentum(g, Weights, prev_Weights, alpha, beta, it):
    """Update weights with momentum."""
    if it != 0:
        Weights[3] -= alpha * g[3] - beta * (Weights[3] - prev_Weights[3])
        Weights[1] -= alpha * g[1] - beta * (Weights[1] - prev_Weights[1])
        if len(Weights[2]) != 0:
            Weights[2] = [Weights[2][h] - alpha * g[2][h] + beta * (Weights[2][h] - prev_Weights[2][h])
                          for h in range(len(Weights[2]))]
            Weights[0] = [Weights[0][h] - alpha * g[0][h] + beta * (Weights[0][h] - prev_Weights[0][h])
                          for h in range(len(Weights[0]))]
    else:
        Weights[3] -= alpha * g[3]
        Weights[1] -= alpha * g[1]
        if len(Weights[2]) != 0:
            Weights[2] = [Weights[2][h] - alpha * g[2][h] for h in range(len(Weights[2]))]
        Weights[0] = [Weights[0][h] - alpha * g[0][h] for h in range(len(Weights[0]))]
    return Weights
