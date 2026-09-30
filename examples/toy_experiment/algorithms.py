def algorithm_a(n):
    total = 0
    additions = 0
    for i in range(1, n + 1):
        total += i
        additions += 1
    return total, additions


def algorithm_b(n):
    # Closed-form sum: one addition, one multiplication and one integer division.
    return n * (n + 1) // 2, 1
