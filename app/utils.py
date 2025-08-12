def classify_status(param, value):
    """Classify parameter value into safe/caution/danger."""
    thresholds = {
        "pH": [(6.5, 8.5, "safe"), (5.5, 6.5, "caution"), (8.5, 9.5, "caution")],
        "Nitrate": [(0, 10, "safe"), (10, 20, "caution")],
        "Nitrite": [(0, 1, "safe"), (1, 3, "caution")],
        "Chlorine": [(0, 1, "safe"), (1, 2, "caution")],
        "Hardness": [(0, 100, "safe"), (100, 200, "caution")],
        "Carbonate": [(0, 1, "safe"), (1, 3, "caution")]
    }

    for low, high, status in thresholds.get(param, []):
        if low <= value <= high:
            return status
    return "danger"
