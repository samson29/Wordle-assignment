"""Public-history closing labels; never accepts an actual game secret."""
def closing_target(decision, engine):
    remaining = engine.state(decision.history).answers
    if len(remaining) != 1:
        return None
    word = engine.answers[remaining[0]]
    return decision.words.index(word) if word in decision.words else None
