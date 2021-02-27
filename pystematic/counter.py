
class Counter:
    """Just a simple counter"""

    def __init__(self):
        self._count = 0

    @property
    def count(self):
        return self._count
    
    def step(self):
        """Increments the counter by 1."""
        self._count += 1

    def state_dict(self):
        return {
            "count": self.count
        }
    
    def load_state_dict(self, state):
        self._count = state["count"]
