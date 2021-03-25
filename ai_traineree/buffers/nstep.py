from . import BufferBase, Experience


class NStepBuffer(BufferBase):

    type = "NStep"

    def __init__(self, n_steps: int, gamma: float):
        super().__init__()
        self.gamma = gamma
        self.n_steps = n_steps
        self.n_gammas = [gamma**i for i in range(1, n_steps+1)]

        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    @property
    def available(self):
        return len(self.buffer) >= self.n_steps

    def add(self, **kwargs):
        self.buffer.append(Experience(**kwargs))

    def get(self) -> Experience:
        current_exp = self.buffer.pop(0)

        for (idx, exp) in enumerate(self.buffer):
            if exp.done[0]:
                break
            current_exp.reward[0] += self.n_gammas[idx]*exp.reward[0]
        return current_exp

    @property
    def all_data(self):
        return self.buffer
