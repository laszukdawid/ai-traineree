import gzip
import json
import time

import gymnasium as gym
from gym.utils.play import PlayPlot, play

from ai_traineree.buffers.replay import ReplayBuffer


def buffer_callback(buffer):
    def callback(obs_t, obs_next, action, rew, done, *args, **kwargs):
        buffer.add(**dict(state=obs_t, action=[action], reward=[rew], done=[done]), next_state=obs_next)
        return [
            rew,
        ]

    return callback


buffer = ReplayBuffer(10, 2000)
callback = buffer_callback(buffer)
plotter = PlayPlot(callback, 30 * 5, ["reward"])

env_name = "Breakout-v0"
env = gym.make(env_name)
env.reset()
play(env, fps=20, callback=plotter.callback)

t = []
exp_dump = buffer.dump_buffer(serialize=True)
t.append(time.time())
with gzip.open("buffer.gzip", "wt") as f:
    for exp in exp_dump:
        f.write(json.dumps(exp))
        f.write("\n")
t.append(time.time())
print(f"Writing to gzip took: {t[1]-t[0]} s")
