"""The one data shape every reader produces and the decoder consumes.

Coordinates are metres with the origin at the centre spot, x along the pitch length and y across
it. Each half is decoded on its own, so a match is a list of halves.
"""

from dataclasses import dataclass, field

import numpy as np

HOME, AWAY = 0, 1


@dataclass
class Half:
    ball: np.ndarray          # (T, 2) metres, NaN where the ball is not observed
    players: np.ndarray       # (T, P, 2) metres, NaN where a slot is empty
    team: np.ndarray          # (P,) HOME or AWAY for each player slot
    start: int = 0            # first frame of this half on the output timeline
    labels: np.ndarray = None  # (T,) reference codes 0=H 1=A 2=D, when known
    restarts: list = field(default_factory=list)  # [(frame_in_half, team or -1)] from tags

    @property
    def n(self):
        return len(self.ball)


@dataclass
class Match:
    name: str
    pitch: tuple              # (length, width) in metres
    halves: list
