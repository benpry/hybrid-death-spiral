from agents import Agent, MeetingSequence
import pandas as pd
import numpy as np

def run_simulations(params, n_sims=1000):
    rows = []
    for simulation in range(n_sims):
        meeting_sequence = MeetingSequence(**params)
        attendances, utilities = meeting_sequence.hold_meeting_sequence(10)
        for i, (attendance, utility) in enumerate(zip(attendances, utilities)):
            rows.append({"meeting_num": i, "attendance": attendance, "utility": utility})

    return pd.DataFrame(rows)

