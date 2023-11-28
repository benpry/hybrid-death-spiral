from agents import Agent, MeetingSequence
import pandas as pd
import numpy as np


def run_simulations(params, n_sims=1000):
    rows = []
    for _ in range(n_sims):
        meeting_sequence = MeetingSequence(**params)
        ns_in_person, ns_remote, utilities = meeting_sequence.hold_meeting_sequence(10)
        for i, (n_in_person, n_remote, utility) in enumerate(
            zip(ns_in_person, ns_remote, utilities)
        ):
            rows.append(
                {
                    "meeting_num": i,
                    "n_in_person": n_in_person,
                    "ns_remote": n_remote,
                    "utility": utility,
                }
            )

    return pd.DataFrame(rows)
