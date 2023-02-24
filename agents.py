"""
The code for hybrid meeting agents
"""
import numpy as np

class Agent:
    """
    A hybrid meeting agent
    """
    def __init__(self, s, B):
        self.s = s
        self.B = B

    def comes_in_person(self, Np_estimate):
        """
        Decide whether to come to a meeting
        """
        return self.s * Np_estimate > self.B

    def compute_utility(self, Np, chose_in_person):
        if chose_in_person:
            return self.s * Np
        else:
            return self.B

class MeetingSequence:
    """
    A sequence of meetings
    """
    def __init__(self, n_agents, B_params=(10, 5), s_params=(1, 0.5)):
        self.agents = []
        for _ in range(n_agents):
            s = np.random.normal(loc=s_params[0], scale=s_params[1])
            B = np.random.normal(loc=B_params[0], scale=B_params[1])
            agent = Agent(s, B)
            self.agents.append(agent)
        
        self.n_agents = n_agents
        # initialize the expected attendance to everyone
        self.attendance_estimate = self.n_agents

    def hold_meeting(self):
        """Hold a meeting and return the total utility"""
        agents_in_person = [a.comes_in_person(self.attendance_estimate) for a in self.agents]
        n_in_person = sum([int(x) for x in agents_in_person])

        sum_utility = 0
        for i, agent in enumerate(self.agents):
            sum_utility += agent.compute_utility(n_in_person, agents_in_person[i])
        
        # this iteration's actual number becomes the next iteration's estimate
        self.attendance_estimate = n_in_person

        return n_in_person, sum_utility

    def hold_meeting_sequence(self, n):
        """Hold a sequence of n meetings and return the statistics"""
        ns_in_person = []
        utilities = []
        for _ in range(n):
            n_in_person, utility = self.hold_meeting()
            ns_in_person.append(n_in_person)
            utilities.append(utility)

        return ns_in_person, utilities
            
    def reset(self):
        self.attendance_estimate = self.n_agents


