"""
The code for hybrid meeting agents
"""
import numpy as np

class Agent:
    """
    A hybrid meeting agent that tries to match the majority of others.
    """

    def __init__(self, sp, sr, B):
        self.sp = sp
        self.sr = sr
        self.B = B

    def comes_in_person(self, Np_estimate, Nr_estimate):
        return self.compute_utility(Np_estimate, Nr_estimate, True) > \
            self.compute_utility(Np_estimate, Nr_estimate, False)
    
    def compute_utility(self, Np, Nr, chose_in_person):
        if chose_in_person:
            return self.sp * Np
        else:
            return self.B + self.sr * Nr

class MeetingSequence:
    """
    A sequence of meetings
    """
    def __init__(self, n_agents, AgentType=Agent, B_params=(10, 5), sp_params=(1, 0.5), sr_params=(0.2, 0.1)):
        self.agents = []
        for _ in range(n_agents):
            sp = np.random.normal(loc=sp_params[0], scale=sp_params[1])
            sr = np.random.normal(loc=sr_params[0], scale=sr_params[1])
            B = np.random.normal(loc=B_params[0], scale=B_params[1])
            agent = AgentType(sp, sr, B)
            self.agents.append(agent)
        
        self.n_agents = n_agents
        # initialize the expected attendance to everyone
        self.reset()

    def hold_meeting(self):
        """Hold a meeting and return the total utility"""
        agents_in_person = [a.comes_in_person(self.in_person_estimate, self.remote_estimate) for a in self.agents]
        n_in_person = sum([int(x) for x in agents_in_person])
        n_remote = sum([1 - int(x) for x in agents_in_person])

        sum_utility = 0
        for i, agent in enumerate(self.agents):
            sum_utility += agent.compute_utility(n_in_person, n_remote, agents_in_person[i])
        
        # this iteration's actual number becomes the next iteration's estimate
        self.in_person_estimate = n_in_person
        self.remote_estimate = n_remote

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
        self.in_person_estimate = self.n_agents
        self.remote_estimate = 0


class LogUtilityAgent(Agent):
    """
    A hybrid meeting agent that tries to match the majority of others.
    """
    
    def compute_utility(self, Np, Nr, chose_in_person):
        if chose_in_person:
            return self.sp * np.log(Np)
        else:
            return self.B + self.sr * np.log(Nr)