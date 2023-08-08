"""
The code for hybrid meeting agents
"""
import numpy as np

class Agent:
    """
    A hybrid meeting agent that tries to match the majority of others.
    """
    def __init__(self, sp, sr, Bp, Br):
        self.sp = sp
        self.sr = sr
        self.Bp = Bp
        self.Br = Br

    def attends(self, Np_estimate, Nr_estimate, meeting_format):
        if meeting_format == "hybrid":
            # hybrid meeting: do we prefer in-person attendance?
            if self.compute_utility(Np_estimate, Nr_estimate, "in-person") > \
                self.compute_utility(Np_estimate, Nr_estimate, "remote") and \
                self.compute_utility(Np_estimate, Nr_estimate, "in-person") > 0:
                return "in-person"
            # if not, would we rather attend remotely than not at all?
            elif self.compute_utility(Np_estimate, Nr_estimate, "remote") > 0:
                return "remote"
            else:
                return None
        elif meeting_format == "irl":
            # we can either attend in-person or not at all
            return "in-person" if self.compute_utility(Np_estimate, Nr_estimate, "in-person") > 0 else None
        elif meeting_format == "online":
            # we can either attend remotely or not at all
            return "remote" if self.compute_utility(Np_estimate, Nr_estimate, "remote") > 0 else None

    def compute_utility(self, Np, Nr, attendance_type):
        if attendance_type == "in-person":
            return self.Bp + self.sp * Np
        elif attendance_type == "remote":
            return self.Br + self.sr * Nr
        else:
            return 0

class MeetingSequence:
    """
    A sequence of meetings
    """
    def __init__(
            self,
            n_agents,
            meeting_format,
            AgentType=Agent,
            Bp_params=(-10, 5),
            Br_params=(-2, 1),
            sp_params=(1, 0.5),
            sr_params=(0.2, 0.1),
            hybrid_fixed_cost = 10
    ):
        self.agents = []
        for _ in range(n_agents):
            sp = np.random.normal(loc=sp_params[0], scale=sp_params[1])
            sr = np.random.normal(loc=sr_params[0], scale=sr_params[1])
            Bp = np.random.normal(loc=Bp_params[0], scale=Bp_params[1])
            Br = np.random.normal(loc=Br_params[0], scale=Br_params[1])
            agent = AgentType(sp, sr, Bp, Br)
            self.agents.append(agent)
        
        self.meeting_format = meeting_format
        self.n_agents = n_agents
        self.hybrid_fixed_cost = hybrid_fixed_cost
        # initialize the expected attendance to everyone
        self.reset()

    def hold_meeting(self, meeting_format):
        """Hold a meeting and return the total utility"""
        agent_attendances = [a.attends(self.in_person_estimate, self.remote_estimate, meeting_format) for a in self.agents]
        n_in_person = sum([int(x == "in-person") for x in agent_attendances])
        n_remote = sum([int(x == "remote") for x in agent_attendances])

        sum_utility = 0
        for i, agent in enumerate(self.agents):
            sum_utility += agent.compute_utility(n_in_person, n_remote, agent_attendances[i])
        mean_utility = sum_utility / self.n_agents
        
        # this iteration's actual number becomes the next iteration's estimate
        self.in_person_estimate = n_in_person
        self.remote_estimate = n_remote

        return n_in_person, n_remote, mean_utility

    def hold_meeting_sequence(self, n):
        """Hold a sequence of n meetings and return the statistics"""
        ns_in_person = []
        ns_remote = []
        utilities = []
        for _ in range(n):
            n_in_person, n_remote, utility = self.hold_meeting(self.meeting_format)
            if self.meeting_format == "hybrid":
                utility -= self.hybrid_fixed_cost
            ns_in_person.append(n_in_person)
            ns_remote.append(n_remote)
            utilities.append(utility)

        return ns_in_person, ns_remote, utilities
            
    def reset(self):
        if self.meeting_format == "online":
            self.remote_estimate = self.n_agents
            self.in_person_estimate = 0
        else:
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
