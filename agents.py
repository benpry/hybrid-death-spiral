"""
The code for hybrid meeting agents
"""
import numpy as np


class Agent:
    """
    A hybrid meeting agent that tries to match the majority of others.
    """

    def __init__(
        self,
        utility_from_others_matrix: np.ndarray,
        intrinsic_utilities: np.ndarray,
    ):
        self.intrinsic_utilities = intrinsic_utilities
        self.utility_from_others_matrix = utility_from_others_matrix

    def compute_utilities(self, others_attendance):
        """
        Compute the utilities from attending in each format
        """
        return (
            self.utility_from_others_matrix.dot(others_attendance)
            + self.intrinsic_utilities
        )

    def attends(self, attendance_estimates, meeting_format):
        """
        Return whether the agent attends a meeting in a given format
        """
        utilities = self.compute_utilities(attendance_estimates)
        if meeting_format == "hybrid":
            return (
                "in-person"
                if utilities[0] >= utilities[1] and utilities[0] > 0
                else "remote"
                if utilities[1] > utilities[0] and utilities[1] > 0
                else False
            )
        elif meeting_format == "online":
            return "remote" if utilities[1] > 0 else False
        elif meeting_format == "irl":
            return "in-person" if utilities[0] > 0 else False
        else:
            raise ValueError(f"Invalid meeting format: {meeting_format}")


class MeetingSequence:
    """
    A sequence of meetings
    """

    def __init__(
        self,
        n_agents,
        meeting_format,
        AgentType=Agent,
        intrinsic_utility_means=np.array((0, 0)),
        intrinsic_utility_stds=np.array((1, 1)),
        utility_from_others_means=np.array(((0, 0), (0, 0))),
        utility_from_others_stds=np.array(((1, 1), (1, 1))),
        hybrid_fixed_cost=10,
    ):
        self.agents = []
        for _ in range(n_agents):
            intrinsic_utilities = np.random.normal(
                loc=intrinsic_utility_means, scale=intrinsic_utility_stds
            )
            utility_from_others = np.random.normal(
                loc=utility_from_others_means, scale=utility_from_others_stds
            )
            agent = AgentType(
                utility_from_others_matrix=utility_from_others,
                intrinsic_utilities=intrinsic_utilities,
            )
            self.agents.append(agent)

        self.meeting_format = meeting_format
        self.n_agents = n_agents
        self.hybrid_fixed_cost = hybrid_fixed_cost
        # initialize the expected attendance to everyone
        self.reset()

    def reset(self):
        if self.meeting_format == "online":
            self.remote_estimate = self.n_agents
            self.in_person_estimate = 0
        else:
            self.in_person_estimate = self.n_agents
            self.remote_estimate = 0

    def hold_meeting(self, meeting_format):
        """Hold a meeting and return the total utility"""
        agent_attendances = [
            a.attends(
                np.array([self.in_person_estimate, self.remote_estimate]),
                meeting_format,
            )
            for a in self.agents
        ]
        n_in_person = sum([int(x == "in-person") for x in agent_attendances])
        n_remote = sum([int(x == "remote") for x in agent_attendances])

        sum_utility = 0
        for i, agent in enumerate(self.agents):
            if agent_attendances[i]:
                if agent_attendances[i] == "in-person":
                    sum_utility += agent.compute_utilities(
                        np.array([n_in_person, n_remote])
                    )[0]
                elif agent_attendances[i] == "remote":
                    sum_utility += agent.compute_utilities(
                        np.array([n_in_person, n_remote])
                    )[1]
                else:
                    raise ValueError(f"Invalid attendance type: {agent_attendances[i]}")

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


class LogUtilityAgent(Agent):
    """
    A hybrid meeting agent that tries to match the majority of others.
    """

    def compute_utility(self, Np, Nr, chose_in_person):
        if chose_in_person:
            return self.sp * np.log(Np)
        else:
            return self.B + self.sr * np.log(Nr)
