#Assignment 3 Exercise 2
import numpy as np


transition_m = np.array([[0.7, 0.3],
                         [0.3, 0.7]])

observation_true_m = np.array([[0.9, 0], 
                               [0, 0.2]])

init_prob = np.array([0.5, 0.5])

#umbrella true = 1, umbrella false = 0
observations_part1 = [1, 1]
observations_part2 = [1, 1, 0, 1, 1]

def forward_algorithm_filtering(observations, transition_m, observation_true_m, init_prob):
    forward_msgs = []
    normalized_forward_msgs = []
    forward_msgs.append(init_prob)
    observation_false_m = 1 - observation_true_m

    for i in range(len(observations)):
        if observations[i] == 1:
            forward_msgs.append(observation_true_m@transition_m@forward_msgs[i])
        else:
            forward_msgs.append(observation_false_m@transition_m@forward_msgs[i])

        normalized_forward_msgs.append(forward_msgs[-1] / np.sum(forward_msgs[-1]))

    return normalized_forward_msgs

# normalized_forward_msgs_part1 = forward_algorithm_filtering(observations_part1, transition_m, observation_true_m, init_prob)
# print(f"Probability of rain at day 2 (after observations): {normalized_forward_msgs_part1[-1]} \n")

normalized_forward_msgs_part2 = forward_algorithm_filtering(observations_part2, transition_m, observation_true_m, init_prob)
print(f"All normalized forward messages with the 5 days worth of observations:\n {normalized_forward_msgs_part2}\n")