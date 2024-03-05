import random
import numpy as np 

#Exersice 2c)

def slot_machine(coins):
    wheels = ["BAR", "BELL", "LEMON", "CHERRY"]
    payout_scheme={
        ("BAR", "BAR", "BAR"): 20,
        ("LEMON", "LEMON", "LEMON"): 5,
        ("CHERRY", "CHERRY", "CHERRY"): 3,
        ("CHERRY", "CHERRY", "?"): 2,
        ("CHERRY", "?", "?"): 1,
        
    }

    plays = 0
    while coins > 0:
        #Pay
        coins -= 1
        #Spin
        wheel1 = np.random.choice(wheels)
        wheel2 = np.random.choice(wheels)
        wheel3 = np.random.choice(wheels)
        #Payout
        if wheel1=="BAR" and wheel2=="BAR" and wheel3=="BAR":
            coins = coins + 20
        elif wheel1=="BELL" and wheel2=="BELL" and wheel3=="BELL":
            coins = coins + 15
        elif wheel1=="LEMON" and wheel2=="LEMON" and wheel3=="LEMON":
            coins = coins + 5
        elif wheel1=="CHERRY" and wheel2=="CHERRY" and wheel3=="CHERRY":
            coins = coins + 3
        elif wheel1=="CHERRY" and wheel2=="CHERRY":
            coins = coins + 2
        elif wheel1=="CHERRY":
            coins = coins + 1
        else:
            coins = coins + 0
        
        plays = plays + 1

    return plays
    
def simulate(num_simulations, starting_coins):
    plays_list = [slot_machine(starting_coins) for i in range(num_simulations)]
    mean_plays = np.mean(plays_list)
    median_plays = np.median(plays_list)
    return mean_plays, median_plays


num_simulations = 10000
starting_coins = 10
mean_plays, median_plays = simulate(num_simulations, starting_coins)

print("Exercise 2c: slot machine")
print(f"Mean: {mean_plays}")
print(f"Median: {median_plays}\n")

#Exercise 3 Part 1 a 

def probability_of_birthday_overlapp(N, sim_num =10000):
    overlapp_count = 0
    
    for i in range(sim_num):
        birthdays = [random.randint(1,365) for j in range(N)]

        if len(birthdays) != len(set(birthdays)):
            overlapp_count = overlapp_count + 1
    
    probability = overlapp_count/sim_num
    return probability

N=10
probability = probability_of_birthday_overlapp(N)

print("Exercise 3 part 1 a: probability_of_birthday_overlapp")
print(f"Choosen N:{N}, Chance: {probability}\n")

#Exercise 3 part 1 b

def find_smallest_N_where_probability_is_50():
    smallest_N = np.infty
    for N in range(10,50):
        probability = probability_of_birthday_overlapp(N)
        if probability>=0.5 and N < smallest_N:
            smallest_N = N

    return smallest_N

smallest_N = find_smallest_N_where_probability_is_50()

print("Exercise 3 part 1 b: find_smallest_N_where_probability_is_50")
print(f"smallest N: {smallest_N}\n")

#Exercise part 2 a

def find_N_needed_for_birthday_every_day():
    birthdays_covered = set()
    group_size = 0

    while len(birthdays_covered) < 365:
        birthdays_covered.add(random.randint(1, 365))
        group_size = group_size + 1
    
    return group_size

sim_num = 10000
total_group_size = sum(find_N_needed_for_birthday_every_day() for i in range(sim_num))      
average_group_size = total_group_size / sim_num

print("Exercise 3 part 2 a: find_N_needed_for_birthday_every_day")
print(f"Average simulated group_size needed: {average_group_size}\n")
                    




