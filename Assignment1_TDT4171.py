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


num_simulations = 1000
starting_coins = 10

mean_plays, median_plays = simulate(num_simulations, starting_coins)

print("Exercise 2c: slot machine")
print(f"Mean: {mean_plays}")
print(f"Median: {median_plays}")

#Exercise 3 Part 1a

