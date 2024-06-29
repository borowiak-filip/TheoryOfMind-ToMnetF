from Env import World
import Agent as Agent
import numpy as np

"""
# Code taken from https://github.com/Nik-Kras/ToMnet-N
@Author Nikita Krasnytskyi
@Modified by Filip Borowiak
"""

ROWS = 13
COLS = 13
SIGHT = 3

env = World(row_size=ROWS, col_size=COLS, shuffle=False, no_walls=True)
for i in range(50):
    env.reset()
    env.render()
    agent = Agent.AgentStar(env, SIGHT, observability="full")

    while True:
        agent.update_world_observation()
        agent.render()

        action = agent.chose_action(observability="full")
        #print(action)
        
        observe, terminate, goal_picked, reward = env.execute(action)
        

        if goal_picked:
            #print("You have picked a goal, reward = {}".format(reward))
            agent.on_pickup(reward)
            # added to terminate after picking one goal
            #terminate = True

        if terminate:
            #print("Game result: ", reward)
            break

        #input("Press the <Enter> key to continue...")

    agent.save_game(name="demo")