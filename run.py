""" An example to show how to set up an pommerman game programmatically.  """

import sys
from pathlib import Path

# Adding pommerman directory, for terminal use
directory = str(Path(__file__).parents[1])
sys.path.insert(0, directory)

import pommerman
from pommerman import agents
from my_agent import MyAgent


def main():
    """ Simple function to bootstrap a game.  """

    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        MyAgent(),
        agents.SimpleAgent(),
    ]

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('OneVsOne-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for episode in range(1): #1 or 10
        state = env.reset()
        done = False
        while not done:
            # This renders the game
            env.render() #do_sleep=False

            # This is where we give an action to the environment
            actions = env.act(state)

            # This performs the step and gives back the new information
            state, reward, done, info = env.step(actions)

        print(f"Episode: {episode:2d} finished, result: {'Win' if 0 in info.get('winners', []) else 'Lose'}")
    env.close()


if __name__ == '__main__':
    main()


