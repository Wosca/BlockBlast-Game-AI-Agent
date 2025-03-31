# Block Blast Game + Reinforcement learning AI agent
This repository is my implementation of the Block Blast game as well as an AI agent that learns how to play BlockBlast using Reinforcement Learning.

![Screenshot of the BlockBlast gameplay](demo/image.png)

### Credits: The game's visual design & assets have been borrowed from https://github.com/Kefrov/Blast/. Everything else, including the game logic, AI agent, etc. have been implemented by me.

Steps:
- [x] Implement the game logic. Make BlockBlast playable by humans.
- [x] Create the game environment for the AI agent.
- [x] Implement an agent that plays the game randomly.
- [x] Set up an environment for RL agents
- [x] Train an gent using PPO
- [x] Train an agent using DQN
- [ ] Create infrastructure for comparing the effectiveness of different agents.

## How to Run

1. Clone the repository
2. Create a virtual environment and install the dependencies
```bash 
python -m venv blockblast
source blockblast/bin/activate # if on mac
blockblast\Scripts\activate # if on windows
```
pip install -r requirements.txt
```


### Run the human playable version of the game

Running the game in human mode allows you to play it using your keyboard and mouse controls.

To run it, execute the following command:
```bash
python human_play.py
```

Commands:

- *E,R,T* to select the piece.
- Hover over the board with your mouse to decide where to place the block.
- Left click or click SPACE to place the piece at the selected location.
- If you want to restart the game, press ESC.

### Train or visualize PPO agent (with or without masking)

in `ppo_agent.py` you can modify the following lines as well as any other hyperparameters or setting that you want
```
    train_ppo_without_masking = True
    train_ppo_with_masking = False
    visualize_ppo_without_masking = False
    visualize_ppo_with_masking = False

```

you can then run
```bash
python ppo_agent.py
```
and depending on your choice, the training for ppo_agent will start or will have the trained model from models runnning.



### Train or visualize DQN agent

in `dqn_agent.py` you can modify the following lines as well as any other hyperparameters or setting that you want
```
    train_dqn_agent = False
    visualize_dqn_agent = True
```

you can then run
```bash
python dqn_agent.py
```
and depending on your choice, the training for dqn_agent will start or will have the trained model from models runnning.




