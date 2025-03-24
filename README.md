# Block Blast Game + Reinforcement learning AI agent
This repository is my implementation of the Block Blast game as well as an AI agent that learns how to play BlockBlast using Reinforcement Learning.

![Screenshot of the BlockBlast gameplay](demo/image.png)

### Credits: The game's visual design & assets have been borrowed from https://github.com/Kefrov/Blast/. Everything else, including the game logic, AI agent, etc. have been implemented by me.

Steps:
- [x] Implement the game logic. Make BlockBlast playable by humans.
- [x] Create the game environment for the AI agent.
- [x] Implement an agent that plays the game randomly.
- [x] Set up an environment for RL agents
- [ ] Implement an agent that plays the game using Reinforcement Learning techniques.

## How to Run

1. Clone the repository
2. Create a virtual environment and install the dependencies
```bash 
python -m venv blockblast
source blockblast/bin/activate #if on mac
pip install -r requirements.txt
```


### Run the human playable version of the game

Running the game in human mode allows you to play it using your keyboard and mouse controls.

To run it, execute the following command:
```bash
python human_play.py
```

Commands:

- E,R,T to select the piece.
- Hover over the board with your mouse to decide where to place the block.
- Left click or click SPACE to place the piece at the selected location.
- If you want to restart the game, press ESC.

### Run the small AI agent demo.

```bash

python ai_play.py
```