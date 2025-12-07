## Virtual Environment (the same as Eric)
1. Download `uv`
1. `uv venv`
1. `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
1. `uv pip install -r requirements.txt`
1. Use `source` every time you open a new terminal to work on the project

## How to train
### Two kinds of observation mode: ball + 1rod, or whole
`python train_marl_four_agents.py --obs-mode full --updates 200 --rollout-steps 1024 --curriculum 1 --max-episode-steps 2000`
it will output a folder contains of four rods agent .pt files
### How to evaluate/see visual GUI
`python eval_marl_four_agents.py --run-dir saves/marl_rod_obs-ball_self_seed-0 --obs-mode ball_self --curriculum 2 --episodes 10 --render`
Although the result is still bad (sorry about that QQ, I am a little exhausted)