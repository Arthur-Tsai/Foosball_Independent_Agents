# eval_marl_four_agents.py

import argparse
import os
import time
import numpy as np
import torch

from foosball_env_independent import FoosballEnv
from marl_single_rod_policy import SingleRodActorCritic


def load_agents(hidden_dim, action_dim, device, rod_paths=None, prefix=None):
    """
    Load 4 SingleRodActorCritic agents.

    We INFER obs_dim from each checkpoint's fc1.weight shape, so we never
    get shape-mismatch errors again.

    Priority:
    1) If rod_paths (list of 4 paths) is given, use them.
    2) Else, use prefix + '0.pt'...'3.pt'.
    """
    if rod_paths is None:
        if prefix is None:
            raise ValueError("Either --rod-paths or --prefix/--run-dir must be provided.")
        rod_paths = [f"{prefix}{i}.pt" for i in range(4)]

    agents = []
    for i in range(4):
        path = rod_paths[i]
        state_dict = torch.load(path, map_location=device)

        # infer obs_dim from fc1.weight shape
        in_dim = state_dict["fc1.weight"].shape[1]
        print(f"[INFO] Rod {i}: checkpoint input dim = {in_dim}")

        agent = SingleRodActorCritic(
            obs_dim=in_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        ).to(device)

        # remember what this agent expects as input
        agent.obs_dim = in_dim

        agent.load_state_dict(state_dict)
        agent.eval()
        print(f"[INFO] Loaded rod {i} agent from: {path}")
        agents.append(agent)

    return agents


def run_episodes(
    env,
    agents,
    device,
    num_episodes,
    max_steps,
    trained_obs_mode="full",
    render=False,
    sleep_time=1.0 / 240.0,
):
    """
    Run evaluation episodes.
    Red side: 4 MARL agents (rod 0..3).
    Blue side: simple bot logic inside FoosballEnv (since opponent_action=None).

    trained_obs_mode:
        "full"      -> each agent receives truncated global env obs
                       (first obs_dim entries matching its checkpoint).
        "ball_self" -> each agent receives its own 8-dim [ball+self] obs
                       via env.get_per_agent_ball_self_obs().

    Returns: list of episode returns.
    """
    episode_returns = []
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        obs = obs.astype(np.float32)
        done = False
        ep_return = 0.0
        step_count = 0

        while not done and step_count < max_steps:
            step_count += 1

            # ---------- choose per-agent observations ----------
            if trained_obs_mode == "full":
                # each agent gets the first obs_dim entries it was trained with
                per_agent_obs_list = []
                for i in range(4):
                    d_i = agents[i].obs_dim
                    per_agent_obs_list.append(obs[:d_i])
            elif trained_obs_mode == "ball_self":
                # each agent sees its own ball+self local obs
                per_agent_obs_list = env.get_per_agent_ball_self_obs()
            else:
                raise ValueError(f"Unknown trained_obs_mode: {trained_obs_mode}")

            all_actions = []
            with torch.no_grad():
                for i in range(4):
                    obs_i = per_agent_obs_list[i].astype(np.float32)
                    obs_tensor = torch.from_numpy(obs_i).unsqueeze(0).to(device)  # (1, obs_dim_for_this_agent)

                    action_i, _, _ = agents[i].act(obs_tensor)
                    action_i = action_i.squeeze(0).cpu().numpy()  # (2,)
                    all_actions.append(action_i)

            # Pack 4x2 into 8-dim action vector for the env
            joint_action = np.zeros(8, dtype=np.float32)
            for rod_idx in range(4):
                slide, rotate = all_actions[rod_idx]
                joint_action[rod_idx] = slide
                joint_action[rod_idx + 4] = rotate

            next_obs, reward, terminated, truncated, _ = env.step(joint_action)
            ep_return += reward

            # keep env's native obs around for next step if needed
            obs = next_obs.astype(np.float32)

            done = terminated or truncated

            if render:
                time.sleep(sleep_time)

        episode_returns.append(ep_return)
        print(f"[EP {ep}/{num_episodes}] Return: {ep_return:.3f}, steps: {step_count}")

    return episode_returns


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 4 MARL foosball agents (red) vs blue bot."
    )
    parser.add_argument(
        "--rod-paths",
        type=str,
        nargs=4,
        help="Paths to 4 rod model files, in order rod0 rod1 rod2 rod3.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory containing rod0.pt..rod3.pt (overrides --prefix).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="saves/marl_rod_stage1_",
        help="Prefix for rod files if --rod-paths and --run-dir not given (prefix0.pt..prefix3.pt).",
    )

    parser.add_argument(
        "--obs-mode",
        type=str,
        default="full",
        choices=["full", "ball_self"],
        help="(User hint) Desired obs mode. Actual mode is inferred from checkpoint.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dim used when training the agents.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=2000,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--curriculum",
        type=int,
        default=2,
        help="Curriculum level for env. 2 or 3 recommended for simple blue bot.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Use GUI render mode.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    render_mode = "human" if args.render else "direct"

    print(
        f"[INFO] Creating FoosballEnv with curriculum_level={args.curriculum}, "
        f"player_id=1 (red), obs_mode={args.obs_mode}"
    )
    # If your FoosballEnv __init__ signature doesn't have obs_mode, remove "obs_mode=args.obs_mode".
    env = FoosballEnv(
        render_mode=render_mode,
        curriculum_level=args.curriculum,
        player_id=1,
        steps_per_episode=args.max_episode_steps,
        # obs_mode=args.obs_mode,   # keep/remove depending on your env
    )

    env_full_dim = env.observation_space.shape[0]

    # ---------- resolve rod_paths ----------
    if args.rod_paths is not None:
        rod_paths = args.rod_paths
    elif args.run_dir is not None:
        rod_paths = [os.path.join(args.run_dir, f"rod{i}.pt") for i in range(4)]
    else:
        rod_paths = None  # load_agents will use prefix

    # ---------- peek at first checkpoint to infer trained obs mode ----------
    if rod_paths is not None:
        first_path = rod_paths[0]
    else:
        first_path = f"{args.prefix}0.pt"

    state_dict0 = torch.load(first_path, map_location=device)
    in_dim0 = state_dict0["fc1.weight"].shape[1]

    if in_dim0 == 8:
        trained_obs_mode = "ball_self"
    else:
        # any other dim -> assume some version of "full" obs and just truncate env obs
        trained_obs_mode = "full"

    print(
        f"[INFO] Detected checkpoint input dim = {in_dim0}. "
        f"Env full obs_dim = {env_full_dim}. "
        f"Inferred trained_obs_mode = {trained_obs_mode}."
    )
    if trained_obs_mode != args.obs_mode:
        print(
            f"[WARN] You passed --obs-mode {args.obs_mode}, "
            f"but checkpoints appear to be trained with {trained_obs_mode}. "
            f"Evaluation will follow the checkpoint ({trained_obs_mode})."
        )

    action_dim = 2  # per-rod: slide, rotate

    # ---------- load agents (obs_dim inferred inside) ----------
    agents = load_agents(
        hidden_dim=args.hidden_dim,
        action_dim=action_dim,
        device=device,
        rod_paths=rod_paths,
        prefix=args.prefix,
    )

    # ---------- run episodes: red agents vs blue bot ----------
    episode_returns = run_episodes(
        env=env,
        agents=agents,
        device=device,
        num_episodes=args.episodes,
        max_steps=args.max_episode_steps,
        trained_obs_mode=trained_obs_mode,
        render=args.render,
        sleep_time=1.0 / 240.0,
    )

    env.close()

    mean_ret = float(np.mean(episode_returns))
    std_ret = float(np.std(episode_returns))
    print(
        f"\n[RESULT] Episodes: {args.episodes}, "
        f"Mean return: {mean_ret:.3f} ± {std_ret:.3f}"
    )


if __name__ == "__main__":
    main()
