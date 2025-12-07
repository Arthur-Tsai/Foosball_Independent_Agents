# train_marl_four_agents.py

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os   # <- add this

from foosball_env_independent import FoosballEnv
from marl_single_rod_policy import SingleRodActorCritic


def compute_gae(rewards, values, last_value, dones, gamma, lam):
    """
    rewards: (T,)
    values:  (T,)
    dones:   (T,)
    last_value: scalar
    returns:
        advantages: (T,)
        returns:    (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--curriculum", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-prefix", type=str, default="saves/marl_rod")

    # NEW: observation mode (like train_full_obs / train_ball_self)
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="full",
        choices=["full", "ball_self"],
        help="Observation mode: 'full' (ball + all joints) or 'ball_self' (ball + own team joints)",
    )

    args = parser.parse_args()

    # --- device & seeding ---
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- env ---
    env = FoosballEnv(
        render_mode="direct",           # non-GUI for training
        curriculum_level=args.curriculum,
        player_id=1,
        steps_per_episode=args.max_episode_steps,
        obs_mode=args.obs_mode,         # <--- use the chosen observation mode
    )

    obs_dim = env.observation_space.shape[0]
    num_agents = 4
    action_dim = 2

    print(f"[INFO] obs_mode = {args.obs_mode}, obs_dim = {obs_dim}")

    # --- 4 independent rod agents ---
    agents = []
    optimizers = []
    for _ in range(num_agents):
        agent = SingleRodActorCritic(
            obs_dim=obs_dim,
            hidden_dim=args.hidden_dim,
            action_dim=action_dim
        ).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.lr)
        agents.append(agent)
        optimizers.append(optimizer)

    obs, _ = env.reset()
    obs = obs.astype(np.float32)

    for update in range(1, args.updates + 1):
        # Rollout buffers (shared reward & obs, per-agent actions/values/logprobs)
        obs_buf = np.zeros((args.rollout_steps, obs_dim), dtype=np.float32)
        rewards_buf = np.zeros(args.rollout_steps, dtype=np.float32)
        dones_buf = np.zeros(args.rollout_steps, dtype=np.float32)

        actions_buf = np.zeros((num_agents, args.rollout_steps, action_dim), dtype=np.float32)
        values_buf = np.zeros((num_agents, args.rollout_steps), dtype=np.float32)
        logprobs_buf = np.zeros((num_agents, args.rollout_steps), dtype=np.float32)

        for t in range(args.rollout_steps):
            obs_buf[t] = obs

            # Each agent acts independently on the SAME obs (full or ball_self)
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)  # (1, obs_dim)

            all_actions = []
            with torch.no_grad():
                for i in range(num_agents):
                    action_i, logp_i, value_i = agents[i].act(obs_tensor)
                    action_i = action_i.squeeze(0).cpu().numpy()
                    logp_i = logp_i.item()
                    value_i = value_i.item()

                    actions_buf[i, t] = action_i
                    logprobs_buf[i, t] = logp_i
                    values_buf[i, t] = value_i

                    all_actions.append(action_i)

            # Pack 4x2 actions into 8-dim vector
            joint_action = np.zeros(8, dtype=np.float32)
            for rod_idx in range(num_agents):
                slide, rotate = all_actions[rod_idx]
                joint_action[rod_idx] = slide
                joint_action[rod_idx + 4] = rotate

            next_obs, reward, terminated, truncated, _ = env.step(joint_action)
            done = terminated or truncated

            rewards_buf[t] = reward
            dones_buf[t] = float(done)

            obs = next_obs.astype(np.float32)

            if done:
                obs, _ = env.reset()
                obs = obs.astype(np.float32)

        # At end of rollout, get last values for GAE
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        last_values = []
        with torch.no_grad():
            for i in range(num_agents):
                _, _, v = agents[i].forward(obs_tensor)
                last_values.append(v.item())
        last_values = np.array(last_values, dtype=np.float32)

        # --- PPO update per agent ---
        for i in range(num_agents):
            advantages, returns = compute_gae(
                rewards=rewards_buf,
                values=values_buf[i],
                last_value=last_values[i],
                dones=dones_buf,
                gamma=args.gamma,
                lam=args.lam,
            )

            # Normalize advantages
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            # Prepare torch tensors
            obs_tensor_all = torch.from_numpy(obs_buf).to(device)              # (T, obs_dim)
            actions_tensor = torch.from_numpy(actions_buf[i]).to(device)       # (T, 2)
            old_logprobs_tensor = torch.from_numpy(logprobs_buf[i]).to(device) # (T,)
            returns_tensor = torch.from_numpy(returns).to(device)              # (T,)
            advantages_tensor = torch.from_numpy(advantages).to(device)        # (T,)

            T = args.rollout_steps
            batch_size = args.batch_size

            for epoch in range(args.epochs):
                # Mini-batch PPO
                indices = np.arange(T)
                np.random.shuffle(indices)

                for start in range(0, T, batch_size):
                    end = start + batch_size
                    mb_idx = indices[start:end]

                    mb_obs = obs_tensor_all[mb_idx]
                    mb_actions = actions_tensor[mb_idx]
                    mb_old_logprobs = old_logprobs_tensor[mb_idx]
                    mb_returns = returns_tensor[mb_idx]
                    mb_advantages = advantages_tensor[mb_idx]

                    new_logprobs, entropy, values = agents[i].evaluate_actions(mb_obs, mb_actions)

                    ratio = torch.exp(new_logprobs - mb_old_logprobs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = (mb_returns - values).pow(2).mean()
                    loss = policy_loss + args.vf_coef * value_loss - args.entropy_coef * entropy.mean()

                    optimizers[i].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agents[i].parameters(), args.max_grad_norm)
                    optimizers[i].step()

        # Simple logging: average reward in this rollout
        mean_reward = rewards_buf.mean()
        print(f"Update {update}/{args.updates} | Mean rollout reward: {mean_reward:.3f}")

    # --- Save each agent into a dedicated folder ---
    # Folder name encodes prefix, obs_mode, and seed, e.g.:
    # saves/marl_rod_obs-full_seed-0/
    run_dir = f"{args.save_prefix}_obs-{args.obs_mode}_seed-{args.seed}"
    os.makedirs(run_dir, exist_ok=True)

    for i in range(num_agents):
        save_path = os.path.join(run_dir, f"rod{i}.pt")
        torch.save(agents[i].state_dict(), save_path)
        print(f"Saved rod {i} policy to {save_path}")

    env.close()


if __name__ == "__main__":
    main()
