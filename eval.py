from magent2.environments import battle_v4
import torch
import numpy as np
import imageio

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x  # Fallback: tqdm becomes a no-op


def eval():
    max_cycles = 300
    env = battle_v4.env(map_size=45, max_cycles=max_cycles, render_mode = "rgb_array")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blue_path = "/Users/khai/Downloads/RL-final-project-AIT-3007/blue.pt"
    def random_policy(env, agent, obs):
        return env.action_space(agent).sample()

    q_network_red_final = QNetwork_final(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network_red_final.load_state_dict(
        torch.load('red_final.pt', weights_only=True, map_location="cpu")
    )
    q_network_red_final.to(device)

    q_network_red = QNetwork_G(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network_red.load_state_dict(
        torch.load('red.pt', weights_only=True, map_location="cpu")
    )
    q_network_red.to(device)


    q_network_blue = QNetwork(
        env.observation_space("blue_0").shape, env.action_space("blue_0").n
    )
    q_network_blue.load_state_dict(
        torch.load(blue_path, weights_only=True, map_location="cpu")
    )
    q_network_blue.to(device)

    def blue_pretrain_policy(env, agent, obs):
        observation = (
            torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0).to(device)
        )
        with torch.no_grad():
              q_values = q_network_blue(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]

    def red_pretrain_policy(env, agent, obs):
        observation = (
            torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0).to(device)
        )
        with torch.no_grad():
              q_values = q_network_red(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]

    def final_red_pretrain_policy(env, agent, obs):
        observation = (
            torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0).to(device)
        )
        with torch.no_grad():
              q_values = q_network_red_final(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]

    def run_eval(env, red_policy, blue_policy, n_episode: int = 100):
        red_win, blue_win = [], []
        red_tot_rw, blue_tot_rw = [], []
        n_agent_each_team = len(env.env.action_spaces) // 2

        for i in tqdm(range(n_episode)):
          # Thêm vào để tạo gif
            frames = []
            add_frame = True
            env.reset()
            n_kill = {"red": 0, "blue": 0}
            red_reward, blue_reward = 0, 0

            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                agent_team = agent.split("_")[0]

                n_kill[agent_team] += (
                    reward > 4.5
                )  # This assumes default reward settups
                if agent_team == "red":
                    red_reward += reward
                else:
                    blue_reward += reward

                if termination or truncation:
                    action = None  # this agent has died
                else:
                    if agent_team == "red":
                        action = red_policy(env, agent, observation)
                    else:
                        action = blue_policy(env, agent, observation)

                env.step(action)
             # Thêm vào để tạo gif
                if 'red' in agent:
                  if add_frame:
                    frames.append(env.render())
                    add_frame = False
                else:
                  add_frame = True
                # =================

            # Thêm vào để tạo gif
            frames.append(env.render())

            with imageio.get_writer('test-video.gif', mode='I', fps=12) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print("Done recording pretrained agents")
            # ======================================


            who_wins = "red" if n_kill["red"] >= n_kill["blue"] + 5 else "draw"
            who_wins = "blue" if n_kill["red"] + 5 <= n_kill["blue"] else who_wins
            red_win.append(who_wins == "red")
            blue_win.append(who_wins == "blue")
            print('red kill',n_kill['red'])
            print('blue kill',n_kill['blue'])
            red_tot_rw.append(red_reward / n_agent_each_team)
            blue_tot_rw.append(blue_reward / n_agent_each_team)

        return {
            "winrate_red": np.mean(red_win),
            "winrate_blue": np.mean(blue_win),
            "average_rewards_red": np.mean(red_tot_rw),
            "average_rewards_blue": np.mean(blue_tot_rw),
        }

    print("=" * 20)
    ''' ("Eval with random policy")
    print(
        run_eval(
            env=env, red_policy=random_policy, blue_policy=blue_pretrain_policy, n_episode=100
        )
    )
    print("=" * 20)

    print("Eval with trained policy")
    print(
        run_eval(
            env=env, red_policy=red_pretrain_policy, blue_policy=blue_pretrain_policy, n_episode=100
        )
    )
    print("=" * 20)'''
    print("Eval with final trained policy")
    print(
        run_eval(
            env=env, red_policy=final_red_pretrain_policy, blue_policy=blue_pretrain_policy, n_episode=100
        )
    )
    print("=" * 20)


if __name__ == "__main__":
    eval()