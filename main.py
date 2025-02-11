import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

USE_CUDA = torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
 

def shift_elements_3d_array(tensor_list,axis=0):
    shifted_list = []
    # for arr in tensor_list:
    #     if torch.is_tensor(arr):
    #         arr_cpu = arr.cpu()
    #         arr_numpy = arr_cpu.numpy()
    #         arr_shifted = np.roll(arr_numpy, shift=1, axis=axis)
    #         shifted_list.append(torch.from_numpy(arr_shifted).to('cuda:0'))
    if isinstance(tensor_list, np.ndarray):
        arr_shifted = np.roll(tensor_list, shift=1, axis=axis)
        shifted_list=arr_shifted
    elif isinstance(tensor_list, list):
        arr_numpy = np.array(tensor_list)
        arr_shifted = np.roll(arr_numpy, shift=1, axis=axis)
        shifted_list=list(arr_shifted)
    else:
        raise TypeError("Unsupported array type")
    return shifted_list

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True,parameter_sharing=config.parameter_sharing)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions] 
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            if config.parameter_sharing:
                obs_t=obs
                agent_actions_t=agent_actions
                rewards_t=rewards
                next_obs_t=next_obs
                dones_t=dones
                for i in range(maddpg.nagents):
                    obs_t=shift_elements_3d_array(obs_t,axis=1)
                    agent_actions_t=shift_elements_3d_array(agent_actions_t)
                    rewards_t=shift_elements_3d_array(rewards_t)
                    next_obs_t=shift_elements_3d_array(next_obs_t,axis=1)
                    dones_t=shift_elements_3d_array(dones_t)
                    replay_buffer.push(obs_t, agent_actions_t, rewards_t, next_obs_t, dones_t)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                print(f"number of replay_buf is {len(replay_buffer)}")

                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):

                    # if config.parameter_sharing:
                    #     sample = replay_buffer.sample(config.batch_size, to_gpu=USE_CUDA)
                        
                    #     for i in range(maddpg.nagents):                               
                    #         for a_i in range(maddpg.nagents):
                    #             maddpg.update(sample, a_i, logger=logger)  
                    #         maddpg.update_all_targets()

                    #         sample=list(sample)
                    #         for s_i,sam in enumerate(sample):
                    #             sample[s_i]=shift_elements_3d_array(sam)
                    #         sample=tuple(sample)
                        
                               
                    # else:
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                    to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                    # print("update")



                        # if config.parameter_sharing:
                        #     sample=list(sample)
                        #     for s_i,sam in enumerate(sample):
                        #         sample[s_i]=shift_elements_3d_array(sam)
                        #     sample=tuple(sample)
                        #     maddpg.update(sample, a_i, logger=logger)


                        #     sample=list(sample)
                        #     for s_i,sam in enumerate(sample):
                        #         sample[s_i]=shift_elements_3d_array(sam)
                        #     sample=tuple(sample)
                        #     maddpg.update(sample, a_i, logger=logger)


                    

                        # sample=list(sample)
                        # for s_i,sam in enumerate(sample):
                        #     sample[s_i]=shift_elements_3d_array(sam)
                        # sample=tuple(sample)
                            
                maddpg.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=100000, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=2048, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')
    parser.add_argument("--parameter_sharing",default=False, type=bool)
    config = parser.parse_args()

    run(config)
