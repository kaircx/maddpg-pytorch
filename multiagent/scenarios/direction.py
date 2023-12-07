import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_right_agents = 1
        num_left_agents = 1 ##advarsarial
        num_agents = num_right_agents + num_left_agents
        num_landmarks = 2
        # add agents
        r=[True,False]
        random.shuffle(r)
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = False
            agent.right = r[i] 
            agent.size =0.15
            agent.accel = 4.0
            agent.max_speed = 0.5 
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.1
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if agent.right else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos =  np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-1, -0.5)])# if hasattr(agent,"right")else np.array([np.random.uniform(0, 0.5), np.random.uniform(-1, 0.5)])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array([np.random.uniform(-1.0, -0.5), np.random.uniform(-0.7, 1.0)]) if i==0 else np.array([np.random.uniform(0.7, 1.0), np.random.uniform(0.5, 1.0)])
            landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.slow]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.slow]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        # rew = 0
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)
        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= 1
        # return rew
    
        # # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = -0.1  # この値は必要に応じて調整可能

        # 各ランドマークに対して最も近いエージェントの距離を計算
        min_dists_to_landmarks = []
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dist = min(dists)
            min_dists_to_landmarks.append(min_dist)
            rew -= min_dist
        # 全てのエージェントが全てのランドマークに到達した場合の報酬調整

        if all(dist < 0.03 for dist in min_dists_to_landmarks):  # some_threshold は適切な閾値
            rew += sum(min_dists_to_landmarks)
            rew += 0.1   # マイナスされた報酬を戻す
            print("clear!!")
            print(min_dists_to_landmarks)

        # 衝突に対するペナルティ
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        # print(f"reward is {rew}")
        return rew


    
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate( [agent.state.p_pos] + entity_pos + other_pos + comm + [np.array([1.0 if agent.right else 0.0])])
        return np.concatenate([agent.state.p_pos] + entity_pos + other_pos)#+ [np.array([1.0 if agent.right else 0.0])])
