import numpy as np
import random
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        world.num_agents = num_agents
        num_adversaries = 1
        num_landmarks = num_agents
        self.landmark_size = 0.1
        self.agent_size = 0.15
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = self.agent_size
            agent.max_speed=random.uniform(0.1,0.9)
            agent.accel=0.4
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = self.landmark_size
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for agent in world.agents:
            agent.color = self.speed_to_color(agent.max_speed, 1.5)
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < (self.landmark_size + self.agent_size)/2:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
                    
        return (rew, collisions, min_dists, occupied_landmarks)



    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
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
        other_ability = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_ability.append(np.array([other.max_speed - agent.max_speed]))
        # import ipdb; ipdb.set_trace()
        # return np.concatenate([agent.state.p_pos] + entity_pos + other_pos)
        # print(np.concatenate( [agent.state.p_pos] + other_pos ))
        # print(entity_pos)
        # print(np.concatenate( other_ability+[np.array([agent.max_speed])]))
        return np.concatenate( [agent.state.p_pos] + entity_pos + other_pos)# + other_ability+[np.array([agent.max_speed])])
        
    def speed_to_color(self, speed, max_speed):
        # 速度に応じた色を計算する
        norm_speed = speed / max_speed  # 速度を正規化
        color = np.array([norm_speed, 0.35, 1 - norm_speed])  # 赤-青のグラデーション
        return color
