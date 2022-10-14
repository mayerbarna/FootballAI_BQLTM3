import gfootball.env as football_env
from keras.models import load_model
import numpy as np
import keras.backend as K

from agent import Agent

trained_actor = 'content/model_checkpoints/saved_model.h5'
# scenario = 'academy_counterattack_hard'
scenario = 'academy_empty_goal'

env = football_env.create_environment(env_name=scenario, representation='simple115v2', render=True)

n_actions = env.action_space.n
state_dimension = env.observation_space.shape


state = env.reset()
done = False
actor_critic_agent = Agent(n_actions, input_dims=state_dimension)
actor_critic_agent.load_models()
while True:
    executable_action, action_dist, action_onehot, q_value = actor_critic_agent.choose_action(state)
    print(executable_action)
    next_state, _, done, _ = env.step(executable_action)
    state = next_state
    if done:
        state = env.reset()
