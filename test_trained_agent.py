import gfootball.env as football_env
import keras.backend
from keras.models import load_model
import numpy as np
import keras.backend as K
import tensorflow as tf
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
    input_state = np.expand_dims(state, 0)
    input_state = tf.convert_to_tensor(input_state)
    actions_probabilities = actor_critic_agent.policy.predict([input_state],
                                                              batch_size=128)
    executable_action = np.argmax(actions_probabilities)
    print(executable_action)
    next_state, _, done, _ = env.step(executable_action)
    state = next_state
    if done:
        state = env.reset()
