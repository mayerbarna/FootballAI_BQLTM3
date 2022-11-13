import gfootball.env as football_env
import numpy as np
import tensorflow as tf
from agent import Agent

from tensorflow.python.framework.ops import disable_eager_execution

from losses import PPOLoss
from neural_networks import ActorNetworkFromSimple
from tensorboard_config import TensorBoardConf

CALLBACK = True
CONTINUE_TRAINING = False
RENDER = True
# SCENARIO = 'academy_counterattack_hard'
SCENARIO = 'academy_empty_goal'
REPRESENTATION = 'simple115v2'
# REWARDS = 'scoring,checkpoints'
REWARDS = 'scoring'

env = football_env.create_environment(env_name=SCENARIO, representation=REPRESENTATION, render=RENDER,
                                      rewards=REWARDS)

state = env.reset()
print(np.shape(state))
state_dimension = env.observation_space.shape
n_actions = env.action_space.n

# INIT AGENT
if CALLBACK:
    actor_critic_agent = Agent(n_actions, input_dims=state_dimension, tensorboard_callback=TensorBoardConf())
else:
    actor_critic_agent = Agent(n_actions, input_dims=state_dimension)

if CONTINUE_TRAINING:
    actor_critic_agent.load_models()

# TESTING
def test_reward(actor: ActorNetworkFromSimple):
    dummy_n = np.zeros((1, 1, n_actions))
    dummy_1 = np.zeros((1, 1, 1))
    state = env.reset()

    is_done = False
    total_reward = 0

    print('...testing models...')

    max_num_steps_limit = 0  # if it takes forever to score it is considered as not scoring
    while not is_done:
        state_input = np.expand_dims(np.array(state), 0)
        actions_probabilities = actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1],
                                                                 steps=1)
        executable_action = np.argmax(actions_probabilities)
        print(executable_action)
        next_state, reward, is_done, information = env.step(executable_action)
        state = next_state
        total_reward += reward
        max_num_steps_limit += 1
        if max_num_steps_limit > 80:
            break
    return total_reward

# LEARNING

reached_model_target = False  # tells us whether the model is good enough or not
iters = 0
max_iters = 10000
steps_of_ppo = 128
best_reward = 0

while not reached_model_target and iters < max_iters:

    actor_critic_agent.memory.clear()

    for _ in range(steps_of_ppo):
        executable_action, action_dist, action_onehot, q_value = actor_critic_agent.choose_action(state)


        observation, reward, is_done, information = env.step(executable_action)
        print(
            'itr: ' + str(iters) + ', action=' + str(executable_action) + ', reward=' + str(reward) + ', q val=' + str(
                q_value))

        mask = not is_done

        actor_critic_agent.memory.store_in_memory(state, executable_action, action_dist, action_onehot, q_value, mask,
                                                  reward)

        state = observation  # set the next state to our observation

        if is_done:
            env.reset()

    # GAE
    returns, advantages = actor_critic_agent.get_advantage(state)
    actor_critic_agent.train_models(advantages, returns)


    avg_reward = np.mean([test_reward(actor_critic_agent.actor) for _ in range(2)])
    print('total test reward=' + str(avg_reward))
    if best_reward < avg_reward:
        best_reward = avg_reward
        actor_critic_agent.save_models()
    if best_reward > 0.9 or iters > max_iters:
        reached_model_target = True
    iters += 1
    env.reset()
env.close()
