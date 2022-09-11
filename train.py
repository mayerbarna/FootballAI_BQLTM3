import time

from keras.callbacks import TensorBoard

import log_conf
import os

import gfootball.env as football_env
import keras.backend as keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

GAMMA = 0.993
LAMBDA = 0.95
CLIPPING_RANGE = 0.08
CRITIC_DISCOUNT = 0.5
ENTROPY = 0.003

loss_tracking = TensorBoard(log_dir='./logs/loss_tracking', histogram_freq=1)

selected_env = 'academy_counterattack_hard'
# selected_env = 'academy_empty_goal'
env = football_env.create_environment(env_name=selected_env, representation='simple115v2', render=True,
                                      rewards='scoring,checkpoints')

state = env.reset()

state_dimension = env.observation_space.shape
n_actions = env.action_space.n

print(state_dimension)
print(n_actions)


def custom_ppo_loss_print(old_policy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        y_true = tf.print(y_true, [y_true], 'y_true: ')
        y_pred = tf.print(y_pred, [y_pred], 'y_pred: ')
        newpolicy_probs = y_pred
        # newpolicy_probs = y_true * y_pred
        newpolicy_probs = tf.print(newpolicy_probs, [newpolicy_probs], 'new policy probs: ')
        ratio = keras.exp(keras.log(newpolicy_probs + 1e-10) - keras.log(old_policy_probs + 1e-10))
        ratio = tf.print(ratio, [ratio], 'ratio: ')
        p1 = ratio * advantages
        p2 = keras.clip(ratio, min_value=1 - CLIPPING_RANGE, max_value=1 + CLIPPING_RANGE) * advantages
        actor_loss = -keras.mean(keras.minimum(p1, p2))
        actor_loss = tf.print(actor_loss, [actor_loss], 'actor_loss: ')
        critic_loss = keras.mean(keras.square(rewards - values))
        critic_loss = tf.print(critic_loss, [critic_loss], 'critic_loss: ')
        term_a = CRITIC_DISCOUNT * critic_loss
        term_a = tf.print(term_a, [term_a], 'term_a: ')
        term_b_2 = keras.log(newpolicy_probs + 1e-10)
        term_b_2 = tf.print(term_b_2, [term_b_2], 'term_b_2: ')
        term_b = ENTROPY * keras.mean(-(newpolicy_probs * term_b_2))
        term_b = tf.print(term_b, [term_b], 'term_b: ')
        total_loss = term_a + actor_loss - term_b
        total_loss = tf.print(total_loss, [total_loss], 'total_loss: ')
        return total_loss

    return loss


def custom_ppo_loss(old_policy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        new_policy_probs = y_pred
        ratio = keras.exp(keras.log(new_policy_probs + 1e-10) - keras.log(old_policy_probs + 1e-10))
        P1 = ratio * advantages
        P2 = keras.clip(ratio, min_value=1 - CLIPPING_RANGE, max_value=1 + CLIPPING_RANGE) * advantages
        actor_loss = -keras.mean(keras.minimum(P1, P2))
        critic_loss = keras.mean(keras.square(rewards - values))
        total_loss = critic_loss * CRITIC_DISCOUNT + actor_loss - ENTROPY * keras.mean \
                (
                -(new_policy_probs * keras.log(new_policy_probs + 1e-10))
            )
        return total_loss

    return loss


def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t + 1] * masks[t] - values[t]
        gae = delta + GAMMA * LAMBDA * masks[t] * gae
        returns.insert(0, gae + values[t])
    advantages = np.array(returns) - values[:-1]
    normalized_adv = (advantages - np.mean(advantages)) / (
            np.std(advantages) + 1e-10)  # +1e-10 to make sure not to devide by 0
    # logging.info(f'RETURNS: {returns}\r ADVANTAGES: {advantages}\r NORMALIZED: {normalized_adv}')
    return returns, normalized_adv


# defined the actor model to give us an action depending on the current state instead of a random one

def get_ppo_actor_model_from_simple(input_dimensions, output_dimensions):
    state_input_shape = Input(shape=input_dimensions)  # define the input shape from the simple data

    # Custom loss
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))
    oldpolicy_probs = Input(shape=(1, output_dimensions,))

    # Classification block
    x = Dense(512, activation='relu', name='fc1')(state_input_shape)
    x = Dense(256, activation='relu', name='fc2')(x)
    output_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

    # simple representation of the game:
    # 22 - (x,y) coordinates of left team players
    # 22 - (x,y) direction of left team players
    # 22 - (x,y) coordinates of right team players
    # 22 - (x, y) direction of right team players
    # 3 - (x, y and z) - ball position
    # 3 - ball direction
    # 3 - one hot encoding of ball ownership (noone, left, right)
    # 11 - one hot encoding of which player is active
    # 7 - one hot encoding of game_mode

    # model input --> miert kell bele az input shapen kivul a tobbi?
    model = Model(inputs=[state_input_shape, oldpolicy_probs, advantages, rewards, values],
                  outputs=[output_actions])

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=[custom_ppo_loss(old_policy_probs=oldpolicy_probs, advantages=advantages,
                                        rewards=rewards, values=values)])
    model.summary()
    return model


def get_ppo_critic_model_from_simple(input_dimensions, output_dimensions):
    """
    Indicates how good/bad the taken action
    """
    state_input_shape = Input(shape=input_dimensions)  # define the input shape from the simple data

    # Classification block
    x = Dense(512, activation='relu', name='fc1')(state_input_shape)
    x = Dense(256, activation='relu', name='fc2')(x)

    # gives back a real number in connection with how good was the taken action (not a probability distribution)
    output_actions = Dense(1, activation='tanh', name='predictions')(x)

    model = Model(inputs=[state_input_shape],
                  outputs=[output_actions])

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    model.summary()
    return model


# PPO

initial_n = np.zeros((1, 1, n_actions))
initial_1 = np.zeros((1, 1, 1))

# use pretrained models
# actor_model = load_model('actor_model_250_0.0.hdf5', custom_objects={'loss': 'categorical_hinge'})

actor_model = get_ppo_actor_model_from_simple(input_dimensions=state_dimension, output_dimensions=n_actions)
critic_model = get_ppo_critic_model_from_simple(input_dimensions=state_dimension, output_dimensions=n_actions)


def test_reward():
    state = env.reset()  # reset in order to start a new game
    is_done = False
    total_reward = 0
    print('testing...')
    max_num_steps_limit = 0  # if it takes forever to score it is considered as not scoring
    while not is_done:
        state_input = keras.expand_dims(state, 0)
        actions_probabilities = actor_model.predict([state_input, initial_n, initial_1, initial_1, initial_1], steps=1)
        executable_action = np.argmax(actions_probabilities)
        next_state, reward, is_done, information = env.step(executable_action)
        state = next_state
        total_reward += reward
        max_num_steps_limit += 1
        if max_num_steps_limit > 20:
            break
    return total_reward


reached_model_target = False  # tells us whether the model is good enough or not
best_reward = 0
iters = 0
max_iters = 500
steps_of_ppo = 128

while not reached_model_target and iters < max_iters:

    states = []
    actions = []
    actions_probabilities = []
    actions_onehot = []
    values = []  # generated by the critic model
    masks = []  # used to check if the game is completed
    rewards = []
    input_tensor = None

    for _ in range(steps_of_ppo):
        input_tensor = keras.expand_dims(state, 0)
        distribution_of_actions = actor_model.predict([input_tensor, initial_n, initial_1, initial_1, initial_1],
                                                      steps=1)

        q_value = critic_model.predict([input_tensor], steps=1)

        executable_action = np.random.choice(n_actions, p=distribution_of_actions[0,
                                                          :])  # select a random action according to the calculated distribution

        action_onehot = np.zeros(n_actions)
        action_onehot[executable_action] = 1
        observation, reward, is_done, information = env.step(executable_action)
        print(
            'itr: ' + str(iters) + ', action=' + str(executable_action) + ', reward=' + str(reward) + ', q val=' + str(
                q_value))
        mask = not is_done

        # logging.info(
        #     f'ENVIRONMENT STATE --> OBSERVATION:\r {observation},\r REWARD: [{reward}],\r IS DONE: [{is_done}],\r ADD.INFO: [{information}] \r')

        states.append(state)
        actions.append(executable_action)
        actions_onehot.append(action_onehot)
        actions_probabilities.append(distribution_of_actions)
        values.append(q_value)
        masks.append(mask)
        rewards.append(reward)

        state = observation  # state variable needs to be equal with the new observation or the next state

        #  time.sleep(1)

        if is_done:
            env.reset()

    # logging.info(
    #     f'\rSTATES: {states}\r ACTIONS: {actions}\r ACTIONS ONEHOT: {actions_onehot}\r ACTION PROBS: {actions_probabilities}\r VALUES: {values}\r MASKS: {masks}\r REWARDS: {rewards}')

    # Generalized Advantage Estimation (GAE)
    input_tensor = keras.expand_dims(state, 0)

    q_value = critic_model.predict(input_tensor, steps=1)  # needs one more q_values as we need the value of t+1 state
    values.append(q_value)

    # calculate advantage from the returned values to train the actor model
    returns, advantages = get_advantages(values, masks, rewards)

    # model training

    actor_model.fit([states, actions_probabilities, advantages,
                     np.reshape(rewards, newshape=(-1, 1, 1)), values[:-1]],
                    # values only the last one
                    [np.reshape(actions_onehot, newshape=(-1, n_actions))],
                    shuffle=True, verbose=True, epochs=8, callbacks=[loss_tracking])

    critic_model.fit([states],
                     [np.reshape(returns, newshape=(-1, 1))],
                     shuffle=True, verbose=True, epochs=8, callbacks=[loss_tracking])

    # model evaluation

    avg_reward = np.mean([test_reward() for _ in range(5)])
    print('total test reward=' + str(avg_reward))

    # after every 10 iters we save the models
    if iters % 25 == 0:
        actor_model.save('saved_agents/actor_model_new_{}_{}.hdf5'.format(iters, avg_reward))
        critic_model.save('saved_agents/critic_model_new_{}_{}.hdf5'.format(iters, avg_reward))

    # if we scored a goal 90% of the total times we need to save the checkpoint

    if avg_reward >= best_reward:
        print('best reward=' + str(avg_reward))
        actor_model.save('saved_agents/actor_model_new_{}_{}.hdf5'.format(iters, avg_reward))
        critic_model.save('saved_agents/critic_model_new_{}_{}.hdf5'.format(iters, avg_reward))
        best_reward = avg_reward
    if best_reward > 0.9 or iters > max_iters:
        reached_model_target = True

    print(str('iters: ') + str(iters))
    iters += 1
    env.reset()

env.close()
