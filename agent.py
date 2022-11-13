import numpy as np
import tensorflow as tf
from tensorflow import keras

from losses import PPOLoss
from neural_networks import ActorNetworkFromSimple, CriticNetworkFromSimple
from ppo_memory import Memory
from tensorboard_config import TensorBoardConf


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, entropy=0.001,
                 gae_lambda=0.95, policy_clipping_range=0.2, critic_discount=0.5, chkpt_dir='saved_agents/',
                 tensorboard_callback: TensorBoardConf = None):
        self.gamma = gamma
        self.policy_clip = policy_clipping_range
        self.critic_discount = critic_discount
        self.entropy = entropy
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.memory = Memory()
        self.actor, self.policy = ActorNetworkFromSimple(n_actions).build_model(input_dims=input_dims, summary=True)
        self.critic = CriticNetworkFromSimple().build_model(input_dims=input_dims, summary=True)
        self.callback = tensorboard_callback
    def store_transition(self, state, action, action_prob, action_onehot, q_value, mask, reward):
        self.memory.store_in_memory(state, action, action_prob, action_onehot, q_value, mask, reward)

    def clear_agent_memory(self):
        self.memory.clear()

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir + 'model_actor.hdf5')
        self.critic.save(self.chkpt_dir + 'model_critic.hdf5')


    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir + 'model_actor.hdf5', compile=False)
        self.critic = keras.models.load_model(self.chkpt_dir + 'model_critic.hdf5')

    def choose_action(self, observation):
        dummy_n = np.zeros((1, 1, self.n_actions))
        dummy_1 = np.zeros((1, 1, 1))
        # state = tf.convert_to_tensor([observation])
        state = np.expand_dims(np.array(observation), 0)
        # state = tf.convert_to_tensor(state)
        # state = keras.backend.expand_dims(tf.convert_to_tensor(observation), 0)
        action_dist = self.actor.predict([state, dummy_n, dummy_1, dummy_1, dummy_1], batch_size=128)
        q_value = self.critic.predict(state, batch_size=128)
        executable_action = np.random.choice(self.n_actions, p=action_dist[0, :])
        # select a random action according to the calculated distribution
        action_onehot = np.zeros(self.n_actions)
        action_onehot[executable_action] = 1

        return executable_action, action_dist, action_onehot, q_value

    def get_advantage(self, state):
        # input_state = np.expand_dims(np.array(state), 0)
        # input_state = tf.convert_to_tensor(input_state)
        input_state = keras.backend.expand_dims(tf.convert_to_tensor(state), 0)

        q_value = self.critic.predict(input_state,
                                      batch_size=128)  # needs one more q_values as we need the value of t+1 state
        self.memory.add_q_value(q_value)

        returns = []
        gae = 0
        for t in reversed(range(self.memory.get_batch_size())):
            delta = self.memory.rewards[t] + self.gamma * self.memory.values[t + 1] * self.memory.masks[t] - \
                    self.memory.values[t]
            gae = delta + self.gamma * self.gae_lambda * self.memory.masks[t] * gae
            returns.insert(0, gae + self.memory.values[t])
        advantages = np.array(returns) - self.memory.values[:-1]
        normalized_adv = (advantages - np.mean(advantages)) / (
                np.std(advantages) + 1e-10)  # +1e-10 to make sure not to devide by 0

        # logging.info(f'RETURNS: {returns}\r ADVANTAGES: {advantages}\r NORMALIZED: {normalized_adv}')
        return returns, normalized_adv

    def train_models(self, advantages, returns, n_epochs=8):
        self.actor.fit(
            [np.array(self.memory.states), np.array(self.memory.action_probabilities), np.array(advantages),
             np.reshape(self.memory.rewards, newshape=(-1, 1, 1)), np.array(self.memory.values[:-1])],
            # values only the last one
            [np.reshape(self.memory.actions_onehot, newshape=(-1, self.n_actions))],
            shuffle=True, verbose=True, epochs=n_epochs, batch_size=128, callbacks=self.callback())

        self.critic.fit([np.array(self.memory.states)],
                        [np.reshape(returns, newshape=(-1, 1))],
                        shuffle=True, verbose=True, epochs=n_epochs, batch_size=128)
