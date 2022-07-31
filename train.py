import gfootball.env as football_env
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

CLIPPING_VALUE = 0.2
CRITIC_DISCOUNT = 0.5
ENTROPY_BETA = 0.001
GAMMA = 0.99
LMBDA = 0.95

env = football_env.create_environment(env_name='academy_counterattack_hard', representation='simple115v2', render=False)

# initialize the state env

# state = env.reset()
state_dims = env.observation_space.shape
n_actions = env.action_space.n  # number of possible actions
state_dims = env.observation_space.shape


def get_model_actor_simple(input_dims, output_dims):
    # define the neural network initial shape
    state_input = Input(shape=input_dims)  # 115
    oldpolicy_probs = Input(shape=(1, output_dims,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))

    # Classification block --> finding the optimal action from action set
    x = Dense(512, activation='relu', name='fc1')(state_input)  # 512 neuron
    x = Dense(256, activation='relu', name='fc2')(x)  # new layer with x input
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)
    # softmax --> from vector, it creates a probability distribution
    # fuggetlen esemenyek vszinusegei, aminek osszege 1 --> a legnagyobbat valasztjuk (NEM ITT)

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                  outputs=[out_actions])

    # define how the model works during learning phase
    # Adam --> how to optimize the new weights during backpropagation (lr: learning rate)
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])
    # model.summary()
    return model


def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    # custom loss function
    def loss(y_true, y_pred, K=None):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - CLIPPING_VALUE, max_value=1 + CLIPPING_VALUE) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return total_loss

    return loss

# while True:
#     observation, reward, done, info = env.step(env.action_space.sample())
#     if done:
#         env.reset()

print('new feature')
