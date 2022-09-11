import gfootball.env as football_env
from keras.models import load_model
import numpy as np
import keras.backend as K


trained_actor = 'actor_model_250_0.0.hdf5'
scenario = 'academy_counterattack_hard'

env = football_env.create_environment(env_name=scenario, representation='simple115v2', render=True)

n_actions = env.action_space.n
dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))

model_actor = load_model(trained_actor, custom_objects={'loss': 'categorical_hinge'})


state = env.reset()
done = False

while True:
    state_input = K.expand_dims(state, 0)
    action_probs = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
    action = np.argmax(action_probs)
    print(action)
    next_state, _, done, _ = env.step(action)
    state = next_state
    if done:
        state = env.reset()