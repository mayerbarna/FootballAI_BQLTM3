import gfootball.env as football_env

env = football_env.create_environment(env_name='academy_counterattack_hard', representation='simple115', render=True)

# initialize the state env

state = env.reset()

while True:
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()
