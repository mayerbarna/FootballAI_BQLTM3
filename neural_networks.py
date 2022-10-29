from keras import Input, Model
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow import keras


from losses import PPOLoss


class CriticNetworkFromSimple(keras.Model):
    """
        Indicates how good/bad the taken action was
    """

    def __init__(self, hl1_dims=512, hl2_dims=256):
        super(CriticNetworkFromSimple, self).__init__()

        selected_action_num = 1

        self.layer_1 = Dense(hl1_dims, activation='relu', name='fc1')
        self.layer_2 = Dense(hl2_dims, activation='relu', name='fc2')
        self.output_layer = Dense(selected_action_num, activation='tanh', name='predictions')
        # tanh activation -> as the given q_value can be negative and positive as well

    def build_model(self, input_dims, summary=True):
        state_input_shape = Input(shape=input_dims)  # define the input shape from the simple data

        # Classification block
        x = self.layer_1(state_input_shape)
        x = self.layer_2(x)
        q = self.output_layer(x)

        model = Model(inputs=[state_input_shape],
                      outputs=[q])

        model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
        if summary:
            model.summary()
        return model


class ActorNetworkFromSimple(keras.Model):
    def __init__(self, n_actions, hl1_dims=512, hl2_dims=256):
        super(ActorNetworkFromSimple, self).__init__()

        self.n_actions = n_actions

        self.layer_1 = Dense(hl1_dims, activation='relu', name='fc1')
        self.layer_2 = Dense(hl2_dims, activation='relu', name='fc2')
        self.output_layer = Dense(n_actions, activation='softmax', name='predictions')

    def build_model(self, input_dims, summary=True):
        state_input_shape = Input(shape=input_dims)

        advantages = Input(shape=(1, 1,))
        rewards = Input(shape=(1, 1,))
        values = Input(shape=(1, 1,))
        oldpolicy_probs = Input(shape=(1, self.n_actions,))

        # Classification block
        x = self.layer_1(state_input_shape)
        x = self.layer_2(x)
        output_actions = self.output_layer(x)

        model = Model(inputs=[state_input_shape, oldpolicy_probs, advantages, rewards, values],
                      outputs=[output_actions])

        loss = PPOLoss()

        model.add_loss(loss.get_custom_ppo_loss(
            y_true=None,
            y_pred=output_actions,
            old_policy_probs=oldpolicy_probs,
            advantages=advantages,
            rewards=rewards,
            values=values))
        model.compile(optimizer=Adam(learning_rate=1e-4))

        policy = Model(inputs=[state_input_shape], outputs=[output_actions])
        if summary:
            model.summary()
        return model, policy
