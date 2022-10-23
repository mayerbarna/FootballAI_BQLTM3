import keras.backend as keras
import tensorflow as tf



class PPOLoss:
    def __init__(self, clipping_range=0.2, critic_discount=0.5, entropy=0.001):
        self.clipping_range = clipping_range
        self.critic_discount = critic_discount
        self.entropy = entropy

    def get_custom_ppo_loss(self, old_policy_probs, advantages, rewards, values):
        def loss(y_true, y_pred):
            print("...calculating loss...")
            new_policy_probs = y_pred

            ratio = keras.exp(keras.log(new_policy_probs + 1e-10) - keras.log(old_policy_probs + 1e-10))

            P1 = advantages * ratio
            P2 = keras.clip(ratio, min_value=1 - self.clipping_range, max_value=1 + self.clipping_range) * advantages
            actor_loss = -keras.mean(keras.minimum(P1, P2))
            critic_loss = keras.mean(keras.square(rewards - values))
            total_loss = critic_loss * self.critic_discount + actor_loss - self.entropy * keras.mean \
                    (
                    -(new_policy_probs * keras.log(new_policy_probs + 1e-10))
                )

            return total_loss

        return loss