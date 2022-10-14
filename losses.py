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

            P1 = ratio * advantages
            P2 = keras.clip(ratio, min_value=1 - self.clipping_range, max_value=1 + self.clipping_range) * advantages
            actor_loss = -keras.mean(keras.minimum(P1, P2))
            critic_loss = keras.mean(keras.square(rewards - values))
            total_loss = critic_loss * self.critic_discount + actor_loss - self.entropy * keras.mean \
                    (
                    -(new_policy_probs * keras.log(new_policy_probs + 1e-10))
                )

            return total_loss

        return loss

    def ppo_loss_print(self, old_policy_probs, advantages, rewards, values):
        def loss(y_true, y_pred):
            newpolicy_probs = y_pred
            y_true = tf.print(y_true, [y_true], 'y_true: ')
            y_pred = tf.print(y_pred, [y_pred], 'y_pred: ')

            # newpolicy_probs = y_true * y_pred
            tf.print(newpolicy_probs, [newpolicy_probs], 'new policy probs: ')

            ratio = keras.exp(keras.log(newpolicy_probs + 1e-10) - keras.log(old_policy_probs + 1e-10))
            ratio = tf.print(ratio, [ratio], 'ratio: ')
            p1 = ratio * advantages
            p2 = keras.clip(ratio, min_value=1 - self.clipping_range, max_value=1 + self.clipping_range) * advantages
            actor_loss = -keras.mean(keras.minimum(p1, p2))
            actor_loss = tf.print(actor_loss, [actor_loss], 'actor_loss: ')
            critic_loss = keras.mean(keras.square(rewards - values))
            critic_loss = tf.print(critic_loss, [critic_loss], 'critic_loss: ')
            term_a = self.critic_discount * critic_loss
            term_a = tf.print(term_a, [term_a], 'term_a: ')
            term_b_2 = keras.log(newpolicy_probs + 1e-10)
            term_b_2 = tf.print(term_b_2, [term_b_2], 'term_b_2: ')
            term_b = self.entropy * keras.mean(-(newpolicy_probs * term_b_2))
            term_b = tf.print(term_b, [term_b], 'term_b: ')
            total_loss = term_a + actor_loss - term_b
            total_loss = tf.print(total_loss, [total_loss], 'total_loss: ')
            return total_loss

        return loss