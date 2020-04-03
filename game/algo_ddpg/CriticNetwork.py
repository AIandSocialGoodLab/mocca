import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400

import tensorflow.keras.backend as K

def create_critic_network(state_size, action_size, learning_rate):
    """Creates a critic network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
        learning_rate: (float) learning rate for the critic.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
        action_input: a tf.placeholder for the batched action.
    """

   
    state_input = tf.keras.layers.Input(state_size)
    action_input = tf.keras.layers.Input(action_size)

    # -------- more advanced network structure --------------
    # b_state_1 = tf.keras.layers.Dense(HIDDEN1_UNITS, activation=tf.nn.relu)(state_input)
    # b_state_2 = tf.keras.layers.Dense(HIDDEN1_UNITS, activation='linear')(b_state_1)
    # b_action = tf.keras.layers.Dense(HIDDEN1_UNITS, activation='linear')(action_input)
    
    # mer = tf.keras.layers.add([b_state_2, b_action])
    # mer2 = Dense(HIDDEN2_UNITS, activation='tanh')(mer)
    # value = tf.keras.layers.Dense(1, activation='linear')(mer2)

    # -------- simple concatenate network structure with relu --------
    inputs = tf.keras.backend.concatenate((state_input, action_input), axis=-1)
    l1 = tf.keras.layers.Dense(HIDDEN1_UNITS, activation=tf.nn.relu)(inputs)
    l2 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation=tf.nn.relu)(l1)
    value = tf.keras.layers.Dense(1, activation=None)(l2)

    # ------- simple concatenate network structure with tanh
    # inputs = tf.keras.backend.concatenate((state_input, action_input), axis=-1)
    # l1 = tf.keras.layers.Dense(HIDDEN1_UNITS, activation=tf.nn.tanh)(inputs)
    # l2 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation=tf.nn.tanh)(l1)
    # value = tf.keras.layers.Dense(1)(l2)


    model = tf.keras.Model(inputs=[state_input, action_input], outputs=value)
    model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    return model, state_input, action_input, model.trainable_weights


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate):
        """Initialize the CriticNetwork.
        This class internally stores both the critic and the target critic
        nets. It also handles computation of the gradients and target updates.

        Args:
            sess: A Tensorflow session to use.
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        K.set_session(sess)

        self.critic, self.state_input, self.action_input, self.vars = create_critic_network(state_size, action_size, learning_rate)
        self.critic_target, _, _, self.target_vars = create_critic_network(state_size, action_size, learning_rate)

        self.critic_target.set_weights(self.critic.get_weights())

        self.tau = tau

        self.grads = tf.gradients(self.critic.output, self.action_input)

        self.updateOp = [tf.assign(target, tau*orig + (1-tau)*target) for orig,target in zip(self.vars, self.target_vars)]


        self.sess = sess
        self.sess.run(tf.initialize_all_variables())


    def gradients(self, states, actions):
        """Computes dQ(s, a) / da.
        Note that tf.gradients returns a list storing a single gradient tensor,
        so we return that gradient, rather than the singleton list.

        Args:
            states: a batched numpy array storing the state.
            actions: a batched numpy array storing the actions.
        Returns:
            grads: a batched numpy array storing the gradients.
        """
        # QUESTIONS: How to pass in states and actions?
        out = self.sess.run(self.grads, 
            feed_dict={self.state_input: states, self.action_input: actions})

        return out[0]

        
    def update_target(self):
        """Updates the target net using an update rate of tau."""
        orig_weights = self.critic.get_weights()
        targ_weights = self.critic_target.get_weights()

        for i in range(len(targ_weights)):
            targ_weights[i] = self.tau * orig_weights[i] + (1-self.tau) * targ_weights[i]

        self.critic_target.set_weights(targ_weights)

    # def update_target(self):

    #     self.sess.run(self.updateOp)

if __name__ == '__main__':
    m,s,a,_ = create_critic_network(1, 2, 0.1)


