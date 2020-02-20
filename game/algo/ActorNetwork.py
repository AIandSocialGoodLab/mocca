import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model

from tensorflow.keras.optimizers import Adam

import tensorflow.keras.backend as K

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400


def create_actor_network(state_size, action_size):
    """Creates an actor network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
    """
    # raise NotImplementedError

    state_input = Input(shape=[state_size])

    #---------- relu -------
    # model = tf.keras.Sequential([
    #   state_input,
    #   tf.keras.layers.Dense(HIDDEN1_UNITS, activation=tf.nn.relu), 
    #   tf.keras.layers.Dense(HIDDEN2_UNITS, activation=tf.nn.relu),
    #   tf.keras.layers.Dense(action_size, activation=tf.nn.tanh)
    # ])

    # -------- tanh --------
    model = tf.keras.Sequential([
      state_input,
      tf.keras.layers.Dense(HIDDEN1_UNITS, activation=tf.nn.tanh), 
      tf.keras.layers.Dense(HIDDEN2_UNITS, activation=tf.nn.tanh),
      tf.keras.layers.Dense(action_size, activation=tf.nn.tanh)
    ])


    # model.compile(loss="mse", optimizer=Adam(lr=learning_rate))

    return model, state_input, model.trainable_weights
    


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate):
        """Initialize the ActorNetwork.
        This class internally stores both the actor and the target actor nets.
        It also handles training the actor and updating the target net.

        Args:
            sess: A Tensorflow session to use.
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        K.set_session(sess)

        self.actor, self.state, self.vars = create_actor_network(state_size, action_size)

        self.actor_target, _, self.target_vars = create_actor_network(state_size, action_size)
        self.actor_target.set_weights(self.actor_target.get_weights())

        self.batch_size = batch_size
        self.tau = tau

        self.updateOp = [tf.assign(target, tau*orig + (1-tau)*target) for orig,target in zip(self.vars, self.target_vars)]

        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.actor.output, self.vars, -self.action_gradient)
        grads = zip(self.params_grad, self.actor.weights)

        self.optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)

        self.learning_rate = learning_rate

        self.sess = sess
        self.sess.run(tf.initialize_all_variables())


    # def train(self, states, action_grads):
    #     """Updates the actor by applying dQ(s, a) / da.

    #     Args:
    #         states: a batched numpy array storing the state.
    #         action_grads: a batched numpy array storing the
    #             gradients dQ(s, a) / da.
    #     """
    #     grad = tf.gradients(self.actor.output, self.actor.trainable_weights, -action_grads)
    #     self.optimizer.apply_gradients(zip(grad, self.actor.variables))

    def train(self, states, action_grads):

        self.sess.run(self.optimizer, feed_dict={
                self.state: states,
                self.action_gradient: action_grads
            })


    def update_target(self):
        """Updates the target net using an update rate of tau."""
        # raise NotImplementedError

        orig_weights = self.actor.get_weights()
        targ_weights = self.actor_target.get_weights()

        for i in range(len(targ_weights)):
            targ_weights[i] = self.tau * orig_weights[i] + (1-self.tau) * targ_weights[i]

        self.actor_target.set_weights(targ_weights)

    # def update_target(self):

    #     self.sess.run(self.updateOp)





if __name__ == '__main__':
    # state_input = Input(shape=[5])
    m,i,_ = create_actor_network(5, 5)










