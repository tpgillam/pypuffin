''' Parameter-exploring Policy Gradients, implemented with tensorflow and applied to CartPole '''

from abc import ABCMeta, abstractmethod

import gym
import numpy
import tensorflow as tf


class RLModel(metaclass=ABCMeta):
    ''' An abstract representation of a reinforcement learning model.

        This is based around the concept of an 'episode'. Any episode is assumed to be independent of any others.
        Within an episode the entire recorded history (via the 'record' method) is usable in the 'get_action'
        method, which will suggest an action to maximise the overall reward.
    '''

    def __init__(self):
        self._episode_index = -1  # So that after initialising for the first time, index == 0
        self._reset()

    def _reset(self):
        ''' Reset all internal state '''
        # We always expect to have N actions, rewards, and observations. There is additionally an initial observation
        # which we store separately
        self._actions = []
        self._rewards = []
        self._observations = []
        self._initial_observation = None

    def new_episode(self, observation):
        ''' Indicate that a new episode is beginning, and specify the initial state. '''
        self._episode_index += 1
        self._reset()
        self._initial_observation = observation

    def record(self, action, reward, observation):
        ''' Record an action, along with the resultant reward and observation, to use for subsequent training and
            prediction.
        '''
        self._actions.append(action)
        self._rewards.append(reward)
        self._observations.append(observation)

    @property
    def total_reward(self):
        ''' Return the total reward seen in this episode '''
        return sum(self._rewards)

    @abstractmethod
    def get_action(self, observation):
        ''' Get the model's suggested action given this observation, and potentially any history from the current
            episode. One is not necessarily required to *take* this action.
        '''

    @abstractmethod
    def train(self):
        ''' Perform training with the latest episode '''


def tf_dot(vector, matrix):
    ''' Helper function that takes inputs of shape (N,), (N, M) and does a dot product to give output
        of shape (M,)
    '''
    return tf.reshape(tf.matmul(tf.reshape(vector, (1, -1)), matrix), (-1,))


class PGPEModel(RLModel):
    ''' Model to learn the CartPole-v0, implementing:

        http://kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Neural-Networks-2010-Sehnke_%5b0%5d.pdf
    '''

    def __init__(self, learning_rate=0.002):
        super().__init__()

        # Hyperparameters
        self._learning_rate = learning_rate

        self._observed_total_rewards = []

        # TODO unhardcode dimensionalities
        num_observations = 4
        # num_observations = 2
        num_actions = 2

        # We only ever consider one observation at a time, and map the function to it
        observation = tf.placeholder(tf.float32, shape=(num_observations,), name='observation')

        values = tf.placeholder(tf.float32, shape=(None,), name='values')

        # Scalar baseline -- this represents a moving average over values
        baseline = tf.placeholder(tf.float32, shape=(), name='baseline')

        with tf.name_scope('model'):
            action, parameters, mu, sigma = self._get_model(observation, num_actions)

        update_op = self._get_update_op(parameters, mu, sigma, baseline, values)

        # Store variables on the instance that we need access to later
        self._observation = observation
        self._values = values
        self._baseline = baseline
        self._action = action
        self._update_op = update_op

        # # FIXME temp
        # self._mu = mu
        # self._sigma = sigma

    def _get_model(self, observation, num_actions):
        ''' Build a model mapping from the given observation to an action out of num_actions
            Returns a tensor of shape (num_parameters,), as well as a tensor of shape (None,) representing
            the action to take.
        '''
        # TODO this is just a linear model, something better!
        parameters = tf.Variable(tf.ones(shape=(observation.shape[0], num_actions)), name='parameters')
        mu = tf.Variable(tf.zeros_like(parameters), name='mu')
        sigma = tf.Variable(1000 * tf.ones_like(parameters), name='sigma')

        # observation is shape (4,), parameters is shape (4, 2).
        # In order to do matrix multiplication first argument converted to shape (1, 4)
        action = tf.argmax(tf.nn.softmax(tf_dot(observation, parameters)))

        return action, parameters, mu, sigma

    def _get_update_op(self, parameters, mu, sigma, baseline, values):
        ''' Tensorflow operation to update the state given an update '''
        # TODO assume that we only have one value
        value = tf.reshape(values, ())

        # The algorithm from the paper
        T = parameters - mu
        S = (T * T - sigma * sigma) / sigma
        r = value - baseline

        # TODO should work the same way if we do e.g. mu = tf.assign_add(mu, ...) -- mu ought to then
        # refer to the state of mu after the addition.
        # Perform updates
        updates = tf.group(mu.assign_add(self._learning_rate * r * T),
                           sigma.assign_add(self._learning_rate * r * S))

        with tf.control_dependencies((updates,)):
            # Draw a new value for the parameters from updated mu & nu
            param_op = parameters.assign(tf.contrib.distributions.Normal(mu, sigma).sample())

        # Introduce dependency on all updates finishing
        return param_op

    def get_action(self, observation):
        ''' Determine the action to execute given the observation '''
        return self._action.eval(feed_dict={self._observation: observation})

    def train(self):
        ''' Do training '''
        self._observed_total_rewards.append(self.total_reward)

        # Compute an exponentially weighted average with given window. This definitely performs better than the
        # non-rolling version
        window = 10
        gamma = 1 - 1 / window
        weights = numpy.flip(gamma ** numpy.arange(len(self._observed_total_rewards)), axis=0)
        baseline = numpy.average(self._observed_total_rewards, weights=weights)

        # Here we are effectively setting N = 1, that is we run only one episode before
        # doing our parameter updates
        feed_dict = {self._values: [self.total_reward],
                     self._baseline: baseline}

        self._update_op.eval(feed_dict=feed_dict)


def main():
    env = gym.make('CartPole-v1')
    # env = gym.make('MountainCar-v0')
    env.seed(42)
    tf.set_random_seed(42)

    model = PGPEModel()

    # The number of episodes to use for training
    num_episodes = 1000

    # Maximum length of a training epoch
    # TODO do we create a bias by having this maximum length? It stands to reason that we *should* penalise
    # early exit in some way (since we get lower reward), but not finishing due to exhausting the number of steps.
    max_num_steps = 1000

    with tf.Session():
        tf.global_variables_initializer().run()

        for episode in range(num_episodes):
            observation = env.reset()
            model.new_episode(observation)

            for _ in range(max_num_steps):
                # env.render()
                action = model.get_action(observation)
                observation, reward, done, _ = env.step(action)
                model.record(action, reward, observation)

                if done:
                    break

            print(f'{episode}:  finished with reward {model.total_reward}')

            # Now train the model with all available data
            model.train()


if __name__ == '__main__':
    main()
