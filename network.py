import os
import pdb
import tensorflow as tf
import numpy as np 

class A2CContinuous(object):
    """Advantage Actor Critic for continuous action spaces."""
    def __init__(self, config):
    
        self.config = config       
        #pdb.set_trace()
        self.build_networks()
        #pdb.set_trace()
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)
        if self.config["save_model"]:
                tf.add_to_collection("action", self.action)
                tf.add_to_collection("states", self.states)
                self.saver = tf.train.Saver()
        self.rewards = tf.placeholder("float", name="Rewards")
        self.episode_lengths = tf.placeholder("float", name="Episode_lengths")
        summary_actor_loss = tf.summary.scalar("Actor_loss", self.summary_actor_loss)
        summary_critic_loss = tf.summary.scalar("Critic_loss", self.summary_critic_loss)
        summary_rewards = tf.summary.scalar("Rewards", self.rewards)
        summary_episode_lengths = tf.summary.scalar("Episode_lengths", self.episode_lengths)
        #self.summary_op = tf.summary.merge([summary_actor_loss, summary_critic_loss, summary_rewards, summary_episode_lengths])
        #self.writer = tf.summary.FileWriter(os.path.join(self.monitor_path, "summaries"), self.session.graph)

    def get_critic_value(self, state):
        return self.session.run([self.critic_value], feed_dict={self.states: state})[0].flatten()

    def choose_action(self, state):
        """Choose an action."""
        mu = self.session.run([self.action_mu], feed_dict={self.states: [state]})[0]
        sigma = self.session.run([self.action_sigma], feed_dict={self.states: [state]})[0]
        action_sess =  self.session.run([self.action], feed_dict={self.states: [state]})[0]
        #return self.session.run([self.action], feed_dict={self.states: [state]})[0]
        return mu,sigma,action_sess

    def build_networks(self):
        self.states = tf.placeholder(tf.float32, [None, 13], name="states")
        self.actions_taken = tf.placeholder(tf.float32, name="actions_taken")
        self.critic_feedback = tf.placeholder(tf.float32, name="critic_feedback")
        self.critic_rewards = tf.placeholder(tf.float32, name="critic_rewards")

        # Actor network
        with tf.variable_scope("actor"):
            op1 = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.states, 0),
                num_outputs=64,
                activation_fn=tf.tanh,
                weights_initializer=tf.constant_initializer(0.0))

            mu = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(op1, 0),
                num_outputs=1,
                activation_fn=tf.tanh,
                weights_initializer=tf.constant_initializer(0.0))
            mu = tf.squeeze(mu, name="mu")

            op2 = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.states, 0),
                num_outputs=64,
                activation_fn=tf.tanh,
                weights_initializer=tf.constant_initializer(0.0))

            sigma = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(op2, 0),
                num_outputs=1,
                activation_fn=tf.tanh,
                weights_initializer=tf.constant_initializer(0.0))
            sigma = tf.squeeze(sigma)
            sigma = tf.add(tf.nn.softplus(sigma), 1e-5, name="sigma")
            self.normal_dist = tf.contrib.distributions.Normal(mu, sigma)
            self.action = self.normal_dist.sample(1)
            self.action_mu = mu
            self.action_sigma = sigma

            #self.action = tf.clip_by_value(self.action, self.action_space.low[0], self.action_space.high[0], name="action")

            # Loss and train op
            self.loss = -self.normal_dist.log_prob(tf.squeeze(self.actions_taken)) * (self.critic_rewards - self.critic_feedback)
            # Add cross entropy cost to encourage exploration
            self.loss -= 1e-1 * self.normal_dist.entropy()
            #self.loss -=  self.normal_dist.entropy()
            self.summary_actor_loss = tf.reduce_mean(self.loss)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config["actor_learning_rate"])
            self.actor_train = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

        with tf.variable_scope("critic"):
            self.critic_target = tf.placeholder("float", name="critic_target")

            # Critic network
            critic_L1 = tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.config["critic_n_hidden"],
                activation_fn=tf.tanh,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.constant_initializer(0.0))


            self.critic_value = tf.contrib.layers.fully_connected(
                inputs=critic_L1,
                num_outputs=1,
                activation_fn=tf.tanh,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.constant_initializer(0.0))

            critic_loss = tf.reduce_mean(tf.squared_difference(self.critic_target, self.critic_value))
            self.summary_critic_loss = critic_loss
            critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.config["critic_learning_rate"])
            self.critic_train = critic_optimizer.minimize(critic_loss, global_step=tf.contrib.framework.get_global_step())

    def learner(self,state,reward,action):
        returns = reward
        state1 = np.reshape(state,(1,13))
        qw_new = self.get_critic_value(state1)
        #all_state = state
        #all_action = action
        results = self.session.run([self.critic_train, self.actor_train], feed_dict = {
            self.states: state,
            self.critic_target:returns,
            self.states: state1,
            self.actions_taken: action,
            self.critic_feedback: qw_new,
            self.critic_rewards: returns,
            #self.rewards: np.mean(episode_rewards),
            #self.episode_lengths: np.mean(episode_lengths)
        })

    