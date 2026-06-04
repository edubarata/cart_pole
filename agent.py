import numpy as np
import tensorflow as tf
from tensorflow import keras
from replay_buffer import ReplayBuffer


def build_q_network(state_dim: int, action_dim: int) -> keras.Model:
    return keras.Sequential([
        keras.layers.Dense(64,  activation="relu", input_shape=(state_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64,  activation="relu"),
        keras.layers.Dense(action_dim, activation="linear"),
    ])

class DQNAgent:
    def __init__(
        self,
        state_dim       : int   = 4,
        action_dim      : int   = 2,
        lr              : float = 1e-3,
        gamma           : float = 0.99,
        epsilon         : float = 1.0,
        epsilon_min     : float = 0.01,
        epsilon_decay   : float = 0.995,
        batch_size      : int   = 64,
        buffer_size     : int   = 50_000,
        target_update   : int   = 10,     # episódios entre hard updates
    ):
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.train_step    = 0

        # Rede online (treinada a cada passo) e rede alvo (atualizada periodicamente)
        self.q_net        = build_q_network(state_dim, action_dim)
        self.target_q_net = build_q_network(state_dim, action_dim)
        self._sync_target()

        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn   = keras.losses.MeanSquaredError()
        self.buffer    = ReplayBuffer(buffer_size)

    # ------------------------------------------------------------------ #
    # Ação                                                                 #
    # ------------------------------------------------------------------ #
    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy: explora aleatoriamente ou usa a rede Q."""
        if np.random.rand() < self.epsilon:
            random_act = np.random.randint(self.action_dim)
            return random_act, 'random'
        ia_act_raw = self.q_net(state[np.newaxis], training=False)
        ia_act = int(tf.argmax(ia_act_raw[0]))

        return ia_act, 'agent'

    # ------------------------------------------------------------------ #
    # Aprendizado                                                          #
    # ------------------------------------------------------------------ #
    def learn(self) -> float | None:
        """Amostra um mini-batch e faz um passo de gradient descent."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        # Alvo de Bellman: r + γ · max_a Q_target(s', a)  (para não-terminais)
        next_q     = self.target_q_net(next_states, training=False)
        max_next_q = tf.reduce_max(next_q, axis=1)
        targets    = rewards + self.gamma * max_next_q * (1.0 - dones)

        with tf.GradientTape() as tape:
            q_values  = self.q_net(states, training=True)
            # Seleciona apenas o Q-value da ação tomada
            indices   = tf.stack(
                [tf.range(self.batch_size), actions], axis=1
            )
            q_taken   = tf.gather_nd(q_values, indices)
            loss      = self.loss_fn(targets, q_taken)

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.q_net.trainable_variables)
        )
        return float(loss)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def maybe_update_target(self, episode: int):
        if episode % self.target_update == 0:
            self._sync_target()

    def _sync_target(self):
        self.target_q_net.set_weights(self.q_net.get_weights())

    # ------------------------------------------------------------------ #
    # Persistência                                                         #
    # ------------------------------------------------------------------ #
    def save(self, path: str = "checkpoints/dqn_cartpole"):
        self.q_net.save_weights(f"{path}.weights.h5")

    def load(self, path: str = "checkpoints/dqn_cartpole"):
        self.q_net.load_weights(f"{path}.weights.h5")
        self._sync_target()