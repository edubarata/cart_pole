import numpy as np
import math
import pygame


class env_cart:
    def __init__(self):
        self.state = [0.00, 0.00, 0.00, 0.00]
        self.reward = 0
        self.terminated = False
        self.truncated = False
        

    def reset(self):
        self.state = [0.00, 0.00, 0.00, 0.00]
        return self.state, None

    def step(self, action):
        # 
        return self.state, self.reward, self.terminated, self.truncated, None

    def close(self):
        pass




class CartPoleCustom:
    def __init__(
        self,
        x_threshold         : float = 2.4,
        theta_threshold_deg : float = 12.0,
        max_steps           : int   = 500,
        sutton_barto_reward : bool  = False,
        gravity             : float = 10.8,
        masscart            : float = 1.0,
        masspole            : float = 0.1,
        length              : float = 0.5,
        force_mag           : float = 10.0,
        tau                 : float = 0.02,
        render_mode         : str   = None,
    ):
        # Parâmetros físicos
        self.gravity    = gravity
        self.masscart   = masscart
        self.masspole   = masspole
        self.total_mass = masscart + masspole
        self.length     = length           # metade do comprimento do pêndulo
        self.polemass_length = masspole * length
        self.force_mag  = force_mag
        self.tau        = tau              # intervalo de tempo da simulação (s)

        # Limites de término
        self.x_threshold             = x_threshold
        self.theta_threshold_radians = theta_threshold_deg * math.pi / 180.0
        self.max_steps               = max_steps
        self.sutton_barto_reward     = sutton_barto_reward

        # Estado interno
        self.state                   = None
        self.current_step            = 0
        self.steps_beyond_terminated = None

        self.render_mode             = render_mode
        self.screen                  = None
        self.clock                   = None

    def render(self):
        SCREEN_W = 800
        SCREEN_H = 400
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("CartPole")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((255, 255, 255))
        x, _, theta, _ = self.state

        # Chão
        ground_y = SCREEN_H * 2 // 3
        pygame.draw.line(self.screen, (0, 0, 0), (0, ground_y), (SCREEN_W, ground_y), 2)

        # Carrinho
        cart_screen_x = int((x + self.x_threshold + 0.5) * (SCREEN_W / (2 * self.x_threshold + 1)))
        cart_w, cart_h = 80, 40
        cart_rect = pygame.Rect(
            cart_screen_x - cart_w // 2,
            ground_y - cart_h,
            cart_w, cart_h
        )
        pygame.draw.rect(self.screen, (70, 130, 180), cart_rect)

        # Pêndulo
        scale      = SCREEN_W / (2 * self.x_threshold + 1)
        pole_len   = self.length * 2 * scale
        pole_end_x = int(cart_screen_x + pole_len * math.sin(theta))
        pole_end_y = int(ground_y - cart_h - pole_len * math.cos(theta))
        pygame.draw.line(self.screen, (178, 34, 34),
                         (cart_screen_x, ground_y - cart_h),
                         (pole_end_x, pole_end_y), 6)
        pygame.draw.circle(self.screen, (178, 34, 34), (pole_end_x, pole_end_y), 10)

        pygame.display.flip()
        self.clock.tick(60)

    def reset(self):
        """Inicializa o estado com valores aleatórios pequenos."""
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.current_step            = 0
        self.steps_beyond_terminated = None
        return self.state.copy(), {}

    def step(self, action: int):
        assert action in (0, 1), "Ação deve ser 0 (esquerda) ou 1 (direita)"
        assert self.state is not None, "Chame reset() antes de step()"

        x, x_dot, theta, theta_dot = self.state

        # Força aplicada ao carrinho
        force = self.force_mag if action == 1 else -self.force_mag

        # ── Equações do movimento (dinâmica do pêndulo invertido) ──────
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Aceleração angular do pêndulo
        temp = (
            force + self.polemass_length * theta_dot**2 * sin_theta
        ) / self.total_mass

        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * cos_theta**2 / self.total_mass)
        )

        # Aceleração linear do carrinho
        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass

        # ── Integração de Euler ────────────────────────────────────────
        x         = x         + self.tau * x_dot
        x_dot     = x_dot     + self.tau * x_acc
        theta     = theta     + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.current_step += 1

        # ── Condições de término ───────────────────────────────────────
        terminated = bool(                         # terminated == perdeu
            x         < -self.x_threshold
            or x      >  self.x_threshold
            or theta  < -self.theta_threshold_radians
            or theta  >  self.theta_threshold_radians
        )

        truncated = bool(                                              # truncated == atingiu o limite de passos
            not terminated and self.current_step >= self.max_steps
        )

        # ── Recompensa ─────────────────────────────────────────────────
        if self.sutton_barto_reward:
            # Padrão Sutton & Barto: 0 normal, -1 na falha
            reward = -1.0 if terminated else 0.0
        else:
            # Padrão Gym: +1 em todo passo (incluindo o último)
            reward = 1.0

        self.render()
        return self.state.copy(), reward, terminated, truncated, {}

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

