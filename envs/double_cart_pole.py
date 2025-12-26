from gymnasium.error import DependencyNotInstalled

import math
import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding
from scipy.integrate import ode
import numpy as np

sin = np.sin
cos = np.cos

# to use this environment, replace the existing cartpole environment from gym
class DoubleCartPole(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"],
                 "render_fps": 50}

    def __init__(self, render_mode="rgb_array"):
        self.g = 9.81 # gravity constant
        self.m0 = 1.0 # mass of cart
        self.m1 = 0.5 # mass of pole 1
        self.m2 = 0.5 # mass of pole 2
        self.L1 = 1 # length of pole 1
        self.L2 = 1 # length of pole 2
        self.l1 = self.L1/2 # distance from pivot point to center of mass
        self.l2 = self.L2/2 # distance from pivot point to center of mass
        self.I1 = self.m1*(self.L1^2)/12 # moment of inertia of pole 1 w.r.t its center of mass
        self.I2 = self.m2*(self.L2^2)/12 # moment of inertia of pole 2 w.r.t its center of mass
        
        self.tau = 0.02  # seconds between state updates
        self.counter = 0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 100000 * 2 * math.pi / 360
        self.x_threshold = 2.4


        self.render_mode = render_mode

        self.screen_width = 1000
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.state = None

        self.steps_beyond_terminated = None

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(-1,1,(6,), float)
        # Initialize the relevant attributes
        self._configure()

    def _configure(self, display=None):
        self.display = display 

    def step(self, action):
        state = self.state
        x = state.item(0)
        theta = state.item(1)
        phi = state.item(2)
        x_dot = state.item(3)
        theta_dot = state.item(4)
        phi_dot = state.item(5)

        #u = force on cart
        u = action
        self.counter += 1
        
        # (state_dot = func(state))
        def func(t, state, u):
            x = state.item(0)
            theta = state.item(1)
            phi = state.item(2)
            x_dot = state.item(3)
            theta_dot = state.item(4)
            phi_dot = state.item(5)
            state = np.matrix([[x],[theta],[phi],[x_dot],[theta_dot],[phi_dot]]) 
            
            #Taken from https://digitalrepository.unm.edu/cgi/viewcontent.cgi?article=1131&context=math_etds pg 17-20
            d1 = self.m0 + self.m1 + self.m2
            d2 = self.m1*self.l1 + self.m2*self.L1
            d3 = self.m2*self.l2
            d4 = self.m1*pow(self.l1,2) + self.m2*pow(self.L1,2) + self.I1
            d5 = self.m2*self.L1*self.l2
            d6 = self.m2*pow(self.l2,2) + self.I2
            
            D = np.matrix([[d1, d2*cos(theta), d3*cos(phi)], 
                    [d2*cos(theta), d4, d5*cos(theta-phi)],
                    [d3*cos(phi), d5*cos(theta-phi), d6]])
            
            C = np.matrix([[0, -d2*sin(theta)*theta_dot, -d3*sin(phi)*phi_dot],
                    [0, 0, d5*sin(theta-phi)*phi_dot],
                    [0, -d5*sin(theta-phi)*theta_dot, 0]])

            g1 = (self.m1*self.l1 + self.m2*self.L1)*self.g 
            g2 = self.m2*self.l2*self.g    
                    
            G = np.matrix([[0], [-g1*sin(theta)], [-g2*sin(phi)]])
            
            H  = np.matrix([[1],[0],[0]])

            #Identity matrix
            I = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            #0 matrix (row, col) 
            #3x3 matrix
            O_3_3 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            #3x1 matrix
            O_3_1 = np.matrix([[0], [0], [0]])
            
            A = np.bmat([[O_3_3, I],[O_3_3, -np.linalg.inv(D)*C]])
            B = np.bmat([[O_3_1],[np.linalg.inv(D)*H]])
            L = np.bmat([[O_3_1],[-np.linalg.inv(D)*G]])
            state_dot = (A * state) + (B * u) + L  
            return state_dot
        
        solver = ode(func) 
        solver.set_integrator("dop853") # (Runge-Kutta)
        solver.set_f_params(u)

        t0 = 0
        state0 = state
        solver.set_initial_value(state0, t0)
        solver.integrate(self.tau)
        state = solver.y
        
        self.state = state
        
        #limitation for action space
        #Change Theta >10000000 and theta<-100000000 to simulate normal DPIC
        
        terminated =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta > 90*2*np.pi/360 \
                or theta < -90*2*np.pi/360 \
                or phi > 90*2*np.pi/360 \
                or phi < -90*2*np.pi/360 
        truncated = bool(self.counter >= 500)

        done = bool(terminated or truncated)
        if not terminated:
            reward = (
                1/3*math.pow(2+math.cos(theta),2)
                + 1/3*math.pow(2+math.cos(phi),2)
                - 0.2*math.pow(math.cos(theta_dot), 2)
                - 0.2*math.pow(math.cos(phi_dot), 2)
                - 0.5*math.pow(x,2) 
                + np.sign(math.cos(theta))
                + np.sign(math.cos(phi))
            )
        else:
            reward = -100
             
        state = state.squeeze()
        return self.state, reward, done, done, {}
    
    def reward(objective):
        if objective == "stabilize":
            reward = 1

    
    def reset(self, seed, options):
        self.state = np.array([[0],[np.random.uniform(-0.1,0.1)],[0],[0],[0],[0]]).squeeze()
        self.counter = 0
        return self.state, {}
     
    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            if self.clock is None:
                self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width/world_width
        carty = 300 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.8
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None
        
        x = self.state.item(0)  # cart position 
        theta = self.state.item(1) # pole 1 angle
        phi = self.state.item(2) # pole 2 angle
        x_dot = self.state.item(3) # cart acceleration
        theta_dot = self.state.item(4) # pole 1 angular acceleration
        phi_dot = self.state.item(5) # pole 2 angular acceleration 
        


        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 200  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_one_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-theta)
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_one_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_one_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_one_coords, (202, 152, 101))
        
        pole_one_tip = (np.array(pole_one_coords[1]) +  np.array(pole_one_coords[2])) /2
        pole_two_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-(theta+phi))
            coord += pole_one_tip
            pole_two_coords.append(coord)
        pole_two_coords_=pole_two_coords    
        # pole_two_coords = list(map(tuple, np.array(pole_two_coords + np.array(pole_one_coords))))
        gfxdraw.aapolygon(self.surf, pole_two_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_two_coords, (202, 152, 101))
        
        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def normalize_angle(angle):
        """
        3*pi gives -pi, 4*pi gives 0 etc, etc. (returns the negative difference
        from the closest multiple of 2*pi)
        """
        normalized_angle = abs(angle)
        normalized_angle = normalized_angle % (2*np.pi)
        if normalized_angle > np.pi:
            normalized_angle = normalized_angle - 2*np.pi
        normalized_angle = abs(normalized_angle)
        return normalized_angle

