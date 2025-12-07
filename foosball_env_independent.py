from random import seed
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os
import time

class FoosballEnv(gym.Env):
    """
    Two-agent symmetric foosball environment for RL training.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        render_mode='human',
        curriculum_level=1,
        debug_mode=False,
        player_id=1,
        opponent_model=None,
        goal_debug_mode=False,
        steps_per_episode=4000,
        config=None,
        obs_mode="full",      # <--- NEW
    ):
        super(FoosballEnv, self).__init__()

        self.config = config  # can be None
        self.obs_mode = obs_mode  # "full" or "ball_self"

        # -------- FIXED REWARD CONFIG (no YAML, just defaults) --------
        self.reward_cfg = {
            "ball_velocity_scale":        0.000001,
            "goal_distance_penalty":      0.00005,
            "player_ball_distance_scale": 0.000001,
            "ball_movement_scale":        0.00001,
            "contact_reward":             1.0,
            "action_smoothness_penalty":  0.0,
            "rod_angle_penalty_scale":    0.000001,
            "midfield_reward_weight":     0.0001,
            "goal_reward":                30.0,
            "concede_penalty":            15.0,
            "stagnation_slide_steps":     100,
            "stagnation_slide_penalty":   0.0001,
            "stagnation_spin_steps":      100,
            "stagnation_spin_penalty":    0.0001,
            "stuck_penalty_threshold":    300,
            "stuck_penalty_per_step":     0.005,
            "inactivity_threshold":       0.001,
            "inactivity_penalty":         0.001,
        }

        self.render_mode = render_mode
        self.goal_debug_mode = goal_debug_mode
        if self.goal_debug_mode:
            self.render_mode = 'human'

        self.curriculum_level = curriculum_level
        self.debug_mode = debug_mode
        self.player_id = player_id
        self.opponent_model = opponent_model
        self.goals_this_level = 0

        # -------- CONNECT TO PYBULLET --------
        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        else:
            self.client = p.connect(p.DIRECT)

        # -------- ENVIRONMENT / PHYSICS DEFAULTS (no YAML) --------
        self.max_stuck_steps = 5000
        self.max_vel = 1.5
        self.max_force = 1.0
        self.goal_line_x_1 = -0.75
        self.goal_line_x_2 = 0.75
        ball_radius = 0.025
        ball_mass   = 0.025

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(numSubSteps=4)

        # -------- GYM SPACES --------
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

        if self.obs_mode == "full":
            # ball (3 pos + 3 vel) + all 16 joint pos + 16 joint vel = 38
            obs_space_dim = 3 + 3 + 16 + 16
        elif self.obs_mode == "ball_self":
            # ball (3 pos + 3 vel) + ONLY THIS PLAYER'S 8 joint pos + 8 joint vel = 22
            obs_space_dim = 3 + 3 + 8 + 8
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_space_dim,), dtype=np.float32
        )
        # -------- EPISODE TRACKING --------
        self.ball_stuck_counter = 0
        self.episode_step_count = 0
        self.max_episode_steps = steps_per_episode
        self.previous_action = np.zeros(self.action_space.shape)
        self.previous_ball_dist = 0

        # -------- PLANE & TABLE --------
        self.plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])
        urdf_path = os.path.join(os.path.dirname(__file__), 'foosball.urdf')
        self.table_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0.5],
            useFixedBase=True
        )
        p.changeVisualShape(self.table_id, -1, rgbaColor=[0.1, 0.4, 0.1, 1])

        self._setup_camera()

        # -------- JOINT / LINK CONTAINERS --------
        self.team1_slide_joints = []
        self.team1_rev_joints = []
        self.team2_slide_joints = []
        self.team2_rev_joints = []
        self.team1_player_links = []
        self.team2_player_links = []
        self.team1_players_by_rod = {}
        self.team2_players_by_rod = {}
        self.joint_name_to_id = {}
        self.joint_limits = {}
        self.goal_link_a = None
        self.goal_link_b = None

        # PBRS-related
        self.gamma = 0.99              # for PBRS (same as PPO discount)
        self.use_pbrs_reward = True    # toggle this on/off
        self.last_potential = None
        self.goal_half_width = 0.1     # tune to your table

        # -------- PARSE JOINTS / LINKS --------
        self._parse_joints_and_links()

        # -------- CREATE BALL --------
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=ball_radius,
            rgbaColor=[1, 1, 1, 1]
        )
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=ball_radius
        )
        self.ball_id = p.createMultiBody(
            baseMass=ball_mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, 0.55]
        )

        # Ball dynamics (fixed)
        p.changeDynamics(
            self.ball_id,
            -1,
            restitution=0.8,
            rollingFriction=0.001,
            spinningFriction=0.001,
            lateralFriction=0.01,
        )

        # Rod dynamics (fixed)
        for joint_id in (
            self.team1_slide_joints
            + self.team1_rev_joints
            + self.team2_slide_joints
            + self.team2_rev_joints
        ):
            p.changeDynamics(
                self.table_id,
                joint_id,
                linearDamping=1.0,
                angularDamping=1.0,
                restitution=0.7,
            )

        # Joint lists
        self.team1_joints = self.team1_slide_joints + self.team1_rev_joints
        self.team2_joints = self.team2_slide_joints + self.team2_rev_joints
        self.all_joints   = self.team1_joints + self.team2_joints

        # Debug sliders for goal lines
        if self.goal_debug_mode:
            self.goal_line_slider_1 = p.addUserDebugParameter(
                "Goal Line 1", -1.0, 1.0, self.goal_line_x_1
            )
            self.goal_line_slider_2 = p.addUserDebugParameter(
                "Goal Line 2", -1.0, 1.0, self.goal_line_x_2
            )

        if self.debug_mode and self.render_mode == 'human':
            self._add_debug_sliders()


    # ----------------- Extra helper for PBRS -----------------
    def get_single_agent_obs_dim(self, obs_mode=None):
        """
        Return the observation dimension for ONE rod agent,
        depending on obs_mode.

        obs_mode:
          - 'full'      : agent sees full global observation
          - 'ball_self' : agent sees [ball(4) + self_rod(4)] = 8
        """
        if obs_mode is None:
            obs_mode = self.obs_mode

        if obs_mode == "full":
            # each agent just sees the same full env observation
            return self.observation_space.shape[0]
        elif obs_mode == "ball_self":
            # get_ball_obs: 4, self-only obs: 4 → 8
            return 8
        else:
            raise ValueError(f"Unknown obs_mode in get_single_agent_obs_dim: {obs_mode}")
    
    def _ball_potential(self):
        """
        Potential function ψ(s): high when ball close to opponent goal and near goal center line.
        """
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        x, y = ball_pos[0], ball_pos[1]

        if self.player_id == 1:
            opp_goal_x = self.goal_line_x_2
            own_goal_x = self.goal_line_x_1
        else:
            opp_goal_x = self.goal_line_x_1
            own_goal_x = self.goal_line_x_2

        field_length = abs(self.goal_line_x_2 - self.goal_line_x_1) + 1e-6
        dx = min(abs(x - opp_goal_x), field_length)
        potential_x = 1.0 - dx / field_length   # [0, 1]

        dy = abs(y)
        potential_y = max(0.0, 1.0 - dy / (self.goal_half_width + 1e-6))
        potential_y = min(potential_y, 1.0)

        return 0.5 * (potential_x + potential_y)

    def update_opponent_model(self, state_dict):
        if self.opponent_model:
            self.opponent_model.policy.load_state_dict(state_dict)

    def _parse_joints_and_links(self):
        num_joints = p.getNumJoints(self.table_id)
        team1_slide_joints_map, team1_rev_joints_map = {}, {}
        team2_slide_joints_map, team2_rev_joints_map = {}, {}
        link_name_to_index = {p.getBodyInfo(self.table_id)[0].decode('UTF-8'): -1}
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.table_id, i)
            link_name = joint_info[12].decode('UTF-8')
            link_name_to_index[link_name] = i

        for link_name, link_index in link_name_to_index.items():
            if link_name.startswith('p'):
                try:
                    rod_num = int(link_name.split('_')[0][1:])
                    team = 1 if rod_num in [1, 2, 4, 6] else 2
                    if team == 1:
                        self.team1_player_links.append(link_index)
                        if rod_num not in self.team1_players_by_rod:
                            self.team1_players_by_rod[rod_num] = []
                        self.team1_players_by_rod[rod_num].append(link_index)
                    else:
                        self.team2_player_links.append(link_index)
                        if rod_num not in self.team2_players_by_rod:
                            self.team2_players_by_rod[rod_num] = []
                        self.team2_players_by_rod[rod_num].append(link_index)
                except (ValueError, IndexError):
                    continue

        for i in range(num_joints):
            info = p.getJointInfo(self.table_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            self.joint_name_to_id[joint_name] = i
            self.joint_limits[i] = (info[8], info[9])

            if "goal_sensor_A" in joint_name:
                self.goal_link_a = i
            if "goal_sensor_B" in joint_name:
                self.goal_link_b = i
            if joint_type not in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]:
                continue

            rod_num_str = ''.join(filter(str.isdigit, joint_name))
            if not rod_num_str:
                continue
            rod_num = int(rod_num_str)

            if f"rod_{rod_num}" not in joint_name:
                continue

            team = 1 if rod_num in [1, 2, 4, 6] else 2

            if 'slide' in joint_name.lower():
                if team == 1:
                    team1_slide_joints_map[rod_num] = i
                else:
                    team2_slide_joints_map[rod_num] = i
            elif 'rotate' in joint_name.lower():
                if team == 1:
                    team1_rev_joints_map[rod_num] = i
                else:
                    team2_rev_joints_map[rod_num] = i

        self.team1_slide_joints = [team1_slide_joints_map[k] for k in sorted(team1_slide_joints_map)]
        self.team1_rev_joints   = [team1_rev_joints_map[k]   for k in sorted(team1_rev_joints_map)]
        self.team2_slide_joints = [team2_slide_joints_map[k] for k in sorted(team2_slide_joints_map)]
        self.team2_rev_joints   = [team2_rev_joints_map[k]   for k in sorted(team2_rev_joints_map)]

    def _setup_camera(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=0,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0.5],
        )

    def _add_debug_sliders(self):
        self.slider_ids = {}
        for i, joint_id in enumerate(self.team1_rev_joints):
            self.slider_ids[joint_id] = p.addUserDebugParameter(
                f"T1_Rod{i+1}_Rot", -np.pi, np.pi, 0
            )
        for i, joint_id in enumerate(self.team1_slide_joints):
            lower, upper = self.joint_limits[joint_id]
            self.slider_ids[joint_id] = p.addUserDebugParameter(
                f"T1_Rod{i+1}_Slide", lower, upper, (lower + upper) / 2
            )

    def _add_debug_sliders(self):
        self.slider_ids = {}
        for i, joint_id in enumerate(self.team1_rev_joints):
            self.slider_ids[joint_id] = p.addUserDebugParameter(f"T1_Rod{i+1}_Rot", -np.pi, np.pi, 0)
        for i, joint_id in enumerate(self.team1_slide_joints):
            lower, upper = self.joint_limits[joint_id]
            self.slider_ids[joint_id] = p.addUserDebugParameter(f"T1_Rod{i+1}_Slide", lower, upper, (lower + upper) / 2)

    def _curriculum_spawn_ball(self):
        if self.curriculum_level == 1:
            if self.player_id == 1:
                ball_x, ball_y = np.random.uniform(-0.6, 0.0), np.random.uniform(-0.3, 0.3)
                ball_vel = [np.random.uniform(-0.2, -0.1), np.random.uniform(-0.1, 0.1), 0]
            else:
                ball_x, ball_y = np.random.uniform(0.0, 0.6), np.random.uniform(-0.3, 0.3)
                ball_vel = [np.random.uniform(0.1, 0.2), np.random.uniform(-0.1, 0.1), 0]
            ball_pos = [ball_x, ball_y, 0.55]
        elif self.curriculum_level == 2:
            ball_pos = [0, 0, 0.55]
            if self.player_id == 1: ball_vel = [-1, np.random.uniform(-0.5, 0.5), 0]
            else: ball_vel = [1, np.random.uniform(-0.5, 0.5), 0]
        elif self.curriculum_level == 3:
            speed = np.random.uniform(3.0, 4.5)
            if self.player_id == 1:
                spawn_pos = np.array([np.random.uniform(-0.5, -0.4), np.random.uniform(-0.25, 0.25), 0.55])
                target_pos = np.array([self.goal_line_x_1, np.random.uniform(-0.05, 0.05), 0.55])
                direction = target_pos - spawn_pos
                direction_norm = np.linalg.norm(direction)
                if direction_norm == 0: direction_norm = 1
                ball_vel = (direction / direction_norm) * speed
                ball_pos = spawn_pos.tolist()
            else: # player_id == 2
                spawn_pos = np.array([np.random.uniform(0.4, 0.5), np.random.uniform(-0.25, 0.25), 0.55])
                target_pos = np.array([self.goal_line_x_2, 0, 0.55])
                direction = target_pos - spawn_pos
                direction_norm = np.linalg.norm(direction)
                if direction_norm == 0: direction_norm = 1
                ball_vel = (direction / direction_norm) * speed
                ball_pos = spawn_pos.tolist()
        else:
            ball_pos, ball_vel = [np.random.uniform(-0.6, 0.6), np.random.uniform(-0.3, 0.3), 0.55], [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0]
        p.resetBasePositionAndOrientation(self.ball_id, ball_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.ball_id, linearVelocity=ball_vel)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_potential = None
        self.last_ball_x = None
        if seed is not None: np.random.seed(seed)
        for joint_id in self.all_joints: p.resetJointState(self.table_id, joint_id, targetValue=0, targetVelocity=0)
        self._curriculum_spawn_ball()
        if self.curriculum_level == 1:
            for _ in range(100): # More steps to ensure rods are settled
                self._set_opponent_rods_to_90_degrees()
                p.stepSimulation()
        self.ball_stuck_counter, self.episode_step_count, self.goals_this_level = 0, 0, 0
        
        # Reset stagnation tracking
        self.last_slide_positions = np.zeros(4)
        self.slide_stagnation_counter = 0
        self.last_spin_velocities = np.zeros(4)
        self.spin_stagnation_counter = 0
        
        return self._get_obs(), {}

    def step(self, action, opponent_action=None):
        self.episode_step_count += 1
        scaled_action = self._scale_action(action, self.player_id)
        self._apply_action(scaled_action, self.player_id)

        if self.curriculum_level == 1:
            self._set_opponent_rods_to_90_degrees()
        elif self.curriculum_level == 4 and self.opponent_model:
            mirrored_obs = self._get_mirrored_obs()
            opponent_action, _ = self.opponent_model.predict(mirrored_obs, deterministic=True)
            scaled_opponent_action = self._scale_action(opponent_action, 3 - self.player_id)
            self._apply_action(scaled_opponent_action, 3 - self.player_id)
        elif opponent_action is None:
            bot_action = self._simple_bot_logic()
            self._apply_opponent_action(bot_action)
        else: # for testing
            scaled_opponent_action = self._scale_action(opponent_action, 3 - self.player_id)
            self._apply_action(scaled_opponent_action, 3 - self.player_id)
            
        p.stepSimulation()

        # If the ball is stuck, apply a gentle nudge towards the center
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_vel, _ = p.getBaseVelocity(self.ball_id)
        
        if np.linalg.norm(ball_vel) < 0.01 and self.ball_stuck_counter > 2000:
            # Apply a force towards the center of the table, proportional to the distance from the center
            force_magnitude = 0.5
            nudge_force = [-ball_pos[0] * force_magnitude, -ball_pos[1] * force_magnitude, 0]
            p.applyExternalForce(self.ball_id, -1, nudge_force, ball_pos, p.WORLD_FRAME)

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated, truncated = self._check_termination(obs)
        self.previous_action = action
        return obs, reward, terminated, truncated, {}

    def _get_mirrored_obs(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_vel, _ = p.getBaseVelocity(self.ball_id)

        # Get joint states for each team
        team1_joint_states = p.getJointStates(self.table_id, self.team1_joints)
        team2_joint_states = p.getJointStates(self.table_id, self.team2_joints)

        team1_pos = [state[0] for state in team1_joint_states]
        team1_vel = [state[1] for state in team1_joint_states]
        team2_pos = [state[0] for state in team2_joint_states]
        team2_vel = [state[1] for state in team2_joint_states]

        # Mirrored observation for the opponent (team 2)
        # The opponent sees itself as team 1, and the agent as team 2.
        mirrored_ball_pos = (-ball_pos[0], ball_pos[1], ball_pos[2])
        mirrored_ball_vel = (-ball_vel[0], ball_vel[1], ball_vel[2])
        
        # The opponent's joints are now team 1, and the agent's joints are team 2
        mirrored_joint_pos = team2_pos + team1_pos
        mirrored_joint_vel = team2_vel + team1_vel

        return np.concatenate([mirrored_ball_pos, mirrored_ball_vel, mirrored_joint_pos, mirrored_joint_vel]).astype(np.float32)

    def _set_opponent_rods_to_90_degrees(self):
        """Set opponent rods to 90 degrees."""
        revs = self.team2_rev_joints if self.player_id == 1 else self.team1_rev_joints
        for joint_id in revs:
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=np.pi/2, force=self.max_force)

    def _set_all_rods_to_90_degrees(self):
        """Set all rods to 90 degrees for a clear view."""
        all_rev_joints = self.team1_rev_joints + self.team2_rev_joints
        for joint_id in all_rev_joints:
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=np.pi/2, force=self.max_force)
        # Step simulation a few times to let rods settle
        for _ in range(50):
            p.stepSimulation()
            if self.render_mode == 'human':
                time.sleep(1./240.)

    def _simple_bot_logic(self):
        action = np.zeros(8)
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        if self.player_id == 1: mirrored_ball_pos = [-ball_pos[0], ball_pos[1], ball_pos[2]]
        else: mirrored_ball_pos = [ball_pos[0], ball_pos[1], ball_pos[2]]
        
        if self.curriculum_level == 2:
            for i in range(4):
                action[i] = np.random.uniform(-1, 1)
                action[i+4] = np.random.uniform(-np.pi, np.pi)
        elif self.curriculum_level >= 3:
            for i in range(4):
                action[i] = np.clip(-mirrored_ball_pos[1] / 0.3, -1, 1)
                action[i+4] = np.pi/2 # Keep rods vertical
        return action

    def _apply_opponent_action(self, action):
        if self.player_id == 1:
            revs, slides = self.team2_rev_joints, self.team2_slide_joints
        else:
            revs, slides = self.team1_rev_joints, self.team1_slide_joints
        for i, joint_id in enumerate(slides):
            lower, upper = self.joint_limits[joint_id]
            target_pos = lower + (action[i] + 1) / 2 * (upper - lower)
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=target_pos, force=self.max_force)
        for i, joint_id in enumerate(revs):
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=action[i+4], force=self.max_force)

    def _scale_action(self, action, player_id):
        scaled = np.zeros(8)
        if player_id == 1:
            slides, revs = self.team1_slide_joints, self.team1_rev_joints
        else:
            slides, revs = self.team2_slide_joints, self.team2_rev_joints
        for i, joint_id in enumerate(slides):
            lower, upper = self.joint_limits[joint_id]
            scaled[i] = lower + (action[i] + 1) / 2 * (upper - lower)
        for i, joint_id in enumerate(revs):
            scaled[i + 4] = action[i + 4] * 10
        return scaled

    def _apply_action(self, scaled_action, player_id):
        if player_id == 1:
            slides, revs = self.team1_slide_joints, self.team1_rev_joints
        else:
            slides, revs = self.team2_slide_joints, self.team2_rev_joints
        for i, joint_id in enumerate(slides):
            p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=scaled_action[i], maxVelocity=self.max_vel, force=self.max_force)
        for i, joint_id in enumerate(revs):
            #print(f"Applying velocity {scaled_action[i + 4]} to joint {joint_id}")
            p.setJointMotorControl2(self.table_id, joint_id, p.VELOCITY_CONTROL, targetVelocity=scaled_action[i + 4], force=self.max_force)

    def _debug_step(self, action):
        for joint_id in self.team1_rev_joints:
            if joint_id in self.slider_ids: p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=p.readUserDebugParameter(self.slider_ids[joint_id]), maxVelocity=self.max_vel, force=self.max_force)
        for joint_id in self.team1_slide_joints:
            if joint_id in self.slider_ids: p.setJointMotorControl2(self.table_id, joint_id, p.POSITION_CONTROL, targetPosition=p.readUserDebugParameter(self.slider_ids[joint_id]), maxVelocity=self.max_vel, force=self.max_force)

    def _get_obs(self):
        # Ball state
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_vel, _ = p.getBaseVelocity(self.ball_id)

        # Mirror so both players "attack +x"
        if self.player_id == 2:
            ball_pos = (-ball_pos[0], ball_pos[1], ball_pos[2])
            ball_vel = (-ball_vel[0], ball_vel[1], ball_vel[2])

        if self.obs_mode == "full":
            # ball + ALL joints (both teams)
            joint_states = p.getJointStates(self.table_id, self.all_joints)
            if joint_states is None:
                return np.zeros(self.observation_space.shape, dtype=np.float32)

            joint_pos = [state[0] for state in joint_states]
            joint_vel = [state[1] for state in joint_states]

            obs = np.concatenate([ball_pos, ball_vel, joint_pos, joint_vel]).astype(np.float32)

        elif self.obs_mode == "ball_self":
            # ball + ONLY THIS PLAYER'S joints
            if self.player_id == 1:
                own_joints = self.team1_joints
            else:
                own_joints = self.team2_joints

            joint_states = p.getJointStates(self.table_id, own_joints)
            if joint_states is None:
                return np.zeros(self.observation_space.shape, dtype=np.float32)

            joint_pos = [state[0] for state in joint_states]
            joint_vel = [state[1] for state in joint_states]

            obs = np.concatenate([ball_pos, ball_vel, joint_pos, joint_vel]).astype(np.float32)

        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")

        return obs

    def _compute_reward(self, action):
        # --- Basic state info ---
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_vel, _ = p.getBaseVelocity(self.ball_id)

        reward = 0.0
        cfg = self.reward_cfg

        # ------------------------------------------------------
        # 1. Sparse goal / own goal rewards (like teammate)
        # ------------------------------------------------------
        # "Right" side is +x, "Left" side is -x
        # For player 1 (red), scoring means ball crosses goal_line_x_2 (right side)
        # For player 2 (blue), scoring means ball crosses goal_line_x_1 (left side)
        if (self.player_id == 1 and ball_pos[0] > self.goal_line_x_2) or \
           (self.player_id == 2 and ball_pos[0] < self.goal_line_x_1):
            reward += cfg["goal_reward"]  # we scored

        if (self.player_id == 1 and ball_pos[0] < self.goal_line_x_1) or \
           (self.player_id == 2 and ball_pos[0] > self.goal_line_x_2):
            reward -= cfg["concede_penalty"]  # we conceded

        # ------------------------------------------------------
        # 2. Distance-to-ball reward: get close to the ball
        # ------------------------------------------------------
        agent_player_links = self.team1_player_links if self.player_id == 1 else self.team2_player_links
        min_dist_to_ball = float("inf")

        for link_idx in agent_player_links:
            link_state = p.getLinkState(self.table_id, link_idx)
            link_pos = np.array(link_state[0])  # CoM of the player model
            dist = np.linalg.norm(np.array(ball_pos) - link_pos)
            if dist < min_dist_to_ball:
                min_dist_to_ball = dist

        # closer → larger (1 - tanh(dist)) → in (0, 1)
        distance_reward = cfg["player_ball_distance_scale"] * (1 - np.tanh(min_dist_to_ball))
        reward += distance_reward

        # ------------------------------------------------------
        # 3. Ball velocity toward opponent goal
        # ------------------------------------------------------
        if self.player_id == 1:
            # we want ball_x velocity positive (toward +x / goal_line_x_2)
            vel_reward = ball_vel[0]
        else:
            # we want ball_x velocity negative (toward -x / goal_line_x_1)
            vel_reward = -ball_vel[0]

        reward += cfg["ball_velocity_scale"] * vel_reward

        # ------------------------------------------------------
        # 4. Contact reward: touch the ball!
        # ------------------------------------------------------
        contact_with_agent = False
        for link_idx in agent_player_links:
            if p.getContactPoints(bodyA=self.table_id, bodyB=self.ball_id, linkIndexA=link_idx):
                contact_with_agent = True
                break

        if contact_with_agent:
            reward += cfg["contact_reward"]

        # ------------------------------------------------------
        # 5. Agent stagnation penalties (slide / spin)
        # ------------------------------------------------------
        # Slide stagnation: rods not sliding along x for long
        agent_slides = self.team1_slide_joints if self.player_id == 1 else self.team2_slide_joints
        slide_states = p.getJointStates(self.table_id, agent_slides)
        current_slide_pos = np.array([st[0] for st in slide_states])

        if np.linalg.norm(current_slide_pos - self.last_slide_positions) < 1e-3:
            self.slide_stagnation_counter += 1
        else:
            self.slide_stagnation_counter = 0

        if self.slide_stagnation_counter > cfg["stagnation_slide_steps"]:
            reward -= cfg["stagnation_slide_penalty"]

        self.last_slide_positions = current_slide_pos

        # Spin stagnation: rods not changing their spin velocity
        agent_spins = self.team1_rev_joints if self.player_id == 1 else self.team2_rev_joints
        spin_states = p.getJointStates(self.table_id, agent_spins)
        current_spin_vels = np.array([st[1] for st in spin_states])

        if np.linalg.norm(current_spin_vels - self.last_spin_velocities) < 1e-2:
            self.spin_stagnation_counter += 1
        else:
            self.spin_stagnation_counter = 0

        if self.spin_stagnation_counter > cfg["stagnation_spin_steps"]:
            reward -= cfg["stagnation_spin_penalty"]

        self.last_spin_velocities = current_spin_vels

        # ------------------------------------------------------
        # 6. Ball stuck & inactivity penalty
        # ------------------------------------------------------
        ball_speed = np.linalg.norm(ball_vel[:2])

        if ball_speed < 0.01:
            self.ball_stuck_counter += 1
        else:
            self.ball_stuck_counter = 0

        if self.ball_stuck_counter > cfg["stuck_penalty_threshold"]:
            reward -= cfg["stuck_penalty_per_step"] * self.ball_stuck_counter

        if ball_speed < cfg["inactivity_threshold"]:
            reward -= cfg["inactivity_penalty"]

        # ------------------------------------------------------
        # 7. Rod angle penalty (avoid always upright)
        # ------------------------------------------------------
        current_rev_pos = np.array([st[0] for st in spin_states])
        reward -= cfg["rod_angle_penalty_scale"] * np.sum(np.abs(current_rev_pos))

        # ------------------------------------------------------
        # 8. Optional midfield shaping (currently 0)
        # ------------------------------------------------------
        mf_weight = cfg["midfield_reward_weight"]
        if mf_weight != 0.0:
            reward += mf_weight * (1 - np.tanh(abs(ball_pos[1])))

        return float(reward)


    def _check_termination(self, obs):
        ball_pos, ball_vel = obs[:3], obs[3:6]
        terminated, truncated = False, False
        if (self.player_id == 1 and (ball_pos[0] > self.goal_line_x_2 or ball_pos[0] < self.goal_line_x_1)) or \
           (self.player_id == 2 and (ball_pos[0] < self.goal_line_x_1 or ball_pos[0] > self.goal_line_x_2)):
            terminated = True
        table_aabb = p.getAABB(self.table_id)
        if not (table_aabb[0][0] - 0.1 < ball_pos[0] < table_aabb[1][0] + 0.1 and table_aabb[0][1] - 0.1 < ball_pos[1] < table_aabb[1][1] + 0.1):
            truncated = True
        if np.linalg.norm(ball_vel) < 0.001: self.ball_stuck_counter += 1
        else: self.ball_stuck_counter = 0
        if self.ball_stuck_counter > self.max_stuck_steps: truncated = True
        if self.episode_step_count > self.max_episode_steps: truncated = True
        return terminated, truncated

    # =========================
    #  SIMPLE OBSERVATION MODES
    # =========================

    def get_ball_obs(self):
        """
        Global ball-only observation from THIS player's perspective.

        Returns:
            np.ndarray of shape (4,):
            [ball_x, ball_y, ball_vx, ball_vy]
        """
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_vel, _ = p.getBaseVelocity(self.ball_id)

        # Mirror along x so both players "attack +x"
        if self.player_id == 2:
            ball_pos = (-ball_pos[0], ball_pos[1], ball_pos[2])
            ball_vel = (-ball_vel[0], ball_vel[1], ball_vel[2])

        bx, by = float(ball_pos[0]), float(ball_pos[1])
        bvx, bvy = float(ball_vel[0]), float(ball_vel[1])

        return np.array([bx, by, bvx, bvy], dtype=np.float32)

    def get_per_agent_ball_only_obs(self):
        """
        Return a list of 4 ball-only observations, one for each rod-agent.

        Each element is np.ndarray of shape (4,):
            [ball_x, ball_y, ball_vx, ball_vy]

        Right now all 4 agents see the same ball info (you can later
        customize by adding relative coordinates, etc.).
        """
        ball_obs = self.get_ball_obs()
        return [ball_obs.copy() for _ in range(4)]

    def get_per_agent_self_only_obs(self):
        """
        Return a list of 4 self-only observations (one per rod of THIS player).

        Each element obs_i is:
            [slide_pos_i, slide_vel_i, spin_pos_i, spin_vel_i]   # shape (4,)

        This does NOT include ball information – purely local rod state.
        """
        if self.player_id == 1:
            slide_joints = self.team1_slide_joints
            rev_joints   = self.team1_rev_joints
        else:
            slide_joints = self.team2_slide_joints
            rev_joints   = self.team2_rev_joints

        obs_list = []

        for slide_j, rev_j in zip(slide_joints, rev_joints):
            slide_state = p.getJointState(self.table_id, slide_j)
            rev_state   = p.getJointState(self.table_id, rev_j)

            slide_pos, slide_vel = float(slide_state[0]), float(slide_state[1])
            spin_pos,  spin_vel  = float(rev_state[0]),   float(rev_state[1])

            obs = np.array(
                [slide_pos, slide_vel, spin_pos, spin_vel],
                dtype=np.float32,
            )
            obs_list.append(obs)

        return obs_list
    def get_per_agent_obs(self):
        """
        Return a list of 4 observations, one per rod-agent of THIS player.

        Depends on self.obs_mode:

        - 'full':
            each agent gets the same global obs: shape = (obs_space_dim,)
        - 'ball_self':
            each agent i gets [ball_x, ball_y, ball_vx, ball_vy,
                               slide_pos_i, slide_vel_i, spin_pos_i, spin_vel_i]
            → shape = (8,)
        """
        if self.obs_mode == "full":
            # Use the original global observation
            full_obs = self._get_obs().astype(np.float32)
            return [full_obs.copy() for _ in range(4)]

        elif self.obs_mode == "ball_self":
            # 1) Global ball info (4,)
            ball_obs = self.get_ball_obs()  # [bx, by, bvx, bvy]

            # 2) Self-only info for each rod (list of 4 arrays, each (4,))
            self_list = self.get_per_agent_self_only_obs()

            per_agent_list = []
            for i in range(4):
                obs_i = np.concatenate([ball_obs, self_list[i]], axis=0)  # (8,)
                per_agent_list.append(obs_i.astype(np.float32))

            return per_agent_list

        else:
            raise ValueError(f"Unknown obs_mode in get_per_agent_obs: {self.obs_mode}")
    
    def combine_agent_actions(self, rod_actions):
        """
        rod_actions: list of 4 arrays, each shape (2,) = [slide, rotate]
        Returns:
            joint_action: np.ndarray of shape (8,)
            [s1, s2, s3, s4, r1, r2, r3, r4]
        """
        if len(rod_actions) != 4:
            raise ValueError(f"Expected 4 rod actions, got {len(rod_actions)}")

        joint_action = np.zeros(8, dtype=np.float32)
        for i in range(4):
            a = np.asarray(rod_actions[i], dtype=np.float32)
            if a.shape[0] != 2:
                raise ValueError(f"Rod {i} action must be shape (2,), got {a.shape}")
            joint_action[i]     = a[0]  # slide
            joint_action[i + 4] = a[1]  # rotate

        return joint_action

    
    def close(self):
        p.disconnect(self.client)

    def run_goal_debug_loop(self):
        """
        An interactive debug loop for visualizing goal lines and manually controlling the ball,
        now with real-time contact status reporting.
        """
        if not self.goal_debug_mode:
            print("Goal debug mode is not enabled. Please instantiate Env with goal_debug_mode=True.")
            return

        print("\n" + "="*80 + "\nINTERACTIVE DEBUG MODE\n" + "="*80)
        print(" - Use ARROW KEYS to move the ball.")
        print(" - Use the sliders to adjust the goal lines.")
        print(" - Contact status with agent/opponent rods will be printed on change.")
        print(" - Press ESC or close the window to exit.")
        
        table_aabb = p.getAABB(self.table_id)
        y_min, y_max = table_aabb[0][1], table_aabb[1][1]
        z_pos = 0.55  # Approximate height of the playing surface

        line1_id, line2_id = None, None
        move_speed = 0.01
        last_contact_status = "None"

        try:
            while True:
                # Keyboard events for ball control
                keys = p.getKeyboardEvents()
                ball_pos, ball_orn = p.getBasePositionAndOrientation(self.ball_id)
                new_pos = list(ball_pos)

                if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN: new_pos[0] -= move_speed
                if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN: new_pos[0] += move_speed
                if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN: new_pos[1] += move_speed
                if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN: new_pos[1] -= move_speed
                p.resetBasePositionAndOrientation(self.ball_id, new_pos, ball_orn)

                # --- Contact Detection Logic ---
                current_contact_status = "None"
                agent_player_links = self.team1_player_links if self.player_id == 1 else self.team2_player_links
                opponent_player_links = self.team2_player_links if self.player_id == 1 else self.team1_player_links

                # Check for contact with agent rods
                for link_idx in agent_player_links:
                    if p.getContactPoints(bodyA=self.table_id, bodyB=self.ball_id, linkIndexA=link_idx):
                        current_contact_status = f"Contact with AGENT (Player {self.player_id})"
                        break
                
                # If no agent contact, check for opponent contact
                if current_contact_status == "None":
                    for link_idx in opponent_player_links:
                        if p.getContactPoints(bodyA=self.table_id, bodyB=self.ball_id, linkIndexA=link_idx):
                            current_contact_status = f"Contact with OPPONENT (Player {3 - self.player_id})"
                            break
                
                if current_contact_status != last_contact_status:
                    print(f"\n[Contact Status Changed] -> {current_contact_status}")
                    last_contact_status = current_contact_status

                # Read sliders and update goal lines
                self.goal_line_x_1 = p.readUserDebugParameter(self.goal_line_slider_1)
                self.goal_line_x_2 = p.readUserDebugParameter(self.goal_line_slider_2)
                
                # Draw new debug lines
                if line1_id is not None: p.removeUserDebugItem(line1_id)
                if line2_id is not None: p.removeUserDebugItem(line2_id)
                line1_id = p.addUserDebugLine([self.goal_line_x_1, y_min, z_pos], [self.goal_line_x_1, y_max, z_pos], [1, 0, 0], 2)
                line2_id = p.addUserDebugLine([self.goal_line_x_2, y_min, z_pos], [self.goal_line_x_2, y_max, z_pos], [0, 0, 1], 2)

                p.stepSimulation()
                time.sleep(1./240.)

        except p.error as e:
            pass # This can happen if the user closes the window
        finally:
            print("\nExiting interactive debug mode.")
            self.close()


def test_individual_rod_control():
    print("\n" + "="*80 + "\nTEST: INDIVIDUAL ROD CONTROL (Team 1 - RED)\n" + "="*80)
    env = FoosballEnv(render_mode='human', curriculum_level=1, player_id=1)
    obs, _ = env.reset()
    num_rods = 4
    for i in range(num_rods):
        print(f"\nRod {i+1} (rotate index {i+4}): Spinning CLOCKWISE")
        action = np.zeros(8)
        action[i+4] = 1.0
        for _ in range(50):
            env.step(action)
            joint_states = p.getJointStates(env.table_id, env.team1_rev_joints)
            joint_pos = [state[0] for state in joint_states]
            print(f"Team 1 rod positions: {joint_pos}")
            time.sleep(0.01)
        print(f"Rod {i+1} (rotate index {i+4}): Spinning COUNTER-CLOCKWISE")
        action = np.zeros(8)
        action[i+4] = -1.0
        for _ in range(50):
            env.step(action)
            joint_states = p.getJointStates(env.table_id, env.team1_rev_joints)
            joint_pos = [state[0] for state in joint_states]
            print(f"Team 1 rod positions: {joint_pos}")
            time.sleep(0.01)
        action = np.zeros(8)
        env.step(action)
        time.sleep(0.2)
    for i in range(num_rods):
        print(f"\nRod {i+1} (slide index {i}): Sliding IN")
        action = np.zeros(8)
        action[i] = 1.0
        for _ in range(50):
            env.step(action)
            time.sleep(0.01)
        print(f"Rod {i+1} (slide index {i}): Sliding OUT")
        action = np.zeros(8)
        action[i] = -1.0
        for _ in range(50):
            env.step(action)
            time.sleep(0.01)
        action = np.zeros(8)
        env.step(action)
        time.sleep(0.2)
    print("\n✅ Individual rod control test complete")
    env.close()

def test_blue_team_rod_control():
    """Test moving each rod of the blue team individually."""
    print("\n" + "="*80 + "\nTEST: INDIVIDUAL ROD CONTROL (Team 2 - BLUE)\n" + "="*80)
    env = FoosballEnv(render_mode='human', curriculum_level=1, player_id=2)
    obs, _ = env.reset()
    num_rods = 4
    for i in range(num_rods):
        print(f"\nRod {i+1} (rotate index {i+4}): Spinning CLOCKWISE")
        action = np.zeros(8)
        action[i+4] = 1.0
        for _ in range(50):
            env.step(action)
            joint_states = p.getJointStates(env.table_id, env.team2_rev_joints)
            joint_pos = [state[0] for state in joint_states]
            print(f"Team 2 rod positions: {joint_pos}")
            time.sleep(0.01)
        print(f"Rod {i+1} (rotate index {i+4}): Spinning COUNTER-CLOCKWISE")
        action = np.zeros(8)
        action[i+4] = -1.0
        for _ in range(50):
            env.step(action)
            joint_states = p.getJointStates(env.table_id, env.team2_rev_joints)
            joint_pos = [state[0] for state in joint_states]
            print(f"Team 2 rod positions: {joint_pos}")
            time.sleep(0.01)
        action = np.zeros(8)
        env.step(action)
        time.sleep(0.2)
    for i in range(num_rods):
        print(f"\nRod {i+1} (slide index {i}): Sliding IN")
        action = np.zeros(8)
        action[i] = 1.0
        for _ in range(50):
            env.step(action)
            time.sleep(0.01)
        print(f"Rod {i+1} (slide index {i}): Sliding OUT")
        action = np.zeros(8)
        action[i] = -1.0
        for _ in range(50):
            env.step(action)
            time.sleep(0.01)
        action = np.zeros(8)
        env.step(action)
        time.sleep(0.2)
    print("\n✅ Individual rod control test complete for Blue Team")
    env.close()

def test_stage_3_spawning():
    """Test the ball spawning for Stage 3 to ensure it travels towards the goal."""
    print("\n" + "="*80 + "\nTEST: STAGE 3 BALL SPAWNING (Team 1 - RED)\n" + "="*80)

    env = FoosballEnv(render_mode='human', curriculum_level=3, player_id=1)

    for i in range(5):
        print(f"\n--- Test run {i+1}/5 ---")
        obs, _ = env.reset()
        print("  Setting all rods to 90 degrees for a clear view...")
        env._set_all_rods_to_90_degrees()

        # Reset the ball again to the curriculum position after moving the rods
        env._curriculum_spawn_ball()
        obs = env._get_obs()
        ball_pos = obs[:3]
        ball_vel = obs[3:6]
        print(f"  Initial Ball Position: {ball_pos}")
        print(f"  Initial Ball Velocity: {ball_vel}")

        # Let the simulation run to observe the ball's trajectory

        for _ in range(150):
            # Only step the physics, don't apply any agent actions
            p.stepSimulation()
            if env.render_mode == 'human':
                time.sleep(1./240.)

            

        final_ball_pos, _ = p.getBasePositionAndOrientation(env.ball_id)
        print(f"  Final Ball Position:   {final_ball_pos}")

        

        # Check if the ball crossed the goal line
        if final_ball_pos[0] < env.goal_line_x_1:
            print("  ✅ GOAL SCORED!")
        else:
            print("  ❌ NO GOAL. Ball did not reach the goal line.")

        time.sleep(1)

    env.close()

    print("\n✅ Stage 3 spawning test complete.")



def test_mirrored_obs_and_contact_reward():
    """
    Test the _get_mirrored_obs function and the contact reward logic.
    This test is now a visual debugging tool.
    """
    print("\n" + "="*80 + "\nTEST: MIRRORED OBSERVATION & CONTACT REWARD (VISUAL)\n" + "="*80)
    print("This test will now run in 'human' mode for visual inspection.")
    
    env = FoosballEnv(render_mode='human', player_id=1)
    obs, _ = env.reset()

    # --- Test Mirrored Observation Logic (remains a logic test) ---
    print("\n--- Testing Mirrored Observation Logic ---")
    ball_pos = np.array([0.1, 0.2, 0.3])
    ball_vel = np.array([-0.1, -0.2, -0.3])
    team1_pos = np.arange(0, 8, dtype=np.float32) * 0.1
    team1_vel = np.arange(0, 8, dtype=np.float32) * -0.1
    team2_pos = np.arange(8, 16, dtype=np.float32) * 0.1
    team2_vel = np.arange(8, 16, dtype=np.float32) * -0.1
    
    original_get_base_pos = p.getBasePositionAndOrientation
    original_get_base_vel = p.getBaseVelocity
    original_get_joint_states = p.getJointStates
    
    p.getBasePositionAndOrientation = lambda body_id: (ball_pos, None)
    p.getBaseVelocity = lambda body_id: (ball_vel, None)
    
    def mock_get_joint_states(body_id, joint_indices):
        if joint_indices == env.team1_joints:
            return [(pos, vel) for pos, vel in zip(team1_pos, team1_vel)]
        if joint_indices == env.team2_joints:
            return [(pos, vel) for pos, vel in zip(team2_pos, team2_vel)]
        return []

    p.getJointStates = mock_get_joint_states
    mirrored_obs = env._get_mirrored_obs()
    p.getBasePositionAndOrientation = original_get_base_pos
    p.getBaseVelocity = original_get_base_vel
    p.getJointStates = original_get_joint_states

    expected_mirrored_ball_pos = np.array([-0.1, 0.2, 0.3])
    expected_mirrored_ball_vel = np.array([0.1, -0.2, -0.3])
    assert np.allclose(mirrored_obs[:3], expected_mirrored_ball_pos), "Mirrored ball position is incorrect."
    print("  ✅ Mirrored observation logic test passed.")

    # --- Visual Test for Contact Reward ---
    print("\n--- Visual Test for Contact Reward ---")
    print("Watch the simulation window. The ball will be placed on an agent rod.")
    
    agent_rod_link_index = env.team1_rev_joints[2]
    link_state = p.getLinkState(env.table_id, agent_rod_link_index)
    link_pos = link_state[0]
    
    print(f"  Agent rod link position: {link_pos}")
    print(f"  Placing ball at: {link_pos}")
    
    p.resetBasePositionAndOrientation(env.ball_id, link_pos, [0,0,0,1])
    
    # Let the simulation run for a moment
    for _ in range(50):
        p.stepSimulation()
        time.sleep(0.01)

    contact_points = p.getContactPoints(bodyA=env.table_id, bodyB=env.ball_id, linkIndexA=agent_rod_link_index)
    reward = env._compute_reward(np.zeros(8))
    
    print(f"  Detected {len(contact_points)} contact points with the target link.")
    print(f"  Calculated reward: {reward}")
    if len(contact_points) > 0 and reward >= 100:
        print("  ✅ Agent contact reward appears correct.")
    else:
        print("  ❌ Agent contact reward is NOT as expected. Please visually inspect.")
        
    time.sleep(2) # Pause for observation

    env.close()
    print("\n✅ Test finished. Please review the output and visual simulation.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test or debug the FoosballEnv.")
    parser.add_argument("--test", type=str, default="all", choices=["all", "rods_red", "rods_blue", "spawn3", "interactive_debug", "mirrored_contact"], help="Specify which test to run.")
    args = parser.parse_args()

    if args.test == "interactive_debug":
        env = FoosballEnv(goal_debug_mode=True, render_mode='human')
        env.run_goal_debug_loop()
    elif args.test == "rods_red":
        test_individual_rod_control()
    elif args.test == "rods_blue":
        test_blue_team_rod_control()
    elif args.test == "spawn3":
        test_stage_3_spawning()
    elif args.test == "mirrored_contact":
        test_mirrored_obs_and_contact_reward()
    elif args.test == "all":
        test_individual_rod_control()
        test_blue_team_rod_control()
        print("\nSkipping spawn3, mirrored_contact, and interactive_debug tests in 'all' mode. Run with --test <test_name> to see them.")
