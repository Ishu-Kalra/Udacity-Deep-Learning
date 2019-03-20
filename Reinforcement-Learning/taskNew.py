import numpy as np
from physics_sim import PhysicsSim
import math
class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent // changed to target only z
        """
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        self.state_size = self.action_repeat
        self.action_low = 0    
        self.action_high = 900  
        self.action_size = 1     
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        #Adding camel case variables
        self.stateSize = self.action_repeat
        self.actionRepeat = self.action_repeat
        self.actionLow = self.action_low
        self.actionHigh = self.action_high
        self.actionSize = self.action_size
    def get_reward(self):

        reward = 5.0 #Initital Reward is 5
        #Reviewers Suggestions
        penalty = abs(abs(self.sim.v[2] - self.target_pos).sum() - abs(self.sim.v).sum())
        reward -= penalty

        

        if self.sim.pose[2] == 0: #Crash landing
            reward -= 150

        self.distanceFromTarget = math.sqrt((math.fabs(self.sim.pose[2] ** 2 - self.target_pos ** 2))) 

        if self.distanceFromTarget < 2: #near to target
            reward += 50
        
        if self.distanceFromTarget > 15: #Great distance from target
            reward -= 100

        if self.sim.angular_v[2] > 0: #Angular velocity reward and penalty
            reward -= self.sim.angular_v
        else:
            reward += 1

        reward = np.clip(reward, -1, 1) 

        return reward

    def step(self, rotor_speeds):
        reward = 0.0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward()
            pose_all.append(self.sim.pose[2])
        nextState = np.array(pose_all)
        return nextState, reward, done

    def reset(self):
        self.sim.reset() 
        state = [self.sim.pose[2]] * self.actionRepeat
        return state