import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

class Robot():
    def __init__(self,l1, l2):
        self.l1 = l1
        self.l2 = l2
        self.theta1 = 0.
        self.theta2 = 0.
        self.obstacles = []
        self.R = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)],[np.sin(np.pi/2), np.cos(np.pi/2)]])
        
    def set_dof(self,theta):
        self.theta1 = theta[0]
        self.theta2 = theta[1]
        
    def rotate90(self,vec):
        return np.dot(self.R, vec)
    
    def add_obstacle(self, x, y, r):
        obstacle = dict()
        obstacle['rad'] = r
        obstacle['pos'] = np.array([x,y])
        self.obstacles += [obstacle]
        
    def check_collision_single(self, obstacle):    
        c = obstacle['pos'][:,None]
        r = obstacle['rad']
        b1 = c - self.p0
        b2 = c - self.p1
        self.res1 = np.linalg.solve(self.A1,b1)
        self.res2 = np.linalg.solve(self.A2,b2)
        if self.res1[0] >= 0. and self.res1[0] <= 1.:
            if np.abs(self.res1[1]) <= r:
                return True
        else:
            if np.linalg.norm(self.p0-c) < r or np.linalg.norm(self.p1-c) < r:
                return True
        
        if self.res2[0] >= 0. and self.res2[0] <= 1.:
            if np.abs(self.res2[1]) <= r:
                return True
        else:
            if np.linalg.norm(self.p1-c) < r or np.linalg.norm(self.p2-c) < r:
                return True
            
        return False
        
        
    def normalize(self,vec):
        return vec/np.linalg.norm(vec)
        
    def check_collision(self):
        self.p0 = np.array([0.,0.])[:,None]
        self.p1 = np.array([self.l1*np.cos(self.theta1),self.l1*np.sin(self.theta1)])[:,None]
        self.p2 = np.array([self.p1[0,0] + self.l2*np.cos(self.theta1+self.theta2),self.p1[1,0] + self.l2*np.sin(self.theta1+self.theta2)])[:,None]
        self.v1 = self.normalize(self.rotate90(self.p1-self.p0))
        self.v2 = self.normalize(self.rotate90(self.p2-self.p1))
        self.A1 = np.hstack([self.p1-self.p0, -self.v1])
        self.A2 = np.hstack([self.p2-self.p1, -self.v2])
        
        for obstacle in self.obstacles:
            if self.check_collision_single(obstacle):
                return True
            
        return False
        
    def plot(self):
        x = [0., self.l1*np.cos(self.theta1), self.l1*np.cos(self.theta1) + self.l2*np.cos(self.theta1+ self.theta2)]
        y = [0., self.l1*np.sin(self.theta1), self.l1*np.sin(self.theta1) + self.l2*np.sin(self.theta1+ self.theta2)]
        plt.plot(x,y,'b',linewidth=5)
        plt.plot(x[:-1],y[:-1],'oy',markersize=3 )
        fig = plt.gcf()
        ax = fig.gca()
        for obstacle in self.obstacles:
            r = obstacle['rad']
            p = obstacle['pos']
            circle = plt.Circle(p, r, color='r')
            ax.add_artist(circle)
            
        #add base
        rect = patches.Rectangle((-0.35,-0.35),0.7,0.7,linewidth=1,edgecolor='g',facecolor='g')
        ax.add_patch(rect)
        
        ax.axis('equal')
        ax.set(xlim=(-2*self.l1-2*self.l2,2*self.l1+2*self.l2), ylim=( -2*self.l1-2*self.l2,2*self.l1+2*self.l2))
        
        
    def sample_state(self, N=1, get_valid = False):
        samples = []
        samples_status = []
        while len(samples) < N:
            sample = np.random.rand(2)*np.pi*2
            self.set_dof(sample)
            sample_status = self.check_collision()
            if get_valid:
                if sample_status is True:
                    continue
            samples.append(sample)
            samples_status.append(sample_status)
            
        return np.array(samples), np.array(samples_status)
