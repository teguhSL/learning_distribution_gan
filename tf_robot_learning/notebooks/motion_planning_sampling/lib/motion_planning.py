import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import networkx as nx
from pb_utils.visualize import set_q
import matplotlib
import pybullet as p
import time

class RRT():
    def __init__(self, D, sampler, col_checker, interpolator):
        self.D = D
        self.sampler = sampler
        self.col_checker = col_checker
        self.interpolator = interpolator
        self.samples = [] 
        
    def check_collision(self, sample):
        return self.col_checker.check_collision(sample)
    
    def sample(self, get_valid = True):
        status = True
        if get_valid:
            while status is True:
                sample =  self.sampler.sample()
                status = self.check_collision(sample.flatten())
        else:
                sample =  self.sampler.sample()            
        return sample
    
    def interpolate(self, sample1, sample2, N):
        return self.interpolator.interpolate(sample1, sample2, N)
               
    def extend(self, cur_index, sample1, sample2, step_length = 0.1):
        #state1, state2 = self.correct_manifold(sample1, sample2)
        state1, state2 = sample1.copy(), sample2.copy()
        dist = np.linalg.norm(state2-state1)
        N = int(dist/step_length) + 2
        state_list = self.interpolate(state1, state2, N)
        next_states = []
        for state in state_list:
            if self.check_collision(state):
                #print('collision')
                #print(state)
                break
            next_states += [(state)]
            
        #add the nodes to the graph
        for next_state in next_states[1:]:
            next_index = self.G.number_of_nodes()
            self.G.add_nodes_from([next_index])
            self.G.add_edge(cur_index, next_index)
            self.samples += [next_state]
            cur_index = next_index
            
        return next_states

    def find_nearest(self, sample1, sample_set, n = 1):
        #find a nearest node
        if len(sample_set) > 1:
            index = np.argpartition(np.linalg.norm(sample1-sample_set,axis=1),n)[0:n]
            return index,sample_set[index]
        else:
            return [0], sample_set[0]  
    
    def init_plan(self, init_state, goal_state):
        self.G = nx.Graph()
        self.nodes = [0]
        self.edges = []
        self.samples = [init_state]
        self.G.add_node(0)
        self.init_state = init_state.copy()
        self.goal_state = goal_state.copy()
    
    def step_plan(self):
        success = False
        #sample random state
        self.random_sample = self.sample()

        #find a nearest node
        nearest_index, nearest_sample = self.find_nearest(self.random_sample, np.array(self.samples))

        #extend to the random state
        self.next_states = self.extend(nearest_index[0], nearest_sample.flatten(), self.random_sample.flatten() )
        
        #check distance with goal state
        nearest_index, nearest_sample = self.find_nearest(self.goal_state, np.array(self.samples))
        
        #extend to the goal state
        clear_output()
        print('Reached a random state...')
        print(nearest_sample)
        #input()
        self.next_states_goal = self.extend(nearest_index[0], nearest_sample.flatten(), self.goal_state.flatten())
        #print(next_states)
        if len(self.next_states_goal) == 0: return False
        clear_output()
        print('Reached a goal state...')
        print(self.next_states_goal[-2:])
        print(self.goal_state)
        #input()
        
        if np.linalg.norm(self.next_states_goal[-1] - self.goal_state)< 0.001:
            print('Solution is found!')
            success = True
        return success,0
        
    def plan(self, init_state, goal_state):
        self.init_plan(init_state, goal_state)
        success = False
        self.num_plan = 0
        self.nfevs = 0
        while success is False:
            success, nfev = self.step_plan()
            self.nfevs += nfev
            self.num_plan += 1
            print('Planning...')
        #clear_output()
        #find the path
        path = nx.dijkstra_path(self.G, 0, self.G.number_of_nodes()-1)
        return path, self.nfevs
        
    def plot_graph(self):
        edges = self.G.edges().keys()
        segs = []
        for edge in edges:
            node1 = self.samples[edge[0]]
            node2 = self.samples[edge[1]]
            segs.append([[node1[0], node1[1]],[node2[0],node2[1]]])
        ln_coll = matplotlib.collections.LineCollection(segs)
        ax = plt.gca()
        ax.add_collection(ln_coll)
        plt.axis([0,2*np.pi,0,2*np.pi])
        
    def shortcut_path(self, path_in, step_length=0.1):
        #simple shortcutting algo trying to iteratively remove nodes from the path
        path = path_in.copy()
        while len(path) > 2:
            idx_remove = -1
            for idx in range(len(path)-2):
                #try to remove node(idx+1) from path: Is interpolation between node(idx) and node(idx+2) free?
                node1 = self.samples[path[idx]]
                node2 = self.samples[path[idx+2]]
                dist = np.linalg.norm(node2-node1)
                N = int(dist/step_length) + 2
                state_list = self.interpolate(node1, node2, N)
                free = True
                for state in state_list:
                    if self.check_collision(state):
                        free = False
                        break
                        
                if free is True:
                    idx_remove = idx+1
                    break
                
            if idx_remove == -1:
                break
            del path[idx_remove]
        return path

class cRRT(RRT):
    def __init__(self, D, sampler, col_checker, interpolator, projector, max_plan=300):
        self.D = D
        self.sampler = sampler
        self.col_checker = col_checker
        self.interpolator = interpolator
        self.projector = projector
        self.samples = []
        self.max_plan = max_plan
        self.extend_nfevs = []

    def project(self, q, maxiter=50):
#         return q, 0,True 
        res = self.projector.project(q, maxiter=maxiter)
        return res['q'], res['nfev'], res['stat']

    def sample(self, get_valid=True):
        is_collide = True
        status = False
        if get_valid:
            #Get non-colliding configuration
            while status is False or is_collide is True:
                sample = self.sampler.sample()
                proj_sample, nfev, status = self.project(sample.flatten())
                is_collide = self.check_collision(proj_sample.flatten())
        else:
            while status is False:
                sample = self.sampler.sample()
                self.raw_sample = sample
                proj_sample = sample
                status = True
                nfev = 0
                proj_sample, nfev, status = self.project(sample.flatten())                
        return proj_sample, nfev

    def extend(self, cur_index, sample1, sample2, step_length=0.3, max_increments=40):
        cur_state, state_s = sample1.copy(), sample2.copy()
        next_states = [cur_state]
        nfevs = 0
        for n in range(max_increments):
            d = state_s - cur_state
            d_norm = np.linalg.norm(d)
            if d_norm < step_length:
                if self.check_collision(state_s) is False:
                    next_states += [(state_s)]
                break
            next_state = cur_state + step_length * (state_s - cur_state) / np.linalg.norm(d)
            cur_state, nfev, status = self.project(next_state.flatten())
            self.extend_nfevs += [nfev]
            nfevs += nfev
            self.next_state = next_state
            self.next_state_projected = cur_state
            self.prev_state = next_states[-1]
            if self.check_collision(cur_state) or status is False:
                break
            elif (np.linalg.norm(cur_state - next_states[-1])) > 3 * step_length:
                print('moving to larger distance by interpolation')
                print(cur_state)
                print(next_states[-1])
                break

            next_states += [cur_state]

        # add the nodes to the graph
        for next_state in next_states[1:]:
            next_index = self.G.number_of_nodes()
            self.G.add_nodes_from([next_index])
            self.G.add_edge(cur_index, next_index)
            self.samples += [next_state]
            cur_index = next_index

        return next_states, nfevs

    def step_plan(self):
        success = False
        # sample random state
        nfevs = 0
        self.random_sample, nfev = self.sample(False)
        self.random_sample = self.random_sample.flatten()
        nfevs += nfev

        # find a nearest node
        nearest_index, nearest_sample = self.find_nearest(self.random_sample, np.array(self.samples))

        # extend to the random state
        self.next_states, nfev = self.extend(nearest_index[0], nearest_sample.flatten(), self.random_sample.flatten())
        nfevs += nfev
        q_reach_a = self.next_states[-1]
        
        #if only one state, break
        if len(self.next_states) <= 1:
            return False, nfevs

        # extend to the goal
        # find the nearest goal
        nearest_index, nearest_sample = self.find_nearest(q_reach_a, np.array(self.goal_state))
        q_goal = nearest_sample.flatten()

        self.next_states_goal, nfev = self.extend(len(self.samples) - 1, self.next_states[-1], q_goal)
        self.q_goal = q_goal
        nfevs += nfev
        if len(self.next_states_goal) == 0: return False

        if np.linalg.norm(self.next_states_goal[-1] - q_goal) < 0.001:
            print('Solution is found!')
            success = True
        return success, nfevs

    def plan(self, q_init, q_goals, max_extension_steps=300, n_retry=3):
        tic = time.time()
        success = False
        total_extension = 0
        total_projection = 0
        for i in range(n_retry):
            self.init_plan(q_init, q_goals)

            rrt_func_calls = 0
            ext_steps = 0
            while ext_steps < max_extension_steps:
                success, rrt_func_calls_ = self.step_plan()
                rrt_func_calls += rrt_func_calls_
                ext_steps += 1
                if success:
                    break

            total_extension += ext_steps
            total_projection += rrt_func_calls
            if success: break
        self.retry = i
#         clear_output()

        if success is False:
            print("No solution found!")
            return [], 0, 0, False, self.retry, 0, 0
        print("Solution found!")
        # find the path
        path = nx.dijkstra_path(self.G, 0, self.G.number_of_nodes() - 1)
        toc = time.time()
        traj = []
        for i in path:
            traj += [self.samples[i]]
        traj = np.array(traj)
        return traj, total_projection, total_extension, success, self.retry, toc - tic, path

    def shortcut_path(self, path_in, step_length=0.1):
        # simple shortcutting algo trying to iteratively remove nodes from the path
        path = path_in.copy()
        while len(path) > 2:
            idx_remove = -1
            for idx in range(len(path) - 2):
                # try to remove node(idx+1) from path: Is interpolation between node(idx) and node(idx+2) free?
                node1 = self.samples[path[idx]]
                node2 = self.samples[path[idx + 2]]
                dist = np.linalg.norm(node2 - node1)
                N = int(dist / step_length) + 2
                state_list = self.interpolate(node1, node2, N)
                free = True
                for state in state_list:
                    state, nfev, status = self.project(state)
                    if self.check_collision(state) or status is not True:
                        free = False
                        break

                if free is True:
                    idx_remove = idx + 1
                    break

            if idx_remove == -1:
                break
            del path[idx_remove]
        return path

    def interpolate_traj(self, path, step_length=0.1):
        states = []
        for idx in range(len(path) - 1):
            node1 = self.samples[path[idx]]
            node2 = self.samples[path[idx + 1]]
            dist = np.linalg.norm(node2 - node1)
            N = int(dist / step_length) + 2
            path_interpolated = lin_interpolate(node1, node2, N)
            for state in path_interpolated:
                state, _, _ = self.project(state)
                states += [state]
        return states

class sampler():
    """
    General sampler, given the joint limits
    """

    def __init__(self, joint_limits=None):
        self.dof = joint_limits.shape[1]
        self.joint_limits = joint_limits

    def sample(self, N=1):
        samples = np.random.rand(N, self.dof)
        samples = self.joint_limits[0] + samples * (self.joint_limits[1] - self.joint_limits[0])
        return samples

class talos_sampler():
    """
    Sampler for Talos: for base and joint angles
    """
    def __init__(self, base_sampler, base_ori, joint_sampler, ignore_indices = [], q_ref = None):
        self.base_sampler = base_sampler
        self.base_ori = base_ori
        self.joint_sampler = joint_sampler
        self.q_ref = q_ref
        self.ignore_indices = np.array(ignore_indices)

        
    def sample(self, N=1):
        samples = []
        for i in range(N):
            sample = np.concatenate([self.base_sampler.sample()[0], self.base_ori, self.joint_sampler.sample()[0]])
            if self.q_ref is not None:
#                 sample[-14:-7] = self.q_ref[-14:-7] #let the left hand to assume standard values, for talos
#                 sample[8+7:8+7+7] = self.q_ref[8+7:8+7+7] #let the left hand to assume standard values, for walker
                sample[self.ignore_indices] = self.q_ref[self.ignore_indices] #let the left hand to assume     
            samples += [sample]
        return np.array(samples)

class hybrid_sampler():
    def __init__(self, random_sampler, gan_sampler, p_random=0.3):
        self.random_sampler = random_sampler
        self.gan_sampler = gan_sampler
        self.p_random = p_random

    def sample(self, N=1, _poses=None, var=1.):
        if np.random.rand() > self.p_random:
            return self.gan_sampler.sample(N, _poses, var)
        else:
            return self.random_sampler.sample(N)

def lin_interpolate(state1, state2, n=1.):
    state_list = []
    for i in range(n+1):
        state_list.append(state1 + 1.*i*(state2-state1)/n)
    return state_list

class interpolator():
    def __init__(self):
        pass

    def interpolate(self, state1, state2, N):
        states = lin_interpolate(state1, state2, N)
        return states

# def check_collision(robot_id, object_ids, omit_indices=[-1]):
#     col_info = []
#     is_col = False
#     for object_id in object_ids:
#         ress = p.getClosestPoints(robot_id, object_id, distance=2)
#         for res in ress:
#             if res[3] in omit_indices:
#                 continue
#             if res[8] < 0:
#                 is_col = True
#                 col_info += [(res[3], res[4], res[7], res[8])]  # linkA, linkB, normal, distance
#     return is_col, col_info

def check_collision(robot_id, object_ids, omit_indices=[-1], verbose=True):
    p.performCollisionDetection()
    self_collision = False
    collision = False
    conts = p.getContactPoints(robot_id, robot_id)
    # contact distance, positive for separation, negative for penetration
    if len(conts) >= 1:
        for cont in conts:
            distance = cont[8]
            if distance < 1e-3: # if smaller than 1 mm
                if verbose:
                    print("Robot's ", link_names[cont[3]], " is in contact with the link ",
                  link_names[cont[4]], "with a contact distance of ", distance)
                self_collision = True

            if self_collision: 
                return self_collision or collision, None
                
    if not self_collision:
        for object_id in object_ids:
            conts = p.getContactPoints(robot_id, object_id)
            if len(conts) >= 1:
                for cont in conts:
                    if cont[8] < 1e-3:
                        collision = True
                        return self_collision or collision, None
    
    return self_collision or collision, None

class col_checker():
    def __init__(self, robot_id, joint_indices, object_ids, omit_indices=[-1], floating_base = False):
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.object_ids = object_ids
        self.omit_indices = omit_indices
        self.floating_base = floating_base

    def check_collision(self, q):
        if self.floating_base:
            set_q(q, self.robot_id, self.joint_indices, True)
        else:
            set_q(q, self.robot_id, self.joint_indices)
        return check_collision(self.robot_id, self.object_ids, omit_indices=self.omit_indices)[0]
