import numpy as np
import pinocchio as pin
from utils import *
from scipy.optimize import  minimize      


compute_first = False

class CostBoundNew:
    """
    This cost is to keep within the joint limits
    """
    def __init__(self, bounds, thres = 1e-8): 
        self.bounds = bounds
        self.dof = bounds.shape[1]
        self.zeros = np.zeros(self.dof)
        self.thres = thres
        
    def calc(self, q):
        self.rlb_min = np.min([q - self.bounds[0], self.zeros],axis=0)
        self.rlb_max = np.max([q-self.bounds[1], self.zeros],axis=0)
        self.res = self.rlb_min + self.rlb_max 
        return self.res
    
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
        
        self.J = (self.res**2 > self.thres)*np.eye(len(self.res))
        return self.J
    
class CostPostureNew:
    """
    This cost is to regulate the projection around a nominal posture
    """
    def __init__(self, rmodel, rdata, desired_posture, weights = None):
        self.rmodel = rmodel
        self.rdata  = rdata
        self.desired_posture = desired_posture
        if weights is None: 
            self.weights = np.ones(rmodel.nq)
        else:
            self.weights = weights
        
    def calc(self, q):
        self.res = self.weights*(q-self.desired_posture)
        return self.res
    
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
        self.J = np.diag(self.weights) 
        return self.J       

    
class CostFrameTranslationFloatingBaseNew():
    """
    The cost for frame translation of a floating base system. 
    In this version, we remove the Jacobian due to the base orientation
    """   
    def __init__(self, rmodel, rdata, desired_pose, ee_frame_id , weight):  
        self.rmodel = rmodel
        self.rdata  = rdata
        self.desired_pose = desired_pose
        self.ee_frame_id = ee_frame_id
        self.weight = weight
        
    def calc(self, q):
        ### Add the code to recompute your cost here
        if compute_first is False:
            pin.forwardKinematics(self.rmodel, self.rdata, q)
            pin.updateFramePlacement(self.rmodel, self.rdata, self.ee_frame_id)
        pose = self.rdata.oMf[self.ee_frame_id] 
        self.pos, self.ori = pose.translation, pose.rotation
        self.res = self.weight*(self.pos-self.desired_pose[:3]) 
        return self.res
        
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
            
        self.J = self.computeJacobian(q)
        return self.J
            
    def computeJacobian(self, q):
        if compute_first is False:
            pin.computeJointJacobians(self.rmodel, self.rdata, q)
        J = pin.getFrameJacobian(self.rmodel, self.rdata,self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        #modify J to remove the term corresponding to the base frame orientation
        Jt = np.zeros((3, 35))
        Jt[:,:3] = J[:3,:3]
        Jt[:,7:] = J[:3,6:]
        return Jt

        
class CostCOMBoundsNew:
    """
    This cost is to ensure that the COM is within the bounds
    """
    def __init__(self, rmodel, rdata, bounds, thres = 1e-8):
        self.rmodel = rmodel
        self.rdata  = rdata
        self.bounds = bounds
        self.zeros = np.zeros(3)
        self.thres = thres
        
    def calc(self, q):
        if compute_first is False:
            pin.forwardKinematics(self.rmodel, self.rdata, q)
        self.com = pin.centerOfMass(self.rmodel, self.rdata, q)
        self.rlb_min = np.min([self.com - self.bounds[0], self.zeros],axis=0)
        self.rlb_max = np.max([self.com - self.bounds[1], self.zeros],axis=0)
        self.res = self.rlb_min + self.rlb_max
        return self.res
    
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
        J = pin.jacobianCenterOfMass(self.rmodel, self.rdata, q)
        #remove the gradient due to the base orientation
        J = np.hstack([J[:,:3], np.zeros((3,4)), J[:,6:]])
        #filter according to the active bounds
        stat = self.res**2 > self.thres
        self.J = (stat*np.eye(3)).dot(J)
        return self.J
    
    
    
class CostFrameRPYFloatingBaseNew():
    """
    The cost for frame placement of a floating base system. 
    The orientation is described with YPR
    In this version, we remove the Jacobian due to the base orientation
    """   
    def __init__(self, rmodel, rdata, desired_pose, ee_frame_id , sel_vector):  
        self.rmodel = rmodel
        self.rdata  = rdata
        self.desired_pose = desired_pose
        self.ee_frame_id = ee_frame_id
        self.sel_vector = sel_vector
        
    def calc(self, q):
        ### Add the code to recompute your cost here
        if compute_first is False:
            pin.forwardKinematics(self.rmodel, self.rdata, q)
            pin.updateFramePlacement(self.rmodel, self.rdata, self.ee_frame_id)
        pose = self.rdata.oMf[self.ee_frame_id] 
        self.pos, self.ori = pose.translation, pose.rotation
        self.rpy = mat2euler(self.ori)
        self.r_pos = self.sel_vector[:3]*(self.pos-self.desired_pose[:3]) 
        self.r_ori = self.sel_vector[3:]*(self.rpy-self.desired_pose[3:])
        self.res = np.concatenate([self.r_pos,self.r_ori])
        return self.res
    
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
            
        self.J = self.computeJacobian(q)
        self.B = self.compute_Bz(*self.rpy)
        self.Cx1 = self.J[:3]
        self.Cx2 = np.linalg.inv(self.B).dot(self.J[3:])
        self.Cx = np.vstack([self.Cx1, self.Cx2])
        return self.Cx
        
    def compute_Bz(self,z,y,x):
        #zyx
        B = np.zeros((3,3))
        B[0,1] = -np.sin(z)
        B[0,2] = np.cos(y)*np.cos(z)
        B[1,1] = np.cos(z)
        B[1,2] = np.cos(y)*np.sin(z)
        B[2,2] = -np.sin(y)
        B[2,0] = 1.
        return B
    
    def computeJacobian(self, q):
        if compute_first is False:
            pin.computeJointJacobians(self.rmodel, self.rdata, q)
        J = pin.getFrameJacobian(self.rmodel, self.rdata,self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        #modify J to remove the term corresponding to the base frame orientation
        Jt = np.zeros((6, 35))
        Jt[:,:3] = J[:,:3]
        Jt[:,7:] = J[:,6:]
        self.J = Jt
        return self.J
        
class CostSumNew:
    def __init__(self, rmodel,rdata):
        self.costs = dict()
        self.costnames = []
        self.nfev = 0
        self.qs = []
        self.feasibles = []
        self.costvals = []
        self.rmodel = rmodel
        self.rdata = rdata

    def reset_iter(self):
        self.nfev = 0
        self.qs = []

    def addCost(self, cost, w, name, thres=1e-4):
        cost = CostStructureNew(cost, w, name, thres=thres)
        self.costs[name] = cost
        self.costnames += [name]

    def calc(self, q):
        self.qs += [q.copy()]
        self.nfev += 1
        self.feval = 0
        self.feasibles = []
        self.res = []
        
        #compute pinocchio
        if compute_first:
            pin.computeJointJacobians(self.rmodel, self.rdata, q)
            #pin.forwardKinematics(self.rmodel, self.rdata, q)
            pin.updateFramePlacements(self.rmodel, self.rdata)
        
        for i, name in enumerate(self.costnames):
            cost = self.costs[name]
            cost.calc(q)
            self.feasibles += [cost.feasible]
            self.res += [cost.weight * cost.res]
        self.res = np.concatenate(self.res)
        self.feasibles = np.concatenate(self.feasibles)
        return self.res

    def calcDiff(self, q, recalc = False):
        #compute pinocchio
#         if compute_first:
#             pin.forwardKinematics(self.rmodel, self.rdata, q)
#             pin.updateFramePlacements(self.rmodel, self.rdata)
#             pin.computeJointJacobians(self.rmodel, self.rdata, q)
        J = []
        for i, name in enumerate(self.costnames):
            cost = self.costs[name]
            J += [cost.weight * cost.calcDiff(q, recalc)]
        self.J = np.vstack(J)
        return self.J
    
class CostStructureNew():
    def __init__(self, cost, w, name, thres=1e-4):
        self.cost = cost
        self.val = 0
        self.name = name
        self.weight = w
        self.thres = thres
        self.feasible = False

    def calc(self, q):
        self.res = self.cost.calc(q)
        self.feasible = np.abs(self.res) < self.thres
        return self.res

    def calcDiff(self, q, recalc = False):
        self.J = self.cost.calcDiff(q, recalc)
        return self.J