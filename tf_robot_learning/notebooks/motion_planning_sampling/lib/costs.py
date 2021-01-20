import numpy as np
import pinocchio as pin
from utils import *
from scipy.optimize import  minimize

class CostFrameTranslation:
    """
    The cost for frame translation.
    """
    def __init__(self, rmodel, rdata, desired_pos, ee_frame_id , viz=None):  
        self.rmodel = rmodel
        self.rdata  = rdata
        self.viz = viz
        self.desired_pos = desired_pos
        self.ee_frame_id = ee_frame_id
        
    def calc(self, q):
        pin.forwardKinematics(self.rmodel, self.rdata, q)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        pose = self.rdata.oMf[self.ee_frame_id] 
        self.pos, self.ori = pose.translation, pose.rotation
        return 0.5*np.sum((self.pos-self.desired_pos)**2)
    
    def calcDiff(self, q, recalc = False):
        self.J = self.computeJacobian(q)[:3,:]
        if recalc:
            self.calc(q)
        self.Cx = self.J.T.dot(self.pos-self.desired_pos)
        return self.Cx
    
    def computeJacobian(self, q):
        pin.forwardKinematics(self.rmodel,self.rdata,q)
        pin.updateFramePlacements(self.rmodel,self.rdata)
        pin.computeJointJacobians(self.rmodel, self.rdata, q)
        J = pin.getFrameJacobian(self.rmodel, self.rdata,self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J

    def callback(self, q):
        if viz is None: 
            return
       
    
class CostFrameRPY:
    """
    The cost for frame placement. 
    The orientation is described with YPR
    """
    def __init__(self, rmodel, rdata, desired_pose, ee_frame_id , sel_vector, viz=None):  # add any other arguments you like
        self.rmodel = rmodel
        self.rdata  = rdata
        self.viz = viz
        self.desired_pose = desired_pose
        self.ee_frame_id = ee_frame_id
        self.sel_vector = sel_vector
        
    def calc(self, q):
        ### Add the code to recompute your cost here
        pin.forwardKinematics(self.rmodel, self.rdata, q)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        pose = self.rdata.oMf[self.ee_frame_id] 
        self.pos, self.ori = pose.translation, pose.rotation
        self.rpy = mat2euler(self.ori)
        self.r_pos = self.sel_vector[:3]*(self.pos-self.desired_pose[:3]) 
        self.r_ori = self.sel_vector[3:]*(self.rpy-self.desired_pose[3:])
        return 0.5*np.sum(self.r_pos**2 + self.r_ori**2)
    
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
            
        self.J = self.computeJacobian(q)
        self.B = self.compute_Bz(*self.rpy)
        self.Cx1 = self.J[:3].T.dot(self.r_pos)
        self.Cx2 = self.J[3:].T.dot(np.linalg.inv(self.B).T).dot(self.r_ori)
        self.Cx = self.Cx1 + self.Cx2
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
        pin.forwardKinematics(self.rmodel,self.rdata,q)
        pin.updateFramePlacements(self.rmodel,self.rdata)
        pin.computeJointJacobians(self.rmodel, self.rdata, q)
        J = pin.getFrameJacobian(self.rmodel, self.rdata,self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J

    def callback(self, q):
        if viz is None: 
            return

class CostBound:
    def __init__(self, bounds, viz=None):  # add any other arguments you like
        self.bounds = bounds
        self.dof = bounds.shape[1]
        self.zeros = np.zeros(self.dof)
        
    def calc(self, q):
        self.rlb_min = np.min([q - self.bounds[0], self.zeros],axis=0)
        self.rlb_max = np.max([q-self.bounds[1], self.zeros],axis=0)
        return 0.5*np.sum(self.rlb_min**2 + self.rlb_max**2)
    
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
        
        self.J = self.rlb_min + self.rlb_max
        return self.J
    



class CostCOM:
    """
    This cost is for specifying a desired COM position
    """
    def __init__(self, rmodel, rdata, desired_pos):
        self.rmodel = rmodel
        self.rdata  = rdata
        self.desired_pos = desired_pos
        
    def calc(self, q):
        pin.forwardKinematics(self.rmodel, self.rdata, q)
        self.com = pin.centerOfMass(self.rmodel, self.rdata, q)
        return 0.5*np.sum((self.com-self.desired_pos)**2)
    
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
        J = pin.jacobianCenterOfMass(self.rmodel, self.rdata, q)
        #remove the gradient due to the base orientation
        self.J = np.hstack([J[:,:3], np.zeros((3,4)), J[:,6:]])
        self.Cx = self.J.T.dot(self.com-self.desired_pos)
        return self.Cx
    
class CostFrameRPYFloatingBase(CostFrameRPY):
    """
    The cost for frame placement of a floating base system. 
    The orientation is described with YPR
    In this version, we remove the Jacobian due to the base orientation
    """   
    def computeJacobian(self, q):
        pin.forwardKinematics(self.rmodel,self.rdata,q)
        pin.updateFramePlacements(self.rmodel,self.rdata)
        pin.computeJointJacobians(self.rmodel, self.rdata, q)
        J = pin.getFrameJacobian(self.rmodel, self.rdata,self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        #modify J to remove the term corresponding to the base frame orientation
        J = np.hstack([J[:,:3], np.zeros((6,4)), J[:,6:]])
        return J
    
class ActivationQuadraticBarrier():
    def __init__(self, bounds):
        self.bounds = bounds
        self.dof = bounds.shape[1]
        self.zeros = np.zeros(self.dof)
        
    def calc(self, r):
        self.rlb_min = np.min([r - self.bounds[0], self.zeros],axis=0)
        self.rlb_max = np.max([r - self.bounds[1], self.zeros],axis=0)
        return 0.5*np.sum(self.rlb_min**2 + self.rlb_max**2)

    def calcDiff(self, r, recalc=False):
        if recalc:
            self.calc(r)
        
        self.Ar = self.rlb_min + self.rlb_max
        return self.Ar
    
class CostCOMBounds:
    """
    This cost is to ensure that the COM is within the bounds
    """
    def __init__(self, rmodel, rdata, bounds):
        self.rmodel = rmodel
        self.rdata  = rdata
        self.bounds = bounds
        self.activation = ActivationQuadraticBarrier(bounds)
        
    def calc(self, q):
        pin.forwardKinematics(self.rmodel, self.rdata, q)
        self.r = pin.centerOfMass(self.rmodel, self.rdata, q)
        self.cost = self.activation.calc(self.r)
        return self.cost
    
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
        J = pin.jacobianCenterOfMass(self.rmodel, self.rdata, q)
        #remove the gradient due to the base orientation
        self.J = np.hstack([J[:,:3], np.zeros((3,4)), J[:,6:]])
        
        self.activation.calcDiff(self.r)
        
        self.Lx = self.J.T.dot(self.activation.Ar)
        return self.Lx
    
class CostPosture:
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
        return 0.5*np.sum((self.weights*(q-self.desired_posture))**2)
    
    def calcDiff(self, q, recalc = False):
        return self.weights*(q - self.desired_posture)


class CostStructure():
    def __init__(self, cost, w, name, thres=1e-4):
        self.cost = cost
        self.val = 0
        self.name = name
        self.weight = w
        self.thres = thres
        self.feasible = False

    def calc(self, q):
        self.val = self.cost.calc(q)
        self.feasible = self.val < self.thres
        return self.val

    def calcDiff(self, q):
        return self.cost.calcDiff(q)


class CostSum:
    def __init__(self):
        self.costs = dict()
        self.costnames = []
        self.nfev = 0
        self.qs = []
        self.feasibles = []
        self.costvals = []

    def reset_iter(self):
        self.nfev = 0
        self.qs = []

    def addCost(self, cost, w, name, thres=1e-4):
        cost = CostStructure(cost, w, name, thres=thres)
        self.costs[name] = cost
        self.costnames += [name]

    def calc(self, q):
        self.qs += [q.copy()]
        self.nfev += 1
        self.feval = 0
        self.feasibles = []
        self.costvals = []
        for i, name in enumerate(self.costnames):
            cost = self.costs[name]
            cost.calc(q)
            self.feasibles += [cost.feasible]
            self.costvals += [cost.val]
            self.feval += cost.weight * cost.val
        return self.feval

    def calcDiff(self, q):
        J = np.zeros(len(q))
        for i, name in enumerate(self.costnames):
            cost = self.costs[name]
            J += cost.weight * cost.calcDiff(q)
        return J


class TalosCostProjector():
    def __init__(self, cost):
        self.cost = cost

    def __call__(self, xk):
        if False not in self.cost.feasibles:
            # print('Stop at iteration!' + str(self.cost.nfev))
            raise Exception

    def project(self, q, ftol=1e-12, gtol=1e-12, disp=0, maxiter=300):
        # update the variables
        self.cost.reset_iter()
        self.cost.costs['posture'].cost.desired_posture = q.copy()
        self.cost.costs['posture'].cost.desired_posture[-14:-7] = q0Complete[
                                                                  -14:-7]  # for the left hand, use the default posture
        status = False
        try:
            res = minimize(self.cost.calc, q, method='l-bfgs-b', jac=self.cost.calcDiff, callback=self.__call__,
                           options={'ftol': ftol, 'gtol': gtol, 'disp': disp, 'maxiter': maxiter})
        except:
            # Optimization manage to get solution
            status = True
        res = {'stat': status, 'q': self.cost.qs[-1], 'qs': self.cost.qs, 'nfev': self.cost.nfev,
               'feval': self.cost.feval}
        return res
