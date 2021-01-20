import numpy as np
import pinocchio as pin
from utils import *
from scipy.optimize import  minimize      
import scipy

class CostBoundFloatingBaseNew:
    """
    This cost is to keep within the joint limits
    """
    def __init__(self, bounds, margin = 1e-3): 
        self.bounds = bounds
        self.dof = bounds.shape[1]
        self.identity = np.eye(self.dof)
        self.margin = margin
        
    def calc(self, q):
        self.res = ((q - self.bounds[0]) * (q < self.bounds[0]) +  \
                    (q - self.bounds[1]) * ( q > self.bounds[1]))[1:]
        return self.res
    
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)        
        stat = (q - self.margin < self.bounds[0]) + \
                (q + self.margin > self.bounds[1])
        self.J = (stat*self.identity)[1:,1:] #the rotational velocity only has three components
        return self.J 

class CostBoundNew:
    """
    This cost is to keep within the joint limits
    """
    def __init__(self, bounds, margin = 1e-3): 
        self.bounds = bounds
        self.dof = bounds.shape[1]
        self.identity = np.eye(self.dof)
        self.margin = margin
        
    def calc(self, q):
        self.res = ((q - self.bounds[0]) * (q < self.bounds[0]) +  \
                    (q - self.bounds[1]) * ( q > self.bounds[1]))
        return self.res
    
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)        
        stat = (q - self.margin < self.bounds[0]) + \
                (q + self.margin > self.bounds[1])
        self.J = stat*self.identity
        return self.J 

    
class CostCOMBoundsNew:
    """
    This cost is to ensure that the COM is within the bounds
    """
    def __init__(self, rmodel, rdata, bounds, margin = 1e-3):
        self.rmodel = rmodel
        self.rdata  = rdata
        self.bounds = bounds
        self.margin = margin
        self.identity = np.eye(3)
        
    def calc(self, q):
        self.com = pin.centerOfMass(self.rmodel, self.rdata, q)
        self.res =  (self.com - self.bounds[0]) * (self.com < self.bounds[0]) +  \
                (self.com - self.bounds[1]) * ( self.com > self.bounds[1])
        return self.res
        
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
            
        J = pin.jacobianCenterOfMass(self.rmodel, self.rdata, q)
        stat = (self.com - self.margin < self.bounds[0]) + \
                (self.com + self.margin > self.bounds[1])
        
        self.J = (stat*self.identity).dot(J)
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
            self.weights = np.ones(rmodel.nv)
        else:
            self.weights = weights
        self.weight_matrix = np.diag(self.weights) 
        
    def calc(self, q):
        self.res = self.weights*pin.difference(self.rmodel, self.desired_posture, q)
        return self.res
    
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
        self.J = self.weight_matrix.dot(pin.dDifference(self.rmodel, self.desired_posture, q)[1])
        return self.J       

    
class CostFrameTranslationFloatingBaseNew():
    """
    The cost for frame translation of a floating base system. 
    """   
    def __init__(self, rmodel, rdata, desired_pose, ee_frame_id , weight):  
        self.rmodel = rmodel
        self.rdata  = rdata
        self.desired_pose = desired_pose
        self.ee_frame_id = ee_frame_id
        self.weight = weight
        self.weight_matrix = np.diag(weight)
        
    def calc(self, q):
        self.res = self.weight*(self.rdata.oMf[self.ee_frame_id].translation-self.desired_pose) 
        return self.res
        
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
            
        R = self.rdata.oMf[self.ee_frame_id].rotation
        J = R.dot(pin.getFrameJacobian(self.rmodel, self.rdata, self.ee_frame_id,
                                           pin.ReferenceFrame.LOCAL)[:3, :])
        
        self.J = self.weight_matrix.dot(J)

        return self.J
        
class CostFrameSE3FloatingBaseNew():
    """
    The cost for frame placement of a floating base system. 
    The orientation is described with SE3
    """   
    def __init__(self, rmodel, rdata, desired_pose, ee_frame_id , weight):  
        self.rmodel = rmodel
        self.rdata  = rdata
        self.desired_pose = desired_pose
        self.ee_frame_id = ee_frame_id
        self.weight = weight  
        self.weight_matrix = np.diag(weight)
        
    def calc(self, q):
        pose = self.rdata.oMf[self.ee_frame_id] 
        self.rMf = self.desired_pose.actInv(pose)
        self.res = pin.log(self.rMf).vector*self.weight
        return self.res
    
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
        J = np.dot(
            pin.Jlog6(self.rMf),
            pin.getFrameJacobian(self.rmodel, self.rdata, self.ee_frame_id, pin.ReferenceFrame.LOCAL))
        self.J = self.weight_matrix.dot(J)
        return self.J
    
class CostFrameRotationSE3FloatingBaseNew():
    """
    The cost for frame rotation of a floating base system. 
    The orientation is described with SE3
    """   
    def __init__(self, rmodel, rdata, desired_pose, ee_frame_id , weight):  
        self.rmodel = rmodel
        self.rdata  = rdata
        self.desired_pose = desired_pose
        self.ee_frame_id = ee_frame_id
        self.weight = weight  
        self.weight_matrix = np.diag(weight)
        
    def calc(self, q):
        pose = self.rdata.oMf[self.ee_frame_id].rotation
        self.rMf = self.desired_pose.T.dot(pose)
        self.res = pin.log(self.rMf)*self.weight
        return self.res
    
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
        J = np.dot(
            pin.Jlog3(self.rMf),
            pin.getFrameJacobian(self.rmodel, self.rdata, self.ee_frame_id, pin.ReferenceFrame.LOCAL)[3:, :])
        self.J = self.weight_matrix.dot(J)
        return self.J
        
class CostSumNew:
    def __init__(self, rmodel,rdata):
        self.costs = dict()
        self.costnames = []
        self.nfev = 0
        self.qs = []
        self.feval = 0
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
        self.feasibles = []
        self.res = []
        
        for name in self.costnames:
            cost = self.costs[name]
            cost.calc(q)
            self.feasibles += [cost.feasible]
            self.res += [cost.weight * cost.res]
        self.res = np.concatenate(self.res)
        self.feasibles = np.concatenate(self.feasibles)
        return self.res

    def calcDiff(self, q, recalc = False):
        J = []
        for i, name in enumerate(self.costnames):
            cost = self.costs[name]
            J += [cost.weight * cost.calcDiff(q, recalc)]
        self.J = np.vstack(J)
        return self.J
    
class CostStructureNew():
    def __init__(self, cost, w, name, thres=1e-4):
        self.cost = cost
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
    
    
class TalosCostProjectorNew():
    def __init__(self, cost, rmodel, rdata, cost2 = None, alpha=1, alpha2 = 1, alpha_fac = 0.2, c1 = 1e-4, mu = 1e-4, mu_ext = 1e-6, verbose = False, bounds = None):
        self.cost = cost
        self.cost2 = cost2
        self.rmodel = rmodel
        self.rdata = rdata
        self.alpha = alpha
        self.alpha2 = alpha
        self.alpha_fac = alpha_fac
        self.c1 = c1 #coefficient for line search
        self.mu = mu
        self.mu_ext  = mu_ext
        self.verbose = verbose
        self.bounds = bounds
        
    def project(self, q, maxiter = 50, ftol=1e-12, gtol=1e-12, disp=0):
        self.cost.reset_iter()
        if self.cost2 is not None:
            self.cost2.costs['posture'].cost.desired_posture = q.copy() #use the initial guess as the nominal posture
            
        for i in range(maxiter):
            q, status = self.step(q)
            if status is True: break
                
        res = {'stat': status, 'q': self.cost.qs[-1], 'qs': self.cost.qs, 'nfev': i+1,
               'feval': self.cost.feval}
        return res
    
    def update_pinocchio(self, q):
        pin.computeJointJacobians(self.rmodel, self.rdata, q)
        pin.updateFramePlacements(self.rmodel, self.rdata)
    
    def find_direction(self, q):
        self.update_pinocchio(q)
        r1 = self.cost.calc(q)
        J1 = self.cost.calcDiff(q)
        rcond =self.mu_ext + self.mu*r1.T.dot(r1) 
        J1_pinv = scipy.linalg.pinv(J1, rcond=rcond)
        dq1 = J1_pinv.dot(r1)
        
        if self.cost2 is None:
            return dq1
        
        N1 = np.eye(J1.shape[1]) - J1_pinv.dot(J1)
        r2 = self.cost2.calc(q)
        rcond2 = self.mu_ext #+  self.mu*r2.T.dot(r2) 
        J2 = self.cost2.calcDiff(q)
        J2_pinv = scipy.linalg.pinv(J2.dot(N1), rcond = rcond2)

        return dq1, r2, J2, J2_pinv, N1
    
    def step(self, q, max_iter = 20):
        #find step direction
        if self.cost2 is None:
            dq1 = self.find_direction(q)
        else:
            dq1, r2, J2, J2_pinv, N1 = self.find_direction(q)
        #line search
        #first line search
        C1 = self.cost.res.dot(self.cost.J).dot(dq1)
        c0 = np.sum(self.cost.res**2)
        alpha = self.alpha
        c = 1e10
        i = 0
        while c >= c0 - self.c1*alpha*C1 + 1e-5 :
            qn = pin.integrate(self.rmodel, q, -alpha*dq1)
            qn = clip_bounds(qn, self.bounds)
                
            self.update_pinocchio(qn)
            r1 = self.cost.calc(qn)
            c = np.sum(self.cost.res**2)
            i += 1
            alpha = alpha*self.alpha_fac
            if i > max_iter:
                if self.verbose: print('Cannot get a good step length')
                break
        q = qn
            
        if self.cost2 is not None:
            #second line search
            self.update_pinocchio(q)
            self.cost.calc(q)
            c0 = np.sum(self.cost.res**2)
            dq2 = J2_pinv.dot(r2 - J2.dot(alpha*dq1))
            dq2 = N1.dot(dq2)
            alpha2 = self.alpha2
            c = 1e10
            i = 0
            while c >= c0 + 1e-3 :
                qn = pin.integrate(self.rmodel, q, -alpha2*dq2)
                qn = clip_bounds(qn, self.bounds)

                self.update_pinocchio(qn)
                r = self.cost.calc(qn)
                c = np.sum(self.cost.res**2)
                i += 1
                alpha2 = alpha2*self.alpha_fac
                if i > max_iter:
                    if self.verbose: print('Cannot get a good step length2')
                    break

            q = qn

        feasible1 = False not in self.cost.feasibles
        if self.cost2 is None:
            return q, feasible1
        else:
            r2 = self.cost2.calc(q)
            feasible2 = False not in self.cost2.feasibles
            return q, (feasible1 and feasible2)
        
def clip_bounds(q, bounds):
    return np.clip(q, bounds[0], bounds[1])
