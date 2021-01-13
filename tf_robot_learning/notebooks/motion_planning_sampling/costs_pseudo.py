import numpy as np
import pinocchio as pin
from utils import *
from scipy.optimize import  minimize      
import scipy

compute_first = True

class CostBoundNew:
    """
    This cost is to keep within the joint limits
    """
    def __init__(self, bounds, margin = 1e-3): 
        self.bounds = bounds
        self.dof = bounds.shape[1]
        self.zeros = np.zeros(self.dof)
        self.margin = margin
        self.identity = np.eye(self.dof)
        
    def calc(self, q):
        self.res = ((q - self.bounds[0]) * (q < self.bounds[0]) +  \
                    (q - self.bounds[1]) * ( q > self.bounds[1]))[1:]
        return self.res
    
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)        
        stat = (q - self.margin < self.bounds[0]) + \
                (q + self.margin > self.bounds[1])
        self.J = (stat*self.identity)[1:,1:]
        return self.J 
    
class CostCOMBoundsNew:
    """
    This cost is to ensure that the COM is within the bounds
    """
    def __init__(self, rmodel, rdata, bounds, margin = 1e-8):
        self.rmodel = rmodel
        self.rdata  = rdata
        self.bounds = bounds
        self.zeros = np.zeros(3)
        self.margin = margin
        self.identity = np.eye(3)
        self.J = np.zeros((3, 35))
        
    def calc(self, q):
        if compute_first is False:
            pin.forwardKinematics(self.rmodel, self.rdata, q)
        self.com = pin.centerOfMass(self.rmodel, self.rdata, q)
        self.res =  (self.com - self.bounds[0]) * (self.com < self.bounds[0]) +  \
                (self.com - self.bounds[1]) * ( self.com > self.bounds[1])
        return self.res
        
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
        J = pin.jacobianCenterOfMass(self.rmodel, self.rdata, q)
        #remove the gradient due to the base orientation
        stat = (self.com - self.margin < self.bounds[0]) + \
                (self.com + self.margin > self.bounds[1])
        
        J = (stat*self.identity).dot(J)
        self.J = J
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
        self.J = (pin.dDifference(self.rmodel, self.desired_posture, q)[1]).dot(self.weight_matrix)
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
        self.J = np.zeros((3, 35))
        self.weight_matrix = np.diag(weight)
        
    def calc(self, q):
        ### Add the code to recompute your cost here
        if compute_first is False:
            pin.forwardKinematics(self.rmodel, self.rdata, q)
            pin.updateFramePlacement(self.rmodel, self.rdata, self.ee_frame_id)
        self.res = self.weight*(self.rdata.oMf[self.ee_frame_id].translation-self.desired_pose) 
        return self.res
        
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
            
        self.J = self.computeJacobian(q)
        return self.J
            
    def computeJacobian(self, q):
        if compute_first is False:
            pin.computeJointJacobians(self.rmodel, self.rdata, q)
        R = self.rdata.oMf[self.ee_frame_id].rotation
        J = R.dot(pin.getFrameJacobian(self.rmodel, self.rdata, self.ee_frame_id,
                                           pin.ReferenceFrame.LOCAL)[:3, :])
        self.J = self.weight_matrix.dot(J)
        return self.J
        
class CostFrameSE3FloatingBaseNew():
    """
    The cost for frame placement of a floating base system. 
    The orientation is described with SE3
    In this version, we remove the Jacobian due to the base orientation
    """   
    def __init__(self, rmodel, rdata, desired_pose, ee_frame_id , weight):  
        self.rmodel = rmodel
        self.rdata  = rdata
        self.desired_pose = desired_pose
        self.ee_frame_id = ee_frame_id
        self.weight = weight  
        self.weight_matrix = np.diag(weight)
        self.J = np.zeros((6, 35))
        
    def calc(self, q):
        ### Add the code to recompute your cost here
        if compute_first is False:
            pin.forwardKinematics(self.rmodel, self.rdata, q)
            pin.updateFramePlacement(self.rmodel, self.rdata, self.ee_frame_id)
        pose = self.rdata.oMf[self.ee_frame_id] 
        self.rMf = self.desired_pose.actInv(pose)
        self.res = pin.log(self.rMf).vector*self.weight
        return self.res
    
    def calcDiff(self, q, recalc = False):
        if recalc:
            self.calc(q)
        if compute_first is False:
            pin.computeJointJacobians(self.rmodel, self.rdata, q)
            pin.updateFramePlacement(self.rmodel, self.rdata, self.ee_frame_id)
        J = np.dot(
            pin.Jlog6(self.rMf),
            pin.getFrameJacobian(self.rmodel, self.rdata, self.ee_frame_id, pin.ReferenceFrame.LOCAL))
        self.J = J
        self.J = self.weight_matrix.dot(self.J)
        return self.J
        
class CostSumNew:
    def __init__(self, rmodel,rdata):
        self.costs = dict()
        self.costnames = []
        self.nfev = 0
        self.feval = 0
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
        self.feasibles = []
        self.res = []
        
        #compute pinocchio
        if compute_first:
            pin.computeJointJacobians(self.rmodel, self.rdata, q)
            pin.updateFramePlacements(self.rmodel, self.rdata)
        
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
    def __init__(self, cost, rmodel, cost2 = None, alpha=1, alpha2 = 1, alpha_fac = 0.5, c1 = 1e-4, mu = 1e-4, mu_ext = 1e-6):
        self.cost = cost
        self.cost2 = cost2
        self.rmodel = rmodel
        self.alpha = alpha
        self.alpha2 = alpha
        self.alpha_fac = alpha_fac
        self.c1 = c1
        self.mu = mu
        self.mu_ext  = mu_ext
        self.qs = []
        
    def project(self, q, maxiter = 50, ftol=1e-12, gtol=1e-12, disp=0,):
        self.cost.reset_iter()
        if self.cost2 is not None:
            self.cost2.costs['posture'].cost.desired_posture = q.copy()
            #self.cost2.costs['posture'].cost.desired_posture[-14:-7] = q0Complete[-14:-7]  # for the left hand, use the default posture
            
        for i in range(maxiter):
            q, status = self.step(q)
            if status is True: break
        res = {'stat': status, 'q': self.cost.qs[-1], 'qs': self.cost.qs, 'nfev': i+1,
               'feval': self.cost.feval}
        return res
    
    def find_direction(self, q):
        r1 = self.cost.calc(q)
        J1 = self.cost.calcDiff(q)
        rcond = self.mu*r1.T.dot(r1) + self.mu_ext
        self.qs += [q]
        J1_pinv = scipy.linalg.pinv(J1, rcond=rcond)
        #J1_pinv = np.linalg.inv(J1.T.dot(J1)+rcond*np.eye(J1.shape[1])).dot(J1.T)
        #print(np.allclose(J1_pinv, J1_pinv2))
        dq1 = J1_pinv.dot(r1)
        
        if self.cost2 is None:
            return dq1
        
        N1 = np.eye(J1.shape[1]) - J1_pinv.dot(J1)
        r2 = self.cost2.calc(q)
        rcond2 = self.mu_ext #+  self.mu*r2.T.dot(r2) 
        J2 = self.cost2.calcDiff(q)
        J2_pinv = scipy.linalg.pinv(J2.dot(N1), rcond = rcond2)
        #dq2 = np.linalg.lstsq(J2.dot(N1), r2 - J2.dot(dq1), rcond=rcond2)[0]
        #dq2 = N1.dot(dq2)
        #dq2 = np.linalg.pinv(J2.dot(N1), rcond=rcond2).dot(r2 - J2.dot(dq1))
        
        #dq = dq1 + self.alpha2*N1.dot(dq2)
        return dq1, r2, J2, J2_pinv, N1
    
    def step(self, q, max_iter = 20, line_search = True):
        #find step direction
        dq1, r2, J2, J2_pinv, N1 = self.find_direction(q)
        #line search
        if line_search:
            #first line search
            C1 = self.cost.res.dot(self.cost.J).dot(dq1)
            c0 = np.sum(self.cost.res**2)
            alpha = self.alpha
            c = 1e10
            i = 0
            while c >= c0 - self.c1*alpha*C1 + 1e-5 :
               # qn = q - alpha*dq1
                qn = pin.integrate(self.rmodel, q, -alpha*dq1)
                qn = clip_bounds(qn, self.cost.costs['joint_limit'].cost.bounds)
                r1 = self.cost.calc(qn)
                c = np.sum(self.cost.res**2)
                i += 1
                #print(alpha,c,c0)     
                alpha = alpha*self.alpha_fac
                if i > max_iter:
                    print('Cannot get a good step length')
                    break
            q = qn
            
            #second line search
            self.cost.calc(q)
            c0 = np.sum(self.cost.res**2)
            dq2 = J2_pinv.dot(r2 - J2.dot(alpha*dq1))
            dq2 = N1.dot(dq2)
            alpha2 = self.alpha2
            c = 1e10
            i = 0
            while c >= c0 + 1e-3 :
                qn = pin.integrate(self.rmodel, q, -alpha2*dq2)
                qn = clip_bounds(qn, self.cost.costs['joint_limit'].cost.bounds)
                r = self.cost.calc(qn)
                c = np.sum(self.cost.res**2)
                i += 1
                alpha2 = alpha2*self.alpha_fac
                if i > max_iter:
                    print('Cannot get a good step length2')
                    break
        
            q = qn
        else:
            q = pin.integrate(self.rmodel , q , - alpha*dq1)
            r1 = self.cost.calc(q)
        feasible1 = False not in self.cost.feasibles
        if self.cost2 is None:
            return q, feasible1
        else:
            r2 = self.cost2.calc(q)
            feasible2 = False not in self.cost2.feasibles
            return q, (feasible1 and feasible2)
        
def clip_bounds(q, bounds):
    return np.clip(q, bounds[0], bounds[1])
