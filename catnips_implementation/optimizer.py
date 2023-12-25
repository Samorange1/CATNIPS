import numpy as np
import sympy as sp
import scipy
from scipy.special import comb
from qpsolvers import solve_qp

class QuadProg:
    def __init__(self, deg , derivative_order , start, end, box_coords):
        self.integral_soln = None
        self.deg = deg
        self.derivative_order = derivative_order
        self.start = np.asarray(start)
        self.end = np.asarray(end)
        self.box_coords = box_coords
        self.t = sp.symbols('t')
        self.t_vec, self.curve_mat = self.get_curve_params(self.deg, self.derivative_order)
        t_vec_0 , _ = self.get_curve_params(self.deg, 0)

        self.b_t0 = self.evaluate_bezier_curve(t_vec_0, 0)
        self.b_t1 = self.evaluate_bezier_curve(t_vec_0, 1)
        self.P = None
        self.num_boxes = len(self.box_coords)
        self.lb = None
        self.ub = None


    def get_curve_params(self, deg, derivative_order):

        t_poly = np.poly1d(np.ones(deg+1))

        t_der = np.polyder(t_poly, derivative_order)
        t_der = np.pad(t_der, (0, len(t_poly) - len(t_der)), 'constant')
        t_der = np.flip(t_der)

        count = 0
        t_vec = sp.zeros(deg+1,1)
        for i in range(t_der.shape[0]):
            if t_der[i] != 0:
                t_vec[i] = t_der[i] * (self.t**count)
                count+=1

        t_vec = np.asarray(t_vec)
        t_vec = t_vec.reshape(-1,)

        curve_mat = np.zeros((deg+1, deg+1), dtype=int)
        for i in range(deg+1):
            curve_mat[i, :i + 1] = [(-1)**(i-j) * int(comb(deg, i) * comb(i,j)) for j in range(i + 1)]


        return t_vec, curve_mat

    def evaluate_bezier_curve(self, t_vec, ti):
        t_sp = sp.Matrix(t_vec)
        t_tes = t_sp.subs(self.t,ti)
        b_t = t_tes.T @ self.curve_mat
        return b_t

    def compute_integral_jacobian(self):

        con_points = sp.MatrixSymbol('s', self.deg+1,3)
        s_vec = sp.Matrix(con_points).reshape(((self.deg+1) *3), 1)

        b_t = self.t_vec @ self.curve_mat @ con_points
        norm = 0

        for i in range(b_t.shape[0]):
            norm += b_t[i]**2

        norm = sp.simplify(norm)
        integral_result = sp.integrate(sp.FU['TR8'](norm), (self.t,0,1))
        integral_result = sp.Matrix([integral_result])
        jac = integral_result.jacobian(s_vec)
        jac2 = jac.jacobian(s_vec)

        return jac2

    def compute_diff_jacobian(self):
        con_points = sp.MatrixSymbol('s', self.deg+1,3)
        s_mat = sp.Matrix(con_points)

        diff = sp.Matrix([[s_mat[i+1, j] - s_mat[i,j] for i in range(s_mat.shape[0]-1) for j in range(s_mat.shape[1])]])

        diff = sp.Matrix(sp.MatMul(diff, diff.T))
        jac = diff.jacobian(s_mat.reshape(((self.deg+1) *3), 1))
        jac2 = jac.jacobian(s_mat.reshape(((self.deg+1) *3), 1))
        return jac2


    def compute_objective_function(self):


        integral_jacobian = self.compute_integral_jacobian()
        diff_jacobian = self.compute_diff_jacobian()

        obj_jacobian = integral_jacobian + diff_jacobian

        obj_jacobian = np.asarray(obj_jacobian, dtype = np.float64)

        self.P = np.kron(np.eye(self.num_boxes, dtype=float), obj_jacobian)


    def compute_constraints(self):
        s_vec = sp.MatrixSymbol('s', (self.deg+1)*self.num_boxes, 3)
        s_vec = sp.Matrix(s_vec)
        s_vec = s_vec.reshape(s_vec.shape[0]*s_vec.shape[1],1)
        block = ((self.deg+1)*3)
        constraints = None
        # starting point constraint
        s_1 = s_vec[0 : block]
        s_1 = sp.Matrix(s_1)
        s_1 = s_1.reshape((self.deg+1),3)

        constraint = self.b_t0 @ s_1
        self.A = np.asarray(constraint.jacobian(s_vec).copy(), dtype = np.float64)

        # continuty constraint
        for i in range(self.num_boxes-1):
            s_i  = s_vec[block*i : block*i + block]
            s_i = sp.Matrix(s_i)
            s_i = s_i.reshape((self.deg+1),3)

            s_i_1  = s_vec[block*(i+1) : block*(i+1) + block]
            s_i_1 = sp.Matrix(s_i_1)
            s_i_1 = s_i_1.reshape((self.deg+1),3)

            constraint = (self.b_t1 @ s_i) - (self.b_t0 @ s_i_1)
            jac = np.asarray(constraint.jacobian(s_vec), dtype = np.float64)
            self.A = np.concatenate((self.A, jac), axis = 0)

        # ending point constraint
        s_e  = s_vec[-block:]
        s_e = sp.Matrix(s_e)
        s_e = s_e.reshape((self.deg+1),3)
        constraint = self.b_t1 @ s_e
        jac = np.asarray(constraint.jacobian(s_vec), dtype = np.float64)
        self.A = np.concatenate((self.A, jac), axis = 0)

    def get_bounds(self):
        lower_bounds = []
        upper_bounds = []

        for box in self.box_coords:
            mins = np.min(box, axis = 0).reshape(1,-1)
            maxs = np.max(box, axis = 0).reshape(1,-1)
            lb = np.repeat(mins, self.deg+1, axis = 0)
            ub = np.repeat(maxs, self.deg+1, axis = 0)

            lower_bounds.append(lb)
            upper_bounds.append(ub)

        lower_bounds = np.concatenate(lower_bounds, axis = 0).reshape(-1,1)
        upper_bounds = np.concatenate(upper_bounds, axis = 0).reshape(-1,1)

        return lower_bounds, upper_bounds

    def optimize(self):

        self.lb, self.ub = self.get_bounds()
        self.compute_objective_function()
        self.compute_constraints()

        b = np.zeros((self.A.shape[0], 1), dtype=np.float64)
        b[0:3, -1] = self.start
        b[-3:, -1] = self.end

        q = np.zeros((self.num_boxes * (self.deg+1) * 3, 1))

        solution = solve_qp(P=self.P, A = self.A, q = q, b =b, lb = self.lb, ub = self.ub,solver = 'cvxopt', verbose=True)
        return solution