import cvxpy as cp
import time
import numpy as np
# import torch
# from cvxpylayers.torch import CvxpyLayer


# class NUPS_GPSolverTorch(object):
#     def __init__(self, n, eps, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
#         self.device = device
#         self.eps = eps

#         # Set up the problem, and we should guarantee that the problem is DGP. Here y = 1/omega !
#         self.y = cp.Variable(n, pos = True)
#         self.F = cp.Parameter(shape = (n, n), pos = True)
#         self.F.value = np.eye(n) + 1e-4
#         self.D = cp.Parameter(pos = True)
#         obj = cp.prod(self.y ** (-1 / n))
#         obj = cp.Minimize(obj)
#         constraint = eps**2 / (2 * self.D) * cp.sum(cp.multiply(self.y, self.F @ self.y)) <= 1
#         self.prob = cp.Problem(obj, [constraint])

#         assert self.prob.is_dgp()

#         self.layer = CvxpyLayer(
#             self.prob,
#             parameters = [self.F, self.D],
#             variables = [self.y],
#             gp = True
#         )


#     def solve(self, F_val : torch.Tensor, D : float):
#         y_val, = self.layer(
#             F_val, D, 
#             solver_args = {
#                 'solve_method': 'SCS',
#                 'eps_abs': 1e-4, 
#                 'eps_rel': 1e-4,
#             }
#         )
#         omega_val = 1 / y_val
#         nups_eps_val = self.eps / omega_val
#         return nups_eps_val, omega_val
    


class NUPS_GPSolverNumpy(object):
    def __init__(self, n, eps):
        self.eps = eps

        # Set up the problem, and we should guarantee that the problem is DGP. Here y = 1/omega !
        self.y = cp.Variable(n, pos = True)
        self.F = cp.Parameter(shape = (n, n), pos = True)
        self.F.value = np.eye(n) + 1e-4
        self.D = cp.Parameter(pos = True)
        obj = cp.prod(self.y ** (-1 / n))
        obj = cp.Minimize(obj)
        constraint = eps**2 / (2 * self.D) * cp.sum(cp.multiply(self.y, self.F @ self.y)) <= 1
        self.prob = cp.Problem(obj, [constraint])

        assert self.prob.is_dgp()


    def solve(self, F_val : np.ndarray, D_val : float):
        self.F.value = F_val
        self.D.value = D_val
        self.prob.solve(gp = True, solver = cp.SCS, eps_abs = 1e-5, eps_rel = 1e-5)
        y_val = self.y.value
        omega_val = 1 / y_val
        nups_eps_val = self.eps / omega_val
        return nups_eps_val, omega_val



if __name__ == '__main__':
    bs = 1
    n = 17

    # device = 'cuda'
    # solver = NUPS_GPSolverTorch(n, 0.075, 1.0, device = device)
    # for _ in range(5):
    #     u = torch.randn(bs, n)
    #     F = torch.abs(torch.bmm(u.unsqueeze(-1), u.unsqueeze(1))) + 1e-4
    #     # F.requires_grad_(True)
    #     start = time.time()
    #     res = solver.solve(F)
    #     end = time.time()
    #     # print(res)
    #     print(end - start)


    solver = NUPS_GPSolverNumpy(n, 0.075)
    for _ in range(5):
        u = np.random.randn(n)
        F = np.abs(np.outer(u, u)) + 1e-4
        F = F.astype(np.float32)
        start = time.time()
        res = solver.solve(F, 1.0)
        end = time.time()
        # print(res)
        print(end - start)