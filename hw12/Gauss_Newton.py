import numpy as np
# import scipy

def Loss(r):
    return 0.5 * np.sum(r ** 2)  # 0.5*sum(r^2)

#todo: choose alpha
def GaussNewton(Res_and_Jac, x, ITER_MAX, TOL):
    r, J = Res_and_Jac(x)
    n, d = np.shape(J)
    lossvals = np.zeros(ITER_MAX)
    gradnormvals = np.zeros(ITER_MAX)
    lossvals[0] = Loss(r)
    Jtrans = np.transpose(J)
    grad = np.matmul(Jtrans, r)  # grad = J^\top r
    Bmatr = np.matmul(Jtrans, J)  # Bmatr = J^\top J
    gradnorm = np.linalg.norm(grad)
    gradnormvals[0] = gradnorm
    # print("iter 0: loss = ", lossvals[0], " gradnorm = ", gradnorm)
    # start iterations
    iter = 1
    while gradnorm > TOL and iter < ITER_MAX:
        Bmatr = np.matmul(Jtrans, J) + (1.e-6) * np.eye(d)  # B = J^\top J
        p = (-1) * np.linalg.solve(Bmatr, grad)  # p = -Bmatr^{-1}grad
        # print("norm_p = ", np.linalg.norm(p))
        # norm_p = np.linalg.norm(p)
        # evaluate the progress
        xnew = x + p
        rnew, Jnew = Res_and_Jac(xnew)
        lossnew = Loss(rnew)
        #accept step
        x = xnew
        r = rnew
        J = Jnew
        Jtrans = np.transpose(J)
        grad = np.matmul(Jtrans, r)
        gradnorm = np.linalg.norm(grad)

        lossvals[iter] = lossnew
        gradnormvals[iter] = gradnorm
        # print(f"LM, iter #{iter}: loss = {lossvals[iter]:.4e}, gradnorm = {gradnorm:.4e}")
        iter = iter + 1
    print(f"GN, iter #{iter-1}: loss = {lossvals[iter-1]:.4e}, gradnorm = {gradnorm:.4e}")
    return x, iter, lossvals[0:iter], gradnormvals[0:iter]