import numpy as np
import torch
import scipy.sparse as sp

def similar_csr(A, data):
    return sp.csr_matrix((data, A.indices, A.indptr), shape=A.shape)

def similar_coo(A, data):
    return sp.coo_matrix((data, (A.row, A.col)), shape=A.shape)

def c_fd(f, A, args=None, h=1e-3):
    df_dA = None
    is_sparse = False
    clone = np.copy
    similar_sparse = None
    
    if isinstance(A, torch.Tensor):
        if len(A.shape) == 1:
            A = A.reshape((-1, 1))
        df_dA = torch.zeros_like(A)
        clone = torch.clone
    elif isinstance(A, np.ndarray):
        if len(A.shape) == 1:
            A = np.expand_dims(A, axis=0)
        df_dA = np.zeros_like(A)
    elif isinstance(A, sp.csr_matrix):
        df_dA = np.zeros_like(A.data)
        is_sparse = True
        similar_sparse = similar_csr
    elif isinstance(A, sp.coo_matrix):
        df_dA = np.zeros_like(A.data)
        is_sparse = True
        similar_sparse = similar_coo
    
    if not is_sparse:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A_fwd = clone(A)
                A_fwd[i,j] += h
                A_bwd = clone(A)
                A_bwd[i,j] -= h
                
                f_fwd = f(A_fwd)
                f_bwd = f(A_bwd)
                
                df_dA[i,j] = (f_fwd - f_bwd) / (2*h)
        return df_dA
    else:
        for i in range(A.shape[0]):
            A_fwd = clone(A)
            A_fwd[i] += h
            
            A_bwd = clone(A)
            A_bwd[i] -= h
            
            f_fwd = f(similar_sparse(A, A_fwd))
            f_bwd = f(similar_sparse(A, A_bwd))
            
            df_dA[i] = (f_fwd - f_bwd) / (2*h)
        return similar_sparse(A, df_dA)
        
# Atomic torch methods
        
class ScaleVecPrimitive(torch.autograd.Function):
    # computes x[i] = x[i] * alpha, x[j] = x[j]
    
    @staticmethod
    def forward(ctx, x, i, alpha):
        z = torch.clone(x)
        z[i] = z[i] * alpha
        ctx.alpha = alpha
        ctx.i = i
        ctx.xi = x[i].detach()
        return z
    
    @staticmethod
    def backward(ctx, x_grad_in):
        #print(args)
        alpha = ctx.alpha
        i = ctx.i
        
        x_grad = x_grad_in.clone()
        x_grad[i] *= alpha.item()
        
        return x_grad, None, ctx.xi * x_grad_in[i]
    
scale = ScaleVecPrimitive.apply

class TriSolveSub(torch.autograd.Function):
    # computes x[i] = x[i] - L[i, j] * x[j]
    
    @staticmethod
    def forward(ctx, x, i, j, Mij):
        z = torch.clone(x)
        z[i] = z[i] - Mij * x[j]
        
        ctx.Mij = Mij.detach()
        ctx.i = i
        ctx.j = j
        ctx.xj = x[j].detach()
        return z
    
    @staticmethod
    def backward(ctx, x_grad):
        i, j = ctx.i, ctx.j
        mij = ctx.Mij
        
        grad_x = torch.clone(x_grad)
        grad_x[j] += -mij * x_grad[i]
        
        return grad_x, None, None, -ctx.xj * x_grad[i]
    
trisolvesub = TriSolveSub.apply

class LMRowOp(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, X, i, j, alpha):
        Y = X.clone()
        Y[i] = Y[i] + alpha * Y[j]
        
        ctx.alpha = alpha.detach()
        ctx.i = i
        ctx.j = j
        ctx.save_for_backward(X, Y)
        return Y
    
    @staticmethod
    def backward(ctx, X_grad):  
        i, j = ctx.i, ctx.j
        alpha = ctx.alpha
        
        grad_X = X_grad.clone()
        grad_X[j] += X_grad[i] * alpha
        X, Y = ctx.saved_tensors
        
        #grad_alpha = torch.zeros_like(X_grad)
        #grad_alpha[i] = X[j]
        
        return grad_X, None, None, torch.sum(X_grad[i] * X[j])

class RMRowOp(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, X, i, j, alpha):
        Y = X.clone()
        Y[:,j] = Y[:,j] + alpha * Y[:,i]
        
        ctx.alpha = alpha.detach()
        ctx.i = i
        ctx.j = j
        ctx.save_for_backward(X, Y)
        return Y
    
    @staticmethod
    def backward(ctx, X_grad):
        i, j = ctx.i, ctx.j
        alpha = ctx.alpha
        
        grad_X = X_grad.clone()
        grad_X[:, i] += grad_X[:, j] * alpha
        X, Y = ctx.saved_tensors
        
        #grad_alpha = torch.zeros_like(X_grad)
        #grad_alpha[:, j] = X[:, i]
        
        return grad_X, None, None, torch.sum(X_grad[:, j] * X[:, i])
    
lmrow = LMRowOp.apply
rmrow = RMRowOp.apply

def lu_factor(A):
    ''' A -> (L, U) '''
    
    N = A.shape[0]
    
    U = A
    L = torch.eye(N)
    
    for j in range(N-1):
        for i in range(j+1, N):
            if U[i,j] == 0:
                continue
            
            alpha = U[i,j] / U[j,j]
            U = lmrow(U, i, j, -alpha)
            L = rmrow(L, i, j,  alpha)
    return L, U

def lower_tri_solve(L, b, unit_diag=True):
    '''Lx = b'''
    
    x = b
    n = len(b)
    for i in range(n):
        for j in range(0, i):
            if L[i,j] != 0:
                x = trisolvesub(x, i, j, L[i, j])
        if not unit_diag:
            x = scale(x, i, 1/L[i, i])
    return x

def upper_tri_solve(U, b, unit_diag=False):
    '''Ux = b'''
    
    x = b
    n = len(b)
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            if U[i,j] != 0:
                x = trisolvesub(x, i, j, U[i, j])
        if not unit_diag:
            x = scale(x, i, 1/U[i, i])
    return x

def lu_solve(L, U, b):
    '''LUx = b'''
    
    return upper_tri_solve(U, lower_tri_solve(L, b))