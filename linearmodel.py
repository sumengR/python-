#!/usr/bin/python
import numpy as np

def  _solver_mat(X,y,withintercept):
	if withintercept:
		X = np.column_stack((np.ones(len(y)),X))
	xtx = np.dot(X.T,X)
	xty = np.dot(X.T,y)
	coefs = np.linalg.solve(xtx,xty)
	return coefs


def _loss(X,y,w):
    n = len(y)
    y_hat = np.dot(X,w)
    loss = np.dot(np.transpose(y_hat-y),(y_hat-y))/(2*n)
    return loss
def _initialize(size,withintercept):
    if withintercept:
	    w = np.zeros(size+1)
    else:
	    w = np.zeros(size)
    return w

def _solver_grad(X,y,withintercept,alpha,maxtime):
    w = _initialize(X.shape[1],withintercept)
    if withintercept:
	    X = np.column_stack((np.ones(len(y)),X))
    loss_history=[]
    for i in range(maxtime):
    	loss = _loss(X,y,w)
    	loss_history.append(loss)
    	w = w - alpha*np.dot(X.T,(np.dot(X,w)-y))

    return w

def _solver_cholesky(X,y,withintercept=True):
    if withintercept:
	    X = np.column_stack((np.ones(len(y)),X))
    xtx = np.dot(X.T,X)
    xty = np.dot(X.T,y)
    L = np.linalg.cholesky(xtx)
    M = np.linalg.solve(L,xty)
    coefs = np.linalg.solve(L.T,M)
    return coefs


class linearmodel():
	"""docstring for linearre"""
	def __init__(self,withintercept=True,alpha=0.01,maxtime=100,solver="grad"):
		self.withintercept = withintercept
		self.alpha = alpha
		self.maxtime = maxtime
		self.solver = solver
		self.coefs = []
		self.intercept = 0


	def fit(self,X,y):
		if self.solver == "grad":
			self.coefs = _solver_grad(X,y,self.withintercept,self.alpha,self.maxtime)
		elif self.solver == "mat":
			self.coefs = _solver_mat(X,y,self.withintercept)
		elif self.solver == "cholesky":
			self.coefs = _solver_cholesky(X,y,self.withintercept)
		else:
			raise TypeError("solver's type is wrong")
		if self.withintercept:
			self.intercept = self.coefs[0]
			self.coefs = self.coefs[1:]

	def predict(self,X):
		y_hat = np.dot(X,np.array(self.coefs)) + self.intercept
		return y_hat