
import numpy as np
import scipy as sp
import math
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simple algorithm

def vec(m):
	a = []
	a = np.ravel(m, order = 'F')
	return a
	
def hvec(m):
	n = len(m[:][0])
	m = np.tril(m)
	a = np.ravel(m, order = 'F')
	k_0 = n
	k_1 = n + 1
	i = 1
	while (len(a) != n*(n+1)/2):
		indices = list(range(k_0,k_1,1))
		a = np.delete(a, indices)
		k_0 = k_0 + n - i
		k_1 = k_1 + n - i + 1
		i = i + 1
	return a

def svec(m):
	#n = len(m[:][0])
	diagonal = np.diag(m)
	m1 = np.multiply(np.tril(m, k = -1),math.sqrt(2))
	m = m1 + np.diag(diagonal)
	return hvec(m)


#def svec(m):
#	n = len(m[:][0])
#	m1 = np.triu(m,k = 1)
#	m1 = np.multiply(m1,math.sqrt(2))
#	m = np.triu(m)
#	m = m + m1
#	return hvec(m)
	

#def creating_matrix_for_elim(x,n):
#	result = np.zeros(0)
#	begin = 0
#	end = n*n
#	n_trian = n*(n+1)/2
#	for i in range(n_trian):
#		var = np.zeros(n_trian*n*n)
#		for j in range(begin,end):
#			var[j] = x[j - begin]
#		result = np.concatenate((result,var))
#		begin = begin + n*n
#		end = end + n*n
#	return result

#def creating_matrix_for_dupl(x,n):
#	result = np.zeros(0)
#	n_trian = n*(n+1)/2
#	begin = 0
#	end = n_trian
#	for i in range(n*n):
#		var = np.zeros(n_trian*n*n)
#		for j in range(begin,end):
#			var[j] = x[j - begin]
#		result = np.concatenate((result,var))
#		begin = begin + n_trian
#		end = end + n_trian
#	return result

#def elimin_matrix(m):
#	x = vec(m)
#	y = hvec(m)
#	n = len(m[:][0])
#	n_trian = n*(n+1)/2
#	matrix = creating_matrix_for_elim(x,n)
#	matrix = np.reshape(matrix, newshape = (n_trian,n*n*n_trian))
#	l = np.reshape(np.linalg.lstsq(matrix,y)[0],newshape = (n_trian,n*n))
#	return l



def elimin_matrix(m):
	n = len(m[:][0])
	n_trian = n*(n + 1)/2
	q = np.zeros([n_trian,n*n])
	for j in range(1,n + 1):
		for i in range(j,n + 1):
			u_i_j = np.zeros(n_trian)
			u_i_j[(j-1)*n + i - j*(j-1)/2 - 1] = 1
			e_i_j = np.zeros([n,n])
			e_i_j[i-1][j-1] = 1
			u_i_j = u_i_j.reshape([n_trian,1])
			temp = vec(e_i_j)
			temp = temp.reshape([1,n*n]) 
			q = q + np.dot(u_i_j,temp)
	return q
	
	
def dupl_matrix(m):
	n = len(m[:][0])
	n_trian = n*(n + 1)/2
	q = np.zeros([n_trian,n*n])
	for j in range(1, n + 1):
		for i in range(j,n + 1):
			u_i_j = np.zeros(n_trian)
			u_i_j[(j-1)*n + i - j*(j-1)/2 - 1] = 1
			t_i_j = np.zeros([n,n])
			if (i != j):
				t_i_j[i-1][j-1] = 1
				t_i_j[j-1][i-1] = 1
			else:
				t_i_j[i-1][j-1] = 1
			u_i_j = u_i_j.reshape([n_trian,1])
			temp = vec(t_i_j)
			temp = temp.reshape([1,n*n])
			q = q + np.dot(u_i_j,temp)
	return q.transpose()
			
			
def ln_tilda(m):
	n = len(m[:][0])
	n_trian = n*(n + 1)/2
	e_n = np.ones(shape = (n,n))
	d2 = np.diag(svec(e_n))
	l_n = elimin_matrix(m)
	return np.dot(d2,l_n)
	
	
def dn_tilda(m):
	n = len(m[:][0])	
	d_n = dupl_matrix(m)
	e_n = np.ones(shape = (n,n))
	d2 = np.diag(svec(e_n))
	return np.dot(d_n,np.linalg.inv(d2))

## first iteration process
	
def symmetric_mult(a,b):
	return (np.dot(a,b) + np.dot(b,a))/2
	
def X_kron(x):
	n = len(x[:][0])
	return 0.5*(np.kron(x,np.eye(n)) + np.kron(np.eye(n),x))
	
def X_kron_tilda(x):
	n = len(x[:][0])
	return ln_tilda(x).dot(X_kron(x)).dot(dn_tilda(x)) # argument in ln_tilda and dn_tilda?

def Asvec(a): 		# np.array((m1,m2))
	m = len(a)
	n = len(a[0][:][0])
	n_trian = n*(n + 1)/2
	res = np.zeros([m,n_trian])
	res = map(svec, a)
	res = np.reshape(res, newshape = (m,n_trian))
	return res

def gamma_X_tilda(x,a):
	return Asvec(a).dot(X_kron_tilda(x)).dot(Asvec(a).T)

def ux(x,a,c):
	result = np.dot(np.linalg.inv(gamma_X_tilda(x,a)).dot(Asvec(a)).dot(X_kron_tilda(x)),svec(c))
	return result
	
def Vu(x,a,c): 
	res = np.array(c) - sum(map(lambda u,v: sum(np.dot(u,v)),ux(x,a,c),a))
	return res
	

# iteration process for svec and matrices
def iteration(X0,a,c,alpha,eps):
	w = 0
	n_trian = len(X0[:][0])*(len(X0[:][0]) + 1)/2
	cur = X0
	res0 = np.array([0]*len(svec(cur)))
	res = svec(cur)
	point = []
	point = np.array(point)
	point = np.append(point,np.trace(np.dot(c.T,X0)))
	while (np.linalg.norm(res - res0) >= eps):
		delta_svec = np.array(X_kron_tilda(cur).dot(np.eye(n_trian) - Asvec(a).T.dot(np.linalg.inv(gamma_X_tilda(cur,a))).dot(Asvec(a)).dot(X_kron_tilda(cur))).dot(svec(c)))
		delta_matrix = np.array(symmetric_mult(cur,Vu(cur,a,c)))
		cur = cur - alpha*delta_matrix
		res0 = res
		res = svec(cur) - alpha*delta_svec
		point = np.append(point,np.trace(np.dot(c.T,cur)))
		w = w + 1
	return np.array((cur,res,w,point))
	

#print np.linalg.det(X0)
#print vec(X0)
#print hvec(X0)
#print svec(X0)

#---------------------------------------------------------------------------------------

# From minimal edge


def nu_b_tilda_kron(m):
	q = np.linalg.svd(m)[0]
	d_nu = np.linalg.svd(m)[1] # here nu is sorted already
	nu = np.sort(d_nu)[::-1] 
	nu_b = nu[np.nonzero(nu)]
	r = len(nu_b)
	q_b = q[:r:,:r:]
	d_nu_b = np.diag(nu_b)
	nu_b_kron = np.diag(X_kron(d_nu_b))
	return np.dot(elimin_matrix(np.ones([r,r])),nu_b_kron)
	
	
def X_qb_tilda(m):
	n = len(m[:][0])
	q = np.linalg.svd(m)[0]
	nu = np.linalg.svd(m)[1]
	#nu = np.diag(d_nu)
	nu = np.sort(nu)[::-1]
	nu_b = nu[np.nonzero(nu)]
	r = len(nu_b)
	q_b = q[:r:,:r:]
	return dn_tilda(m).transpose().dot(np.kron(q_b,q_b)).dot(ln_tilda(np.ones([r,r])).transpose()).dot(np.diag(nu_b_tilda_kron(m))).dot(elimin_matrix(np.ones([r,r]))).dot(np.kron(q_b.transpose(),q_b.transpose())).dot(dn_tilda(m))
	
def P_X_qb_tilda(m,a):
	n = len(m[:][0])
	n_trian = n*(n + 1)/2
	return np.diag(np.ones(n_trian)) - Asvec(a).transpose().dot(np.linalg.inv(Asvec(a).dot(X_qb_tilda(m)).dot(Asvec(a).T))).dot(Asvec(a)).dot(X_qb_tilda(m))
	
def delta_svec_qb(m,a,c):
	return X_qb_tilda(m).dot(P_X_qb_tilda(m,a)).dot(svec(c))

#------------------------------------------------------------------------------------

# Final Directon 	

def g_hj_tilda(m,a,c):
	n = len(m[:][0])
	V = Vu(m,a,c)    # here theta is sorted already
	h = np.linalg.svd(V)[0]
	Dtheta = np.linalg.svd(V)[1]
	theta = np.sort(Dtheta)[::-1]
	theta_b = theta[np.nonzero(theta)]
	s = len(theta_b)
	hj = h[:][s-1]
	hj = hj.reshape([1,n])
	return np.kron(hj,hj).dot(dn_tilda(m))
	
def e_hj_tilda_kron(m,a,c):
	return g_hj_tilda(m,a,c).transpose().dot(g_hj_tilda(m,a,c))

def psi(m,a,c):
	return g_hj_tilda(m,a,c).dot(Asvec(a).transpose()).dot(np.linalg.inv(Asvec(a).dot(X_qb_tilda(m)).dot(Asvec(a).T))).dot(Asvec(a)).dot(g_hj_tilda(m,a,c).transpose())

def cfunc(eps,m,a,c):
	return eps/(1 + eps*psi(m,a,c))
	
def delta_svec_qb_hj(m,a,c,eps):
	temper = X_qb_tilda(m).dot(P_X_qb_tilda(m,a)) + cfunc(eps,m,a,c)*P_X_qb_tilda(m,a).transpose().dot(e_hj_tilda_kron(m,a,c)).dot(P_X_qb_tilda(m,a))
	return temper.dot(svec(c))
 

def changing(l,n):
	ans = np.zeros([n,n])
	k10 = 0
	k0 = 0
	k1 = n
	while(k0 != n):
		ans[k0][k0:n] = l[k10:k1]
		k10 = k1
		k0 = k0 + 1
		k1 = k1 + (n - k0)
	return (ans + ans.T)/math.sqrt(2) - np.diag(ans.diagonal())*(math.sqrt(2) - 1)


def iteration_min_edge(X0,a,c,alpha,eps):
	w = 0
	n_trian = len(X0[:][0])*(len(X0[:][0]) + 1)/2
	cur = X0
	res0 = np.array([0]*len(svec(cur)))
	res = svec(cur)
	while (np.linalg.norm(res - res0) >= eps):
		delta_svec = delta_svec_qb(cur,a,c)
		delta_matrix = np.array(symmetric_mult(cur,Vu(cur,a,c)))
		cur = cur - alpha*delta_matrix
		res0 = res
		res = svec(cur) - alpha*delta_svec
		w = w + 1
	return np.array((cur,res,w))
	
def iteration_final_dir(X0,a,c,alpha,eps):
	w = 0
	n_trian = len(X0[:][0])*(len(X0[:][0]) + 1)/2
	cur = X0
	res0 = np.array([0]*len(svec(cur)))
	res = svec(cur)
	while (np.linalg.norm(res - res0) >= eps):
		delta_svec = delta_svec_qb_hj(cur,a,c,eps)
		delta_matrix = np.array(symmetric_mult(cur,Vu(cur,a,c)))
		cur = cur - alpha*delta_matrix
		res0 = res
		res = svec(cur) - alpha*delta_svec
		w = w + 1
	return np.array((cur,res,w))	

def Avec(a):
	m = len(a)
	n = len(a[0][:][0])
	n_trian = n*(n + 1)/2
	res = np.zeros([m,n*n])
	res = map(vec, a)
	res = np.reshape(res, newshape = (m,n*n))
	return res
	
def invvec(a):
	n = int(math.sqrt(len(a)))
	res = np.zeros([n,n])
	k0 = 0
	k1 = n
	for i in range(n):
		res[i] = a[k0:k1] 
		k0 = k1
		k1 = n*(i+2)
	return res.T
	
def wk_max(m,a,eps,c):
	n = len(m[:][0])
	n_trian = n*(n+1)/2
	q = np.linalg.svd(m)[0]
	d_nu = np.linalg.svd(m)[1] # here nu is sorted already
	nu = np.sort(d_nu)[::-1] 
	nu_b = nu[np.nonzero(nu)]
	r = len(nu_b)
	q_b = q[:r:,:r:]
	d_nu_b = np.diag(nu_b)
	delta_vec = np.diag(np.diag(X_kron(d_nu_b))).dot(np.kron(q_b.T,q_b.T)).dot(dn_tilda(m)).dot(np.eye(n_trian) - cfunc(eps,m,a,c)*Asvec(a).T.dot(np.linalg.inv(gamma_X_tilda(m,a))).dot(Asvec(a)).dot(e_hj_tilda_kron(m,a,c))).dot(svec(Vu(m,a,c)))
	delta_zk = invvec(delta_vec)
	zk_hat = np.linalg.inv(d_nu_b).dot(delta_zk)
	w = np.linalg.eigvals(zk_hat)
	return np.amax(w)


def iteration_min_edge(X0,a,c,alpha,eps):
	w = 0
	n = len(X0[:][0])
	n_trian = len(X0[:][0])*(len(X0[:][0]) + 1)/2
	cur = X0
	res0 = np.array([0]*len(svec(cur)))
	res = svec(cur)
	point = []
	point = np.array(point)
	point = np.append(point,np.trace(np.dot(c.T,X0)))
	while (np.linalg.norm(res - res0) >= eps):
		delta_svec = delta_svec_qb(cur,a,c)
		res0 = res
		res = svec(cur) - alpha*delta_svec
		cur = cur - alpha*changing(delta_svec,n)
		point = np.append(point,np.trace(np.dot(c.T,cur)))
		w = w + 1
	return np.array((cur,res,w,point))

def iteration_final_dir(X0,a,c,alpha,eps):
	w = 0
	n = len(X0[:][0])
	n_trian = len(X0[:][0])*(len(X0[:][0]) + 1)/2
	cur = X0
	res0 = np.array([0]*len(svec(cur)))
	res = svec(cur)
	point = []
	point = np.array(point)
	point = np.append(point,np.trace(np.dot(c.T,X0)))
	while (np.linalg.norm(res - res0) >= eps):
		delta_svec = delta_svec_qb_hj(cur,a,c,eps)
		res0 = res
		res = svec(cur) - alpha*delta_svec
		cur = cur - alpha*changing(delta_svec,n)
		point = np.append(point,np.trace(np.dot(c.T,cur)))
		w = w + 1
	return np.array((cur,res,w,point))	



#y1 = iteration(X0,a,c1,0.001,0.2)[3]
#y2 = iteration_min_edge(X0,a,c1,0.001,0.2)[3]
#y3 = iteration_final_dir(X0,a,c1,0.001,0.2)[3]

#w1 = iteration(X0,a,c1,0.001,0.2)[2]
#w2 = iteration_min_edge(X0,a,c1,0.001,0.2)[2]
#w3 = iteration_final_dir(X0,a,c1,0.001,0.2)[2]

#x1 = range(w1)
#x2 = range(w2)
#x3 = range(w3)

#np.savetxt('test_1_02.txt',y1,delimiter = ',')
#np.savetxt('test_2_02.txt',y2,delimiter = ',')
#np.savetxt('test_3_02.txt',y3,delimiter = ',')


# New step!

def iteration_step(X0,a,c,eps):
	w = 0
	n_trian = len(X0[:][0])*(len(X0[:][0]) + 1)/2
	cur = X0
	res0 = np.array([0]*len(svec(cur)))
	res = svec(cur)
	point = []
	point = np.array(point)
	point = np.append(point,np.trace(np.dot(c.T,X0)))
	while (np.linalg.norm(res - res0) >= eps):
		delta_svec = np.array(X_kron_tilda(cur).dot(np.eye(n_trian) - Asvec(a).T.dot(np.linalg.inv(gamma_X_tilda(cur,a))).dot(Asvec(a)).dot(X_kron_tilda(cur))).dot(svec(c)))
		delta_matrix = np.array(symmetric_mult(cur,Vu(cur,a,c)))
		cur = cur - delta_matrix/wk_max(cur,a,eps,c)
		res0 = res
		res = svec(cur) - delta_svec/wk_max(cur,a,eps,c)
		point = np.append(point,np.trace(np.dot(c.T,cur)))
		w = w + 1
	return np.array((cur,res,w,point))


def iteration_min_edge_step(X0,a,c,eps):
	w = 0
	n = len(X0[:][0])
	n_trian = len(X0[:][0])*(len(X0[:][0]) + 1)/2
	cur = X0
	res0 = np.array([0]*len(svec(cur)))
	res = svec(cur)
	point = []
	point = np.array(point)
	point = np.append(point,np.trace(np.dot(c.T,X0)))
	while (np.linalg.norm(res - res0) >= eps):
		delta_svec = delta_svec_qb(cur,a,c)
		res0 = res
		res = svec(cur) - delta_svec/wk_max(cur,a,eps,c)
		cur = cur - changing(delta_svec,n)/wk_max(cur,a,eps,c)
		point = np.append(point,np.trace(np.dot(c.T,cur)))
		w = w + 1
	return np.array((cur,res,w,point))

def iteration_final_dir_step(X0,a,c,eps):
	w = 0
	n = len(X0[:][0])
	n_trian = len(X0[:][0])*(len(X0[:][0]) + 1)/2
	cur = X0
	res0 = np.array([0]*len(svec(cur)))
	res = svec(cur)
	point = []
	point = np.array(point)
	point = np.append(point,np.trace(np.dot(c.T,X0)))
	while (np.linalg.norm(res - res0) >= eps):
		delta_svec = delta_svec_qb_hj(cur,a,c,eps)
		res0 = res
		res = svec(cur) - delta_svec/wk_max(cur,a,eps,c)
		cur = cur - changing(delta_svec,n)/wk_max(cur,a,eps,c)
		point = np.append(point,np.trace(np.dot(c.T,cur)))
		w = w + 1
	return np.array((cur,res,w,point))






