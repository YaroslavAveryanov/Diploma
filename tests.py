from diploma import *

c1 = np.array([[1,2,3],[2,6,7],[3,7,11]])
a1 = np.array([[2,2,14],[-1,1,0],[-4,3,2]])
a2 = np.array([[1,2,13],[1,9,8],[-5,1,1]])
a3 = np.array([[2,1,5],[1,3,-7],[-6,7,2]])
X0 = np.array([[31,21,14],[21,50,67],[14,67,135]])
alpha = 0.001
a = np.array((a1,a2,a3))

epsillon = 0.2
for i in range(5):	
	t0 = time.clock()
	print iteration_step(X0,a,c1, epsillon - i*0.02)[0:3]
	print "eps =",epsillon - i*0.02
	print "time =",time.clock() - t0
	
	t0 = time.clock()
	print iteration_min_edge_step(X0,a,c1, epsillon - i*0.02)[0:3]
	print "eps =",epsillon - i*0.02
	print "time =",time.clock() - t0

	
	t0 = time.clock()
	print iteration_final_dir_step(X0,a,c1, epsillon - i*0.02)[0:3]
	print "eps =",epsillon - i*0.02
	print "time =",time.clock() - t0
	


