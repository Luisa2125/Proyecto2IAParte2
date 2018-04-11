import random as rd
import numpy as np
import math as mt
import matplotlib.pyplot as plt

def read():
	points = []
	x = []
	y = []
	file  = open("DS_3Clusters_999Points.txt", "r")
	for point in file:
		tup = (float)(point.split(",")[0]),(float)(point.split(",")[1])
		x.append((float)(point.split(",")[0]))
		y.append( (float)(point.split(",")[1]))
		points.append(tup)
	points = np.array(points)
	print(max(x),max(y))
	return points,x,y
x,x_x,x_y = read()
def resta_vec(v1,v2):
	v = []
	for i in range(len(v1)):
		v.append(v1[i]-v2[i])
	return v

def suma(pi):
	sum=0
	for i in range(len(pi)):
	    sum=sum+pi[i]
	return sum	
k = 5;
def generate_u():
	u = []
	a = []
	for i in range(k):
		u.append(rd.choice(x))
	u = np.array(u).astype(float)
	return u


def sigma1():
	sigma = []
	identO = np.identity(2)
	ident = [[2,0],[0,2]]
	while not np.allclose(ident,identO):
		sigma = [[rd.randint(50,100),rd.randint(1,50)],[rd.randint(1,50),rd.randint(50,100)]]
		inver = np.linalg.inv(sigma)
		ident = np.dot(inver,sigma)
	#print(sigma,ident)
	return sigma
def sigma():
	return [[1,0],[0,1]]
#print(pi)
'''
def sigma():
	sigm = []
	inva = []
	ident = []
	identO =  np.eye(2,2)
	#print(identO)
	ident = [[2,2],[2,2]]
	#print(np.asarray(ident))
	while not np.allclose(identO,ident):
		sigm =[[rd.uniform(0,500),rd.uniform(0,500)],[rd.uniform(0,500),rd.uniform(0,500)]]
		inva = np.linalg.inv(sigm)
		ident = np.dot(sigm,inva)
	return sigm
'''


u = generate_u()
sigma = [[1000,0],[0,1000]]
sigmas = [sigma for i in range(k)]
pi = np.random.dirichlet(np.ones(k),size=1)[0]


print("cc",u)
es = []
def sumatoria(lista):
	suma = 0.0
	for item in lista:
		suma += item
	return suma
def sumatoria_2(e,x):
	suma = [0,0]
	i = 0
	j = 0
	for index in range(len(e)):
		i+= e[index]*x[index][0]
		j+= e[index]*x[index][1]
	suma = np.asarray([i,j])
	return suma

def sumatoria_3(e,x,u):
	suma = [[0,0],[0,0]]
	suma_vec = [0,0]
	for index in range(len(e)):
		suma_vec = resta_vec([suma_vec[0]+x[index][0],suma_vec[1]+x[index][1]],u)
		trans_vec = np.asarray(suma_vec).T
		suma_mat = np.outer(suma_vec,trans_vec)

		suma += suma_mat*e[index]
	return suma

clusters = [[] for i in range(k)]
#hacer un while
for entrada in x:
	R = 0.0
	e = []
	new_e = 0.0
	for index in range(len(u)):
		#print(sigmas[index],"\n")
		#print("1",pi[index]*((2*mt.pi)**(-1)))
		#print("2",(np.linalg.det(sigmas[index])**(-1/2)))
		#print("3.1",0.5*(np.array(resta_vec(x[index],u[index]))).T)
		#print("3.2",np.linalg.inv(sigmas[index]))
		#print("3.3",resta_vec(x[index],u[index]))
		#print("3",np.exp(np.matmul(np.matmul(-0.5*(np.array(resta_vec(x[index],u[index]))).T,np.linalg.inv(sigmas[index])),(resta_vec(x[index],u[index])))))
		new_e = pi[index]*((2*mt.pi)**(-1))*(np.linalg.det(sigmas[index])**(-1/2))*np.exp(np.matmul((-0.5*(np.array(resta_vec(entrada,u[index]))).T),np.matmul((np.linalg.inv(sigmas[index])),(resta_vec(entrada,u[index])))))
		#print(new_e)
		#print(new_e)
		e.append(new_e)
		R += new_e
	#print("e",e)
	prob = max(e)
	#print(prob)
	clus = e.index(prob)
	clusters[clus].append(entrada)

		#e.append(new_e)
	for index in range(len(u)):
		new_e = new_e/R

	es.append(new_e)
for index in range(len(u)):
	pi[index] = sumatoria(es)/len(x)
	u[index] = sumatoria_2(es,x)[0]/sumatoria(es),sumatoria_2(es,x)[1]/sumatoria(es)
	sigmas[index] = sumatoria_3(es,x,u[index])/sumatoria(es)

		#e.append(new_e)
	#es.append(e)
#print(es)
new_u = []

#print(clusters)
for i in clusters:
	clus_pos = np.mean(i)
	new_u.append(clus_pos)

#print(u)
#plt.plot(u,'ro')
#for sigma in sigmas:
#	plt.plot(sigma,'go')

#plt.plot(x,'bo')
#plt.show()
print("u",new_u)
u_x = []
u_y = []
for cl in u:
	u_x.append(cl[0])
	u_y.append(cl[1])





plt.plot(x_x,x_y,'go')
plt.plot(u_x,u_y,'bo')
#plt.plot(x_y,'bo')
plt.show()