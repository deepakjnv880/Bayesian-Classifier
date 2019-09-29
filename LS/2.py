from math import log as ln
import numpy as np
from matplotlib import pyplot as plt
dimention=2
percent_training_data=0.75
def compute_mean_vector(data_vectors):
    mean_vector=[0,0]
    for data in data_vectors:
        mean_vector[0]+=float(data[0])
        mean_vector[1]+=float(data[1])
    mean_vector[0]/=(len(data_vectors))
    mean_vector[1]/=(len(data_vectors))
    return mean_vector

def compute_covariance_matrix(data_vectors,mean_vector):
    covariance_matrix=[[0 for x in range(dimention)] for y in range(dimention)]
    for i in range(dimention):
        for j in range(dimention):
            for k in range(len(data_vectors)):
                covariance_matrix[i][j]+=(data_vectors[k][i]-mean_vector[i])*(data_vectors[k][j]-mean_vector[j])
            covariance_matrix[i][j]/=(len(data_vectors)-1);
    return covariance_matrix

def read_training_dataset(filename):
    file_object=open(filename,'r')
    file_data=file_object.readlines()
    data_vectors=[]#list of list
    for line in file_data:
        data_vectors.append([float(x) for x in line.split()])
    return data_vectors

def calculatingW(mean_vector,covariance_matrix):
    temp=[0 for i in range(3)]
    for i in range(dimention):
        for j in range(dimention):
            temp[i]+=(mean_vector[j]*covariance_matrix[i][j])
    return temp

def calculatingW0(W,mean_vector,prior_probability):
    W0=((W[0]*mean_vector[0])+(W[1]*mean_vector[1]))
    W0/=(-2);
    W0+=ln(prior_probability)
    return W0

def calculatingG(W,W0,X):
    G=0
    for i in range(dimention):
        G+=(W[i]*X[i])
    G+=W0
    return G

def main():
     mean_vector=[]
     covariance_matrix=[[0 for x in range(dimention)] for y in range(dimention)]
     training_data=[[] for i in range(dimention)]
     testing_data_X=[[] for i in range(3)]
     testing_data_Y=[[] for i in range(3)]
     for i in range(1,4):
         filename='data/Class'+str(i)+'.txt'
         data_vectors=read_training_dataset(filename)
         # print('number of traning dataset === ',int(len(data_vectors)*percent_training_data))
         training_data=data_vectors[:int(len(data_vectors)*percent_training_data)]
         for item in data_vectors[int(len(data_vectors)*percent_training_data):]:
             testing_data_X[i-1].append(item[0])
             testing_data_Y[i-1].append(item[1])

         # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
         mean_vector.append(compute_mean_vector(training_data))
         temp=compute_covariance_matrix(training_data,mean_vector[i-1])
         for i1 in range(dimention):
             for i2 in range(dimention):
                 covariance_matrix[i1][i2]+=temp[i1][i2]
         for i1 in range(dimention):
             for i2 in range(dimention):
                 covariance_matrix[i1][i2]/=3
     # print 'mean == ',mean_vector
     # print ' afyet varaienec == ',varience
     #g(x)=WtX+W0
     W=[[0 for i in range(3)] for j in range(3)]
     W0=[0 for i in range(3)]
     # print mean_vector
     # print("===========================",W,W0)#bhai bhai siemens krunga#yaar javascript ne maari li
     # print covariance_matrix
     prior_probability=[0.3,0.3,0.3]
     for i in range(3):
         W[i]= calculatingW(mean_vector[i],np.linalg.inv(np.array(covariance_matrix)))
         W0[i]=calculatingW0(W[i],mean_vector[i],prior_probability[i])
     # deepak=open(,'r')
     print mean_vector
     # print varience
     r=1/3
     print r,' my printint ========== ',W[0],W0[0]
     X=[[] for v in range(3)]
     Y=[[] for v in range(3)]
     # print X,Y
     for i in np.arange(-15,20,0.1):
         for j in np.arange(-20,20,0.1):
             G=[0 for l in range(3)]
             for k in range(3):
                 G[k]=calculatingG(W[k],W0[k],[i,j])
             C=G.index(max(G))
             # print i,j,' belongs to class ',C
             X[C].append(i)
             Y[C].append(j)
                 # print calculatingG(W[k],W0[k],X)
     print 'drawing graph...\n'
     # print X,Y
     #naming the x axis
     plt.xlabel('x - axis')
     # naming the y axis
     plt.ylabel('y - axis')
     plt.plot(X[0],Y[0],color='#aa2889',label='class 1')
     plt.plot(X[1],Y[1],color='#76dc0a',label='class 2')
     plt.plot(X[2],Y[2],color='#2391c8',label='class 3')

     for i in range(125):
         if i==0:
             pass
             plt.plot(testing_data_X[0][i], testing_data_Y[0][i],marker='o', markersize=1, color="red",label='test class 1')
             plt.plot(testing_data_X[1][i], testing_data_Y[1][i],marker='o', markersize=1, color="green",label='test class 2')
             plt.plot(testing_data_X[2][i], testing_data_Y[2][i],marker='o', markersize=1, color="blue",label='test class 3')
         else:
             plt.plot(testing_data_X[0][i], testing_data_Y[0][i],marker='o', markersize=1, color="red")
             plt.plot(testing_data_X[1][i], testing_data_Y[1][i],marker='o', markersize=1, color="green")
             plt.plot(testing_data_X[2][i], testing_data_Y[2][i],marker='o', markersize=1, color="blue")

     plt.legend()
     plt.savefig('2.png')
     # plt.savefig('2.pdf')
     # plt.show()
if __name__ == '__main__':
    main()
    # print("frist")
