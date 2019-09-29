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

def calculatingW(inverse_covariance_matrix):
    return np.array(inverse_covariance_matrix)*2

def calculatingw(mean_vector,inverse_covariance_matrix):
    return np.matmul(np.array(inverse_covariance_matrix),np.transpose(np.array(mean_vector)))

def calculatingW0(w,mean_vector,prior_probability,covariance_matrix):
    W0=(np.matmul(np.array(mean_vector),np.transpose(np.array(w)))/(-2))
    W0-=float(ln(np.linalg.det(np.array(covariance_matrix)))/2)
    W0+=ln(prior_probability)
    return W0

def calculatingG(W,w,W0,X):
    G=0
    G+=np.matmul(np.array(X),np.matmul(np.array(W),np.transpose(np.array(X))))
    G+=np.matmul(np.array(w),np.transpose(np.array(X)));
    return G

def main():
     mean_vector=[]
     covariance_matrix=[]
     training_data=[[] for i in range(dimention)]
     testing_data_X=[[] for i in range(3)]
     testing_data_Y=[[] for i in range(3)]
     print covariance_matrix
     for i in range(1,4):
         filename='data/Class'+str(i)+'.txt'
         data_vectors=read_training_dataset(filename)
         # print('number of traning dataset === ',int(len(data_vectors)*percent_training_data))
         training_data=data_vectors[:int(len(data_vectors)*percent_training_data)]
         for item in data_vectors[int(len(data_vectors)*percent_training_data):]:
             testing_data_X[i-1].append(item[0])
             testing_data_Y[i-1].append(item[1])

         mean_vector.append(compute_mean_vector(training_data))
         temp=compute_covariance_matrix(training_data,mean_vector[i-1])
         # print temp
         temp[0][1]=0;temp[1][0]=0;
         covariance_matrix.append(temp)

     W=[]
     w=[]
     W0=[]
     # print mean_vector
     print("===========================")#bhai bhai siemens krunga#yaar javascript ne maari li
     print covariance_matrix
     print('****************************')
     prior_probability=[0.3,0.3,0.3]
     for i in range(3):
         W.append(calculatingW(np.linalg.inv(np.array(covariance_matrix[i]))))
         w.append(calculatingw(mean_vector[i],np.linalg.inv(np.array(covariance_matrix[i]))))
         W0.append(calculatingW0(w[i],mean_vector[i],prior_probability[i],covariance_matrix[i]))

     X=[[] for v in range(3)]
     Y=[[] for v in range(3)]
     # # print X,Y
     for i in np.arange(-15,20,0.1):
         for j in np.arange(-20,20,0.1):
             G=[0 for l in range(3)]
             for k in range(3):
                 G[k]=calculatingG(W[k],w[k],W0[k],[i,j])
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
     plt.savefig('3.png')
     # plt.savefig('3.pdf')
     # plt.show()
if __name__ == '__main__':
    main()
    # print("frist")
