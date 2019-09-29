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

def calculatingW(mean_vector,sq_sigma):
    temp=[]
    for i in range(dimention):
        temp.append(mean_vector[i]/sq_sigma)
    return temp

def calculatingW0(mean_vector,sq_sigma,prior_probability):
    W0=((mean_vector[0]*mean_vector[0])+(mean_vector[1]*mean_vector[1]))
    W0/=(-2*sq_sigma);
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
     # xmin=ymin=100;xmax=ymax=-100
     upper_limit=-100
     lower_limit=100
     for i in range(1,4):
         filename='data/Class'+str(i)+'.txt'
         data_vectors=read_training_dataset(filename)
         print 'mini ==============================='
         # print
         lower_limit=min(lower_limit, np.amin(data_vectors));
         upper_limit=max(upper_limit, np.amax(data_vectors));
         # xmax=max(xmax, np.amax(data_vectors,axis=0)[0]);
         # ymax=max(ymax, np.amax(data_vectors,axis=1)[1]);

         # print('number of traning dataset === ',int(len(data_vectors)*percent_training_data))
         training_data=data_vectors[:int(len(data_vectors)*percent_training_data)]
         for item in data_vectors[int(len(data_vectors)*percent_training_data):]:
             # print item
             testing_data_X[i-1].append(item[0])
             testing_data_Y[i-1].append(item[1])

         mean_vector.append(compute_mean_vector(training_data))
         temp=compute_covariance_matrix(training_data,mean_vector[i-1])
         for i1 in range(dimention):
             for i2 in range(dimention):
                 covariance_matrix[i1][i2]+=temp[i1][i2]
     varience=covariance_matrix[0][0]+covariance_matrix[1][1]
     # print ' before varaienec == ',varience
     varience=float(varience/6)
     covariance_matrix[0][1]=0;covariance_matrix[1][0]=0
     covariance_matrix[0][0]=varience;covariance_matrix[1][1]=varience

     W=[[0 for i in range(3)] for j in range(3)]
     W0=[0 for i in range(3)]

     prior_probability=[0.3,0.3,0.3]
     for i in range(3):
         W[i]= calculatingW(mean_vector[i],varience)
         W0[i]=calculatingW0(mean_vector[i],varience,prior_probability[i])

     print 'dicision boundry' ,W,W0
     # deepak=open(,'r')
     print mean_vector
     print varience
     r=1/3
     print r,' my printint ========== ',W,W0
     X=[[] for v in range(3)]
     Y=[[] for v in range(3)]
     # print X,Y
     # print xmin,' == ',xmax
     # print ymin,' == ',ymax
     for i in np.arange(lower_limit,upper_limit,0.01):
         for j in np.arange(lower_limit,upper_limit,0.01):
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
     plt.savefig('1.png')
     # plt.savefig('1.pdf')
     plt.show()
if __name__ == '__main__':
    main()
    # print("frist")
