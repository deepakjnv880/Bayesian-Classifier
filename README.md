# Bayesian-Classifier
Implements Bayes classifier using Unimodel Gaussian Distribution as the density function


There are three folder LS(Linearly Seperable) , NLS(Non Linearly Seperable) , RD(Real World Data). Each folder has 4 python files for 4 below listed cases and corresponding classification image.

	(1) Covariance matrix for all the classes is the same and is σ 2 I
	You can obtain the same Covariance matrix for all the classes by
	taking the average of Covariance matrices of all the classes. You can
	obtain the same variance by averaging all the variences.
	(2) Full Covariance matrix for all the classes is the same and is Σ.
	You can obtain the same Covariance matrix for all the classes by
	taking the average of Covariance matrices of all the classes.
	(3) Covariance matric is diagonal and is different for each class
	(4) Full Covariance matrix for each class is different

# Assumption 
Data follow unimodal gaussian distribution

# Package required
Install matplotlib and tkinter using below command:  

	pip install matplotlib
	sudo apt-get install python3-tk

# How to run
Go to the folder and run command python x.py where x can be from 1 to 4 for above four different cases of covarience matrix.
