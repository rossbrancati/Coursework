import numpy as np
import matplotlib.pyplot as plt

print('-'*20)
print('Start of PCA Script')
#Load data
X = np.load('../../Data/X_train.npy')
Y = np.load('../../Data/y_train.npy')
#%% Plotting mean of the whole dataset
mean_all = np.mean(X, axis=0)
#reshape the image so it is 28x28
mean_all_reshape = mean_all.reshape(28,28)
#create an image of mean of all images in dataset
plt.imshow(mean_all_reshape)
#plot title
plt.title('Rendering of Mean of All\nDigit Images in Sample')
#save figure
plt.savefig(('../Figures/mean_of_digits.png'))
#show figure
plt.show()
#clear figure 
plt.clf()


#%% Plotting each digit
#need to calculate the average of all of the same digits (which will give us 10 total digits ranging from 0-9)
#loop over 10 digits (0-9)
for i in range(10):
    #get the average vector for the particular digit
    digit_avgs = np.average(X[Y==i],0)
    #create a subplot
    plt.subplot(2, 5, i+1)
    #reshape the vector to 28x28
    current_digit = digit_avgs.reshape(28,28)
    #add the image plot to the subplot
    plt.imshow(current_digit)
    #remove axes
    plt.axis('off')
#arrange images nicely on subplot
#plt.subplots_adjust(wspace=0, hspace=0)
#plot title
plt.suptitle('Rendering of Average of Same\nDigit Images in Sample\n')
#save figure
plt.savefig(('../Figures/indivdual_digits.png'))
#show figure
plt.show()
#clear figure 
plt.clf()

#%% Center the data (subtract the mean)
a, b = X.shape
#calculate mean of each rcolumn (each column represents a variable)
mean_X = np.mean(X, axis=0)
#subtract the mean from every entry in the matrix
mean_cent_X = np.zeros((a, b))
for i in range(a):
    mean_cent_X[i,:] = X[i,:] - mean_X


#%% Calculate Covariate Matrix
cov_mat = np.cov(mean_cent_X, rowvar=False, bias=True)

#%% Calculate eigen values and vectors
#np.linal.eigh sorts the eigenvectors and eigenvalues in reverse order
eigenvals, eigenvects = np.linalg.eigh(cov_mat)
#sort the eigenvalues in descending order
sorted_eigenvals = eigenvals[::-1]
#sorting the eigenvectors too
sorted_eigenvects = np.flip(eigenvects, axis=1)

#%% Plot eigen values
plt.plot(sorted_eigenvals)
plt.ylabel('Eigenvalue (value)')
plt.xlabel('Eigenvalue Number (count)')
plt.title('Eigenvalue Numbers and Respective Values')
plt.savefig('../Figures/eigenvalues.png')

#%% Plot 5 first eigen vectors
#loop over 5 
for i in range(5):
    #create a subplot
    plt.subplot(1, 5, i+1)
    #reshape the current eigenvector
    current_eigenvector = sorted_eigenvects[:,i].reshape(28,28)
    #plot the reshaped eigenvector
    plt.imshow(current_eigenvector)
    #turn off axes
    plt.axis('off')
    #add title
    plt.title('PC'+str(i+1))
#plot title
#plt.suptitle('Plots of Top 5 Eigenvectors')
#save figure
plt.savefig(('../Figures/top_five_eigenvectors.png'))
#show figure
plt.show()
#clear figure 
plt.clf()

#%% Project to two first bases
projection = np.matmul(X, sorted_eigenvects)

#%% Plotting the projected data as scatter plot
for i in range(10):
    #get the average vector for the particular digit
    plt.scatter(projection[Y==i,0], projection[Y==i,1], label=i)

#axis labels
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
#title
plt.title('Original Data Projected onto PC1 and PC2')
#add a legend
plt.legend(bbox_to_anchor=(1, 0.9))
#add tight layout to pad edges of plot
plt.tight_layout()
#save figure
plt.savefig(('../Figures/projected_data.png'))
#show figure
plt.show()
#clear figure 
plt.clf()

print('End of PCA Script')
print('-'*20)
