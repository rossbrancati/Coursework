import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def SVD(A, s, k):
    # TODO: Calculate probabilities p_i
    n,m = A.shape
    #create a matrix to store probabilities in
    p_i_all = np.zeros(n)
    for i in range(n):
        #each pi should be a scalar, so p_i_all should be vector of length n (number of rows)
        p_i = (np.linalg.norm(A[i,:])*np.linalg.norm(A[i,:]))/(np.linalg.norm(A, ord='fro')*np.linalg.norm(A, ord='fro'))
        p_i_all[i] = p_i
        
    # TODO: Construct S matrix of size s by m
    #create a matrix of zeros to store values
    S = np.zeros((s,m))
    #pick a random integer j with probability Pr(pick j)=p_j
    for i in range(s):
        #use random choice function to select integer and store in S
        j = np.random.choice(n, replace=False, p=p_i_all)
        S[i] = A[j,:]
    
    # TODO: Calculate SS^T
    sst = np.dot(S,np.matrix.transpose(S))

    # TODO: Compute SVD for SS^T
    w, s, vh = np.linalg.svd(sst)

    # TODO: Construct H matrix of size m by k
    # create matrix H to store data
    H = np.zeros((m,k))
    #loop over k rows and calculate h_t for each row
    for i in range(k):
        h_t = (np.dot(np.matrix.transpose(S),vh[i,:]))/(np.linalg.norm((np.dot(np.matrix.transpose(S),vh[i,:]))))
        #store h_t in matrix H
        H[:,i] = h_t

    # Return matrix H and top-k singular values sigma
    return H, s[0:k]

def main():
    print('Start of SVD Script')
    im = Image.open("../../Data/baboon.tiff")
    A = np.array(im)
    H, sigma = SVD(A, 80, 60)
    k = 60
    
    # TO DO: Compute SVD for A and calculate optimal k-rank approximation for A.
    #compute the SVD for A
    U, S, V = np.linalg.svd(A)
    #calculate A_k using equation A_k=USV^T 
    A_k = np.matmul(U[:,0:k],np.matmul(np.diag(S[0:k]),V[0:k,:]))

    # TO DO: Use H to compute sub-optimal k rank approximation for A
    #equation: A_hat = AHH^t
    A_k_hat = np.matmul((np.matmul(A,H)),np.transpose(H))

    # To DO: Generate plots for original image, optimal k-rank and sub-optimal k rank approximation
    #plot original matrix
    plt.imshow(A, cmap='gray', vmin=0, vmax=255)
    #plot title
    plt.title("Original Baboon Image")
    #save, show, and clear figure
    plt.savefig(('../Figures/baboon_original.png'))
    plt.show()
    plt.clf()
    
    #plot optimal k-rank approximation
    plt.imshow(A_k, cmap='gray', vmin=0, vmax=255)
    #title
    plt.title("Optimal K-Rank Approximation")
    #save, show, and clear figure
    plt.savefig(('../Figures/optimal_approx.png'))
    plt.show()
    plt.clf()
    
    #plotsub-optimal k rank approximation
    plt.imshow(A_k_hat, cmap='gray', vmin=0, vmax=255)
    #title
    plt.title("Sub-optimal K-Rank Approximation")
    #save, show, and clear figure
    plt.savefig(('../Figures/sub_optimal_approx.png'))
    plt.show()
    plt.clf()
    
    # TO DO: Calculate the error in terms of the Frobenius norm for both the optimal-k
    # rank produced from the SVD and for the k-rank approximation produced using
    # sub-optimal k-rank approximation for A using H.
    
    #calcualate error of optimal k-rank approximation, A_k, in terms of Frobenius norm
    A_k_fro = np.linalg.norm(A-A_k, ord='fro')
    #calcualate error of sub-optimal k-rank approximation, A_k_hat, in terms of Frobenius norm
    A_k_hat_fro = np.linalg.norm(A-A_k_hat, ord='fro')
    
    #subtract and print errors
    print('-'*20)
    print('Optimal K-Rank Approximation Error:')
    print(A_k_fro)
    print('')
    print('Sub-Optimal K-Rank Approximation Error:')
    print(A_k_hat_fro)
    print('End of SVD Script')
    print('-'*20)
    
    

if __name__ == "__main__":
    main()
