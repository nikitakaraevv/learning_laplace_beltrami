import numpy as np
import scipy.spatial as spatial
import scipy.sparse as sparse
from scipy.sparse.linalg import eigs

########################################################################################################################################################################################
# function that implements Belkins Pointcloud Laplacian Operator as per the paper: http://web.cse.ohio-state.edu/~wang.1016/papers/pcdlaplace_soda.pdf  
# Inputs: 1. Matrix V -- N x 3 matrix of point cloud coordinates
#         2. r_pc (parameter, set to 0.05 or greater)
#         3. numEigs -- number of smallest eigenvalues and eigenvectors desired      

# Outputs: Dictionary with the following items
#          1. L -- N xN Laplacian matrix
#          2. evals -- list of eigenvalues
#          3. evecs  -- N x numEigs vector of eigenvector evecs[:,i] corresponds to i'th eigenvalue

#Comments: L is not necessarily symmetric and hence imaginary eigenvalues are possible 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
def Belkin_Pointcloud_Laplacian(V,numEigs,r_pc=0.05):

    D = spatial.distance.cdist(V,V,'euclidean') # Pairwise Euclidean Distance Matrix
     
    Ns = V.shape[0]
    
    t=np.mean(D); # t parameter for PCD-LAPLACE
    
    #r = 10*np.power(t,(2+zeta))zeta = 1;
    
    r = r_pc*np.max(D);delta = 1.15*r;
    
    # Main loop iterating over all the points
    
    I = []; J = []; S = [];
    
    for pt_iter in range(0,Ns):
              
        #print(pt_iter)
    # -------- Estimate the Local Tangent Plane for each point pt_iter -------------------------------------------------------------------------------------------
    
        distfunct_pt = D[pt_iter,:]; #extracts the euclidean distance of point pt_iter from matrix D
        ind_r = np.where(distfunct_pt<r)[0] #indices of elements within a radius of r
        
        
        V_Tp = V[ind_r,:]; # extract the vertices for local 3-D coordinates for tangent plance estimation
        V_Tp -= V[pt_iter,:] # subtract the center point(pt_iter) from local 3D coordinates
        
        Cv = np.matmul(V_Tp.transpose(),V_Tp) # compute the local covariance matrix
        _, Tp = np.linalg.eigh(Cv) # eigenvalue decomposition of local covariance matrix 
        
        Tp = Tp[:,-2:] # extract largest 2 columns a projection matrix of the tangent space, columns in ascending order
        
    # ------------------- Delaunay Triangulation   --------------------------------------------------------------------------------------------------------------    
        
        ind_delta = np.where(distfunct_pt<delta)[0] #indices of elements within a radius of delta
        ind_delta_by_2 = np.where(distfunct_pt<delta/2)[0] #indices of elements within a radius of delta/2
            
        V_DT_3D = V[ind_delta,:]; # extract the vertices for local 3-D coordinates for delaunay triangulation
        V_DT_3D -= V[pt_iter,:] # subtract the center point from local 3D coordinates
    
        V_DT_2D = np.matmul(Tp.transpose(),V_DT_3D.transpose()).transpose(); # project the points onto the tangent space
    
        TRI = spatial.Delaunay(V_DT_2D) # Delaunay triangulation of the projected co-ordinates
    
        ind_delta_by_2 = np.delete(ind_delta_by_2, np.where(ind_delta_by_2==pt_iter)) # delete the trivial pt_iter index (with 0 distance)
    
    # ------------------- Find the indices and compute areas of delaunay triangles -------------------------------------------------------------------------------    
    
        ind_intersect =  np.arange(0,V_DT_2D.shape[0])[np.in1d(ind_delta,ind_delta_by_2)]; # local indices (in terms of of V_DT_2D/3D) that are less than delta/2
        
        AR = np.zeros(len(ind_intersect)) # array for area of delaunay triangles
        GS = np.zeros(len(ind_intersect)) # array for exponential gaussian squared distance weights
        
        # for each index in the set of point withing the distance delta/2 do the following:
            
        for a_iter in range(0,len(ind_intersect)):
            
            ind_area = np.array(np.where(TRI.simplices == ind_intersect[a_iter]))[0] # pick the indices of TRI that have the element ind_intersect[a_iter]
            
            sum_ar = 0.0;
        
        # ------------------------------------------------------------------------------------
            # compute of areas of triangles incident on the chosen point
            for ar_iter in range(0,len(ind_area)):
                
                triangle = TRI.simplices[ind_area[ar_iter],:]
                
                v1 = V_DT_2D[triangle[0],:]
                v2 = V_DT_2D[triangle[1],:]
                v3 = V_DT_2D[triangle[2],:]
                
                S1 = v2-v1; S2 = v3-v1; 
                
                sum_ar += 0.5*np.abs(np.cross(S1,S2)); # area of triangle
        # ------------------------------------------------------------------------------------   
            # Using convex hull and then standard areas 
#            for ar_iter in range(0,len(ind_area)):
#                
#                triangle = TRI.simplices[ind_area[ar_iter],:]
#                sum_ar += 0.5*spatial.ConvexHull(V_DT_2D[triangle,:]).area; # area of triangle
#                
        # ------------------------------------------------------------------------------------
                
            AR[a_iter] = sum_ar;
            GS[a_iter] = -(1/(12*np.pi*t*t))*np.exp(-np.sum(np.square(V_DT_3D[ind_intersect[a_iter],:]))/(4*t)) # Gaussian squred distance weights
            
    # -------------------Stack into vectors of indices to input sparse matrix format ----------------------------------------------------------------------------    
        
    # Stack in 3 columns: row index, column index and value     
        J = J + (ind_delta[ind_intersect].tolist())
        I = I + ((pt_iter*np.ones(len(ind_intersect))).tolist())
        S = S + (np.multiply(AR,GS).tolist())
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------  
    
    W = sparse.csr_matrix((S, (I, J)), [Ns, Ns]) 
    dg = np.array(-W.sum(axis=1)).squeeze()
    
    D = sparse.spdiags(dg, [0], Ns, Ns)
        
    L = D+W    
    
    evals,evecs = eigs(L, k=numEigs, M=None, sigma=None, which='SM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, OPpart=None)
        
    Laplacian = {'L':L, 'evals':np.real(evals), 'evecs':np.real(evecs)}

    return Laplacian  
########################################################################################################################################################################################   



########################################################################################################################################################################################
# function that implements Liu's Pointcloud Laplacian Operator as per the paper: https://personal.utdallas.edu/~xxg061000/pbmh.pdf  
# Inputs: 1. Matrix V -- N x 3 matrix of point cloud coordinates
#         2. r_pc
#         3. numEigs -- number of smallest eigenvalues and eigenvectors desired      

# Outputs: Dictionary with the following items
#          1. Q -- N x N symmetric weights matrix 
#          2. B -- N x N diagonal area matrix 
#          2. evals -- list of eigenvalues
#          3. evecs  -- N x numEigs vector of eigenvector evecs[:,i] corresponds to i'th eigenvalue
# Comments: The Discrete Laplacian is given by B^-1 Q. The Laplacian gives orthonormal Basis and real eigenvalues. 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
def Liu_Pointcloud_Laplacian(V,numEigs,r_pc=0.05):

    D = spatial.distance.cdist(V,V,'euclidean') # Pairwise Euclidean Distance Matrix
    
    Ns = V.shape[0]
    
    t=np.mean(D); # t parameter for PCD-LAPLACE
    
    #r = 10*np.power(t,(2+zeta))zeta = 1;
    
    r = r_pc*np.max(D);delta = 1.15*r;
    
    # Main loop iterating over all the points
    
    I = []; J = []; S = [];
    
    b = [] # diagonal array of center point voronoi areas
    
    for pt_iter in range(0,Ns):
              
        #print(pt_iter)
    # -------- Estimate the Local Tangent Plane for each point pt_iter -------------------------------------------------------------------------------------------
    
        distfunct_pt = D[pt_iter,:]; #extracts the euclidean distance of point pt_iter from matrix D
        ind_r = np.where(distfunct_pt<r)[0] #indices of elements within a radius of r
        
        
        V_Tp = V[ind_r,:]; # extract the vertices for local 3-D coordinates for tangent plance estimation
        V_Tp -= V[pt_iter,:] # subtract the center point(pt_iter) from local 3D coordinates
        
        Cv = np.matmul(V_Tp.transpose(),V_Tp) # compute the local covariance matrix
        _, Tp = np.linalg.eigh(Cv) # eigenvalue decomposition of local covariance matrix 
        
        Tp = Tp[:,-2:] # extract largest 2 columns a projection matrix of the tangent space, columns in ascending order
        
    # ------------------- Delaunay Triangulation   --------------------------------------------------------------------------------------------------------------    
        
        ind_delta = np.where(distfunct_pt<delta)[0] #indices of elements within a radius of delta
        #ind_delta_by_2 = np.where(distfunct_pt<delta/2)[0] #indices of elements within a radius of delta/2
        
    
        V_DT_3D = V[ind_delta,:]; # extract the vertices for local 3-D coordinates for delaunay triangulation
        V_DT_3D -= V[pt_iter,:] # subtract the center point from local 3D coordinates
    
        V_DT_2D = np.matmul(Tp.transpose(),V_DT_3D.transpose()).transpose(); # project the points onto the tangent space
        
        VOR = spatial.Voronoi(V_DT_2D,qhull_options='Qbb Qc Qx')
        
    # ------------------- Find the indices and compute areas of delaunay triangles -------------------------------------------------------------------------------    
    
        ind_intersect =  np.arange(0,V_DT_2D.shape[0])#[np.in1d(ind_delta,ind_delta_by_2)]; # local indices (in terms of of V_DT_2D/3D) that are less than delta/2
        
        # Comment: Theres a 1-1 mapping between ind_intersect and ind_delta_by_2
        
        AR = np.zeros(len(ind_intersect)) # array for area of delaunay triangles
        GS = np.zeros(len(ind_intersect)) # array for exponential gaussian squared distance weights
        
       
        # for each index in the set of point withing the distance delta/2 do the following:    
        for a_iter in range(0,len(ind_intersect)):
                    
            vor_region_vertices = VOR.regions[np.where(VOR.point_region == ind_intersect[a_iter])[0][0]]
            
            if (-1 in vor_region_vertices):
                AR[a_iter] = 6000; # bad points
                GS[a_iter] = 6000;
            else:
                AR[a_iter] = spatial.ConvexHull(VOR.vertices[vor_region_vertices,:]).area # compute the area of the convex hull spanned by each cell
                GS[a_iter] = -(1/(4*np.pi*t*t))*np.exp(-np.sum(np.square(V_DT_3D[ind_intersect[a_iter],:]))/(4*t)) # Gaussian squred distance weights
                
    
        pt_iter_index = np.where(ind_delta==pt_iter)[0][0] # index of center point
       
        b = b + [AR[pt_iter_index]]
        
        #if AR[pt_iter_index]>5000:
        #    print('yes')
    
        AR = np.delete(AR,pt_iter_index); # delete the trivial pt_iter index (with 0 distance)
        GS = np.delete(GS,pt_iter_index); 
        ind_intersect = np.delete(ind_intersect,pt_iter_index); 
        
        badpoints = np.where(AR == 6000)
        AR = np.delete(AR,badpoints); # delete the bad points
        GS = np.delete(GS,badpoints); 
        ind_intersect = np.delete(ind_intersect,badpoints); 
        
        AR = AR*b[pt_iter];
        
            
    # -------------------Stack into vectors of indices to input sparse matrix format ----------------------------------------------------------------------------    
        
    # Stack in 3 columns: row index, column index and value     
        J = J + (ind_delta[ind_intersect].tolist())
        I = I + ((pt_iter*np.ones(len(ind_intersect))).tolist())
        S = S + (np.multiply(AR,GS).tolist())
        
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------  
    
    W = sparse.csr_matrix((S, (I, J)), [Ns, Ns]) 
    dg = np.array(-W.sum(axis=1)).squeeze()
    
    D = sparse.spdiags(dg, [0], Ns, Ns)
    Q = D+W   
    
    B = sparse.spdiags(b, [0], Ns, Ns)
        
    evals,evecs = sparse.linalg.eigs(Q, k=10, M=B, sigma=-1e-4, which='SM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, OPpart=None)
        
    Laplacian = {'Q':Q,'B':B, 'evals':(evals), 'evecs':(evecs)}

    return Laplacian  
########################################################################################################################################################################################   


########################################################################################################################################################################################
# function that implements a Vanilla Pointcloud Laplacian Operator as per the paper: https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf  
# Inputs: 1. Matrix V -- N x 3 matrix of point cloud coordinates
#         2. k_nn: number of nearest neighbours desired (set to 50)
#         3. numEigs -- number of smallest eigenvalues and eigenvectors desired      

# Outputs: Dictionary with the following items
#          1. L -- N x N Laplacian matrix
#          2. D -- N x N diagonal point weight matrix 
#          3. evals -- list of eigenvalues
#          4. evecs  -- N x numEigs vector of eigenvector evecs[:,i] corresponds to i'th eigenvalue

# Comments:evals and evecs are computed for the generalized eigenvalue problem: Lf = lambda Df
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
def Vanilla_Pointcloud_Laplacian(V,numEigs,k_nn=50):

    D = spatial.distance.cdist(V,V,'euclidean') # Pairwise Euclidean Distance Matrix
    
    Ns = V.shape[0]

    t=np.mean(D);
    
    # Main loop iterating over all the points
    
    I = []; J = []; S = [];
    
    for pt_iter in range(0,Ns):
              
        #print(pt_iter)
    # --------  Nearest Neighbours -------------------------------------------------------------------------------------------
    
        distfunct_pt = D[pt_iter,:]; #extracts the euclidean distance of point pt_iter from matrix D
        ind_knn = distfunct_pt.argsort()[:k_nn+1]; # extract the k_nn nearest neighbours 
        #print(ind_knn[0])
        ind_knn = np.delete(ind_knn,[0]); # delete the trivial pt_iter element
        
        dist_knn = distfunct_pt[ind_knn] # k_nn distances
        
        weights = -np.exp(-np.square(dist_knn)/(4*t*t)) # compute weights for nearest neighbours
        
            
    # -------------------Stack into vectors of indices to input sparse matrix format ----------------------------------------------------------------------------    
        
    # Stack in 3 columns: row index, column index and value     
        J = J + (ind_knn.tolist())
        I = I + ((pt_iter*np.ones(len(ind_knn))).tolist())
        S = S + (weights.tolist())
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------  
    
    W = sparse.csr_matrix((S, (I, J)), [Ns, Ns]) 
    
    W = (W+W.transpose())/2; # symmetrization step
    
    dg = np.array(-W.sum(axis=1)).squeeze()
    
    D = sparse.spdiags(dg, [0], Ns, Ns)
        
    L = D+W    
    
    evals,evecs = sparse.linalg.eigs(L, k=numEigs, M=D, sigma=None, which='SM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, OPpart=None)
        
    Laplacian = {'L':L, 'D':D, 'evals':evals, 'evecs':evecs}

    return Laplacian  

