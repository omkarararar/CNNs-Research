import numpy as np

def distance(point1, point2):
    #dist b/w two points
    diff = point1 - point2
    return np.sqrt(np.sum(diff ** 2))

def kmeans(X, k, max_iter=100, tol=0.0001, seed=0):
    """
    Arguments:
        X: data points 
        k: number of clusters 
        max_iter: maximum number of iterations before stopping
        tol: tolerance - if centers move less than this, algorithm has converged
        seed: random seed for reproducibility (randomised center initialization)
    
    Returns:
        centers: final cluster centers (k x n_features array)
        labels: cluster assignment for each point (n_samples array)
    """
    X = np.array(X, dtype=float)
    n = len(X)
    
    # choose k random points as initial centers
    np.random.seed(seed)  
    random_indices = np.random.choice(n, k, replace=False)  
    centers = X[random_indices].copy()
    print(f"Starting with {k} random centers")
    
    # continue this main loop till convergence or maximum iterations occur
    for iteration in range(max_iter):
        print(f"\nCurrent Iteration {iteration + 1}")
        
        # asign each point to the nearest center
        labels = []  
        for i in range(n):
            point = X[i]  #point to be assigned rn
            
            # find the closest center
            min_distance = float('inf')  # take max distance for reference
            nearest_center = 0  # set the default nearest center
            
            # check the distance to each center
            for center_id in range(k):
                dist = distance(point, centers[center_id])
                
                # if the center is closer, update nearest center
                if dist < min_distance:
                    min_distance = dist
                    nearest_center = center_id
            
            # now, assign the point to the nearest center found
            labels.append(nearest_center)
        
        labels = np.array(labels)
        #  how many points assigned to each cluster
        print(f"Assigned points to clusters: {np.bincount(labels)}")
        
        #  update centers to be the mean of their assigned points
        new_centers = []  
        max_movement = 0  # max change in center positions

        
        for cluster_id in range(k):
            # check all points assigned to this cluster
            cluster_points = []
            for i in range(n):
                if labels[i] == cluster_id:  
                    cluster_points.append(X[i])
            
            if len(cluster_points) > 0:
                # calculate average position of all points in this cluster
                cluster_points = np.array(cluster_points)
                new_center = np.mean(cluster_points, axis=0)
            else:
                # if no points assigned to this cluster, keep the old center
                new_center = centers[cluster_id]
            
            # check the change is distance of this center
            movement = distance(centers[cluster_id], new_center)
            # check and keep the maximum movement
            max_movement = max(max_movement, movement)
            
            new_centers.append(new_center)
        
        new_centers = np.array(new_centers)
        print(f"max change in center position: {max_movement:.6f}")
        

        # if the centers havent changed i.e. less than tolerance, stop
        if max_movement < tol:
            print(f"\ncenters converged, less than tolerance {tol}")
            centers = new_centers
            break

        centers = new_centers

    return centers, labels


if __name__ == '__main__':
    # sample dataset with 3 clusters
    np.random.seed(0)  
    
    # Group 1: 50 points around (0, 0)
    group1 = []
    for i in range(50):
        #sd 0.9 
        x = np.random.normal(0, 0.9)
        y = np.random.normal(0, 0.9)
        group1.append([x, y])
    
    # Group 2: 50 points around (4, 0)
    group2 = []
    for i in range(50):
        # Same as group1 but centered at x=4
        x = np.random.normal(4, 0.9)
        y = np.random.normal(0, 0.9)
        group2.append([x, y])
    
    # Group 3: 50 points around (8, 0)
    group3 = []
    for i in range(50):
        # Same as group1 but centered at x=8
        x = np.random.normal(8, 0.9)
        y = np.random.normal(0, 0.9)
        group3.append([x, y])
    
    # Combine all three groups into one dataset
    # This creates a list of 150 points total
    X = np.array(group1 + group2 + group3)
    
    
    # Run k-means clustering algorithm
    centers, labels = kmeans(X, k=3)
    
    print('Cluster centers:')
    # centers and values
    # show final cluster centers
    print('\nCluster centers:')
    print(centers)

    # show how many points in each cluster
    print(f'\nPoints per cluster: {np.bincount(labels)}')