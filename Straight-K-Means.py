import pandas as pd
import numpy as np
import random

def k_means_iteration(data_points, centroids):
    K = len(centroids)
    cluster_assignments = {}
    S_k = [[] for i in range(K)]
    
    for i, point in enumerate(data_points):
        distances = []
        
        for k in range(K):
            dist = np.sum((point - centroids[k])**2)
            distances.append((dist, k))
            
        min_dist = float('inf')
        nearest_k = None
        for dist, k in distances:
            if dist < min_dist:
                min_dist = dist
                nearest_k = k
                
        cluster_assignments[i] = nearest_k
        S_k[nearest_k].append(point)
        
    return cluster_assignments, S_k

def cluster_recenter(S_k, dimension):
    K = len(S_k)
    new_centroids = []

    for k in range(K):
        if len(S_k[k]) > 0: #Funny check
            centroid = []
            for v in range(dimension):

                feature_values = []
                for point in S_k[k]:
                    feature_values.append(point[v])

                feature_mean = sum(feature_values) / len(feature_values)
                centroid.append(feature_mean)

            new_centroids.append(centroid)
        else:
            new_centroids.append([0] * dimension)

    return new_centroids

def square_error(data_points, cluster_assignments, centroids):
    K = len(centroids)
    total_error = 0
    cluster_errors = []
    
    for k in range(K):
        cluster_error = 0
        
        cluster_points = []
        for i, assigned_cluster in cluster_assignments.items():
            if assigned_cluster == k:
                cluster_points.append(data_points[i])
        
        for point in cluster_points:
            squared_dist = np.sum((point - centroids[k])**2)
            cluster_error += squared_dist
            
        cluster_errors.append(cluster_error)
        total_error += cluster_error
        
    return total_error, cluster_errors

df = pd.read_excel('ProcessedData.xlsx')
Y = df.to_numpy()
data_points = Y[:, 1:]
towns = Y[:, 0]
K = int(input("Number of initial clusters: "))

best_error = float('inf')
best_initial_indices = None

num_initializations = 10

for init in range(num_initializations):
    random_indices = random.sample(range(len(data_points)), K)
    centroids = [data_points[i] for i in random_indices]
    
    last_iteration = None
    iteration = 0
    while iteration < 1000:
        cluster_assignments, S_k = k_means_iteration(data_points, centroids)

        if last_iteration and cluster_assignments == last_iteration:
            break

        centroids = cluster_recenter(S_k, len(data_points[0]))
        last_iteration = cluster_assignments
        iteration += 1
    
    total_error, cluster_errors = square_error(data_points, cluster_assignments, centroids)
    if total_error < best_error:
        best_error = total_error
        best_initial_indices = random_indices
        cluster_lists = [[] for i in range(K)]
        for i, cluster in cluster_assignments.items():
            cluster_lists[cluster].append(towns[i].strip())

print("Best error: ", best_error)
print("Best initial centroid indices:",  best_initial_indices)
print("Best cluster list", cluster_lists)
print(cluster_assignments)
    
