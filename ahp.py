import numpy as np
from scipy.spatial.distance import cdist

# AHP Step: Calculate Normalized Weights
def ahp(pairwise_matrix):
    eigvals, eigvecs = np.linalg.eig(pairwise_matrix)
    max_eigval = np.max(eigvals)
    weights = eigvecs[:, np.argmax(eigvals)].real
    weights = weights / np.sum(weights)
    consistency_index = (max_eigval - len(pairwise_matrix)) / (len(pairwise_matrix) - 1)
    random_index = 1.12  # For 3 criteria
    consistency_ratio = consistency_index / random_index
    return weights, consistency_ratio

# TOPSIS Step: Rank Alternatives
def topsis(decision_matrix, weights):
    normalized_matrix = decision_matrix / np.sqrt((decision_matrix**2).sum(axis=0))
    weighted_matrix = normalized_matrix * weights
    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)
    distance_to_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    distance_to_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
    performance_score = distance_to_worst / (distance_to_best + distance_to_worst)
    return performance_score

# Input Example
pairwise_matrix = np.array([
    [1, 3, 5],  # Saaty scale inputs
    [1/3, 1, 2],
    [1/5, 1/2, 1]
])

decision_matrix = np.array([
    [3, 7, 8],  # Criteria for Location A
    [5, 6, 7],  # Criteria for Location B
    [4, 8, 9]   # Criteria for Location C
])

# AHP Calculations
weights, consistency_ratio = ahp(pairwise_matrix)
if consistency_ratio < 0.1:
    print("Weights:", weights)
    print("Consistency Ratio:", consistency_ratio)
else:
    print("Consistency Ratio too high. Recheck pairwise comparisons.")

# TOPSIS Calculations
scores = topsis(decision_matrix, weights)
rankings = np.argsort(scores)[::-1] + 1  # Higher score = better rank
print("Scores:", scores)
print("Rankings:", rankings)
