# Adapted from https://github.com/georgiosarvanitidis/geometric_ml

from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy.integrate as integrate
import sys
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.interpolate import CubicSpline

class SolverGraph:
    """
    This class initializes with a metric, a dataset, and parameters for nearest neighbors and integration tolerance.
    It uses a metric representing the riemannian metric of the manifold, and some data representing points on the manifold
    It constructs a k-nearest neighbors (kNN) graph using the data. This graph is used to approximate the manifold's structure.
    Weight Matrix Construction: For each data point, the method computes the Riemannian distance to its k-nearest neighbors,
    creating a weight matrix representing these distances
    """
    def __init__(self, metric, data, kNN_num, tol=1e-5, limit=50):
        self.metric = metric
        self.data = data
        self.kNN_num = kNN_num + 1  # The first point for the training data is always the training data point.
        self.kNN_graph = NearestNeighbors(n_neighbors=kNN_num + 1, algorithm='ball_tree').fit(data)  # Find the nearest neighbors
        self.tol = tol
        self.limit = limit
        N_data = data.shape[0]

        # Find the Euclidean kNN
        distances, indices = self.kNN_graph.kneighbors(data)
        Weight_matrix = np.zeros((N_data, N_data))  # The indicies of the kNN for each data point
        for ni in range(N_data):  # For all the data
            p_i = data[ni, :].reshape(-1, 1)  # Get the data point
            kNN_inds = indices[ni, 1:]  # Find the Euclidean kNNs

            for nj in range(kNN_num):  # For each Euclidean kNN connect with the Riemannian distance
                ind_j = kNN_inds[nj]  # kNN index
                p_j = data[ind_j, :].reshape(-1, 1)  # The kNN point
                temp_curve = lambda t: evaluate_failed_solution(p_i, p_j, t)
                # Note: Shortest path on graph prefers "low-weight" connections
                Weight_matrix[ni, ind_j] = curve_length(metric, temp_curve, tol=tol, limit=limit)

            if ni % 100 == 0:
                print("[Initialize Graph] [Processed point: {}/{}]".format(ni, N_data))

        # Make the weight matrix symmetric
        Weight_matrix = 0.5 * (Weight_matrix + Weight_matrix.T)

        self.New_Graph = csr_matrix(Weight_matrix, shape=(N_data, N_data))

        # Find the shortest path between all the points
        self.dist_matrix, self.predecessors = \
            shortest_path(csgraph=self.New_Graph, directed=False, return_predecessors=True)

        self.name = 'graph'



# If the solver failed provide the linear distance as the solution
def evaluate_failed_solution(p0, p1, t):
    # Input: p0, p1 (D x 1), t (T x 0)
    c = (1 - t) * p0 + t * p1  # D x T
    dc = np.repeat(p1 - p0, np.size(t), 1)  # D x T
    return c, dc


# This function computes the length of the geodesic curve
# The smaller the approximation error (tol) the slower the computation.
def curve_length(metric, curve, a=0, b=1, tol=1e-5, limit=50):
    # Input: curve a function of t returns (D x ?), [a,b] integration interval, tol error of the integration
    if callable(curve):
        # function returns: curve_length_eval = (integral_value, some_error)
        curve_length_eval = integrate.quad(lambda t: local_length(metric, curve, t), a, b, epsabs=tol, limit=limit)  # , number of subintervals
    else:
        print("TODO: Not implemented yet integration for discrete curve!\n")
        sys.exit(1)

    return curve_length_eval[0]


# This function computes the infinitesimal small length on a curve
def local_length(metric, curve, t):
    # Input: curve function of t returns (D X T), t (T x 0)
    c, dc = curve(t)  # [D x T, D x T]
    D = c.shape[0]
    dc = dc.T  # D x N -> N x D
    dc_rep = np.repeat(dc[:, :, np.newaxis], D, axis=2)  # N x D -> N x D x D
    M = metric.compute_riemannian_metric(c.T,var_rbfn=True).detach().numpy()  # N x D x D
    Mdc = np.sum(M * dc_rep, axis=1)  # N x D
    dist = np.sqrt(np.sum(Mdc * dc, axis=1))  # N x 1
    return dist


# An approximate graph based solver and a cubic spline result

def solver_graph(solver, metric, c0, c1, solution=None):
    """
     This function finds the approximate shortest path (geodesic) between two points on the manifold.
    The algorithm works like this:
    - Finding the nearest points in the graph to the start and end points.
    - Computing a discrete path between these points using the precomputed shortest paths in the graph.
    - Smoothing this discrete path using a heuristic method.
    - Using a cubic spline to interpolate the points on the smoothed path, creating a continuous curve that approximates the geodesic.
    """

    # The weight matrix
    W = solver.New_Graph.todense()

    # Find the Euclidean closest points on the graph to be used as fake start and end.
    _, c0_indices = solver.kNN_graph.kneighbors(c0.T)  # Find the closest kNN_num+1 points to c0
    _, c1_indices = solver.kNN_graph.kneighbors(c1.T)  # Find the closest kNN_num+1 points to c1
    ind_closest_to_c0 = np.nan  # The index in the training data closer to c0
    ind_closest_to_c1 = np.nan
    cost_to_c0 = 1e10
    cost_to_c1 = 1e10
    for n in range(solver.kNN_num - 1):  # We added one extra neighbor when we constructed the graph

        # Pick the next point in the training data tha belong in the kNNs of c0 and c1
        ind_c0 = c0_indices[0, n]  # kNN index from the training data
        ind_c1 = c1_indices[0, n]  # kNN index from the training data

        x_c0 = solver.data[ind_c0, :].reshape(-1, 1)  # The kNN point near to c0
        x_c1 = solver.data[ind_c1, :].reshape(-1, 1)  # The kNN point near to c1

        # Construct temporary straight lines
        temp_curve_c0 = lambda t: evaluate_failed_solution(c0, x_c0, t)
        temp_curve_c1 = lambda t: evaluate_failed_solution(c1, x_c1, t)

        # Shortest path on graph prefers "low-weight" connections
        temp_cost_c0 = curve_length(metric, temp_curve_c0, tol=solver.tol)
        temp_cost_c1 = curve_length(metric, temp_curve_c1, tol=solver.tol)

        # We found one of the  Euclidean kNNs that has closer Riemannian distance from the other kNNs we have checked.
        if temp_cost_c0 < cost_to_c0:
            ind_closest_to_c0 = ind_c0
            cost_to_c0 = temp_cost_c0

        if temp_cost_c1 < cost_to_c1:
            ind_closest_to_c1 = ind_c1
            cost_to_c1 = temp_cost_c1

    # The closest points in the graph to the test points c0, c1
    source_ind = ind_closest_to_c0
    end_ind = ind_closest_to_c1

    path = [end_ind]
    pairwise_lengths = []
    temp_ind = end_ind

    # Find the discrete path between source and sink. Each cell [i,j] keeps the previous point path before reaching j from i
    while True:
        prev_ind = solver.predecessors[source_ind, temp_ind]  # The previous point to reach the [goal == temp_ind]
        if prev_ind == -9999:  # There is not any other point in the path
            break
        else:
            path.append(prev_ind)
            pairwise_lengths.append(W[temp_ind, prev_ind])  # Weight/distance between the current and previous node
            temp_ind = prev_ind  # Move the pointer to one point close to the source.

    path.reverse()  # Reverse the path from [end, ..., source] -> [source, ..., end]
    inds = np.asarray(path)

    DiscreteCurve_data = solver.data[inds.flatten(), :]  # The discrete path on the graph

    # A heuristic to smooth the discrete path with a mean kernel
    DiscreteCurve_data = np.concatenate((c0.T, DiscreteCurve_data[1:-1, :], c1.T), axis=0)
    DiscreteCurve_new = np.empty((0, c0.shape[0]))
    for n in range(1, DiscreteCurve_data.shape[0]-1):
        new_point = (DiscreteCurve_data[n-1] + DiscreteCurve_data[n+1] + DiscreteCurve_data[n]) / 3
        DiscreteCurve_new = np.concatenate((DiscreteCurve_new, new_point.reshape(1, -1)), axis=0)
    DiscreteCurve_data = DiscreteCurve_new.copy()
    DiscreteCurve = np.concatenate((c0.T, DiscreteCurve_data, c1.T), axis=0)

    # Simple time parametrization of the curve
    N_points = DiscreteCurve.shape[0]  # Number of points in the discrete shortest path
    t = np.linspace(0, 1, num=N_points, endpoint=True)  # The time steps to construct the spline

    # Interpolate the points with a cubic spline.
    curve_spline = CubicSpline(t, DiscreteCurve)  # The continuous curve that interpolates the points on the graph
    dcurve_spline = curve_spline.derivative()  # The derivative of the curve
    curve = lambda t: evaluate_spline_solution(curve_spline, dcurve_spline, t)

    # Return the solution
    solution = {'curve': curve, 'solver': solver.name,
                'points': DiscreteCurve[1:-1, :], 'time_stamps': t[1:-1]}
    curve_length_eval = curve_length(metric, curve, tol=solver.tol, limit=solver.limit)
    logmap = dcurve_spline(0).reshape(-1, 1)  # The initial velocity
    logmap = curve_length_eval * logmap.reshape(-1, 1) / np.linalg.norm(logmap)  # Scaling for normal coordinates.
    failed = False

    return curve, logmap, curve_length_eval, failed, solution


def evaluate_spline_solution(curve, dcurve, t):
    # Input: t (Tx0), t_scale is used from the Expmap to scale the curve in order to have correct length,
    #        solution is an object that solver_bvp() returns
    c = curve(t)
    dc = dcurve(t)
    D = int(c.shape[0])

    # TODO: Why the t_scale is used ONLY for the derivative component?
    if np.size(t) == 1:
        c = c.reshape(D, 1)
        dc = dc.reshape(D, 1)
    else:
        c = c.T  # Because the c([0,..,1]) -> N x D
        dc = dc.T
    return c, dc
