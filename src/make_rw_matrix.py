""" by Paulo CVS"""

import igraph
import numpy as np
# import sys
import scipy.linalg
import scipy.sparse.linalg
from logging import info

# from epidspread.src.simu import generate_graph, initialize_gradients  # Now imported from aux
# from aux import generate_graph, initialize_gradients



def calc_rw_transition_matrix(g, out=None):
    """
    Calculates the random walk (RW) transition matrix p_ij for a weighted graph, where p_ij is the probability that
    a random walker at node i moves to node j.

    Parameters
    ----------
    g : igraph.Graph
        The graph object, with a float attribute called "gradient" defined for each node.
    out : np.ndArray
        A square 2D array (size = number of vertices) to store the resulting matrix.
        If not informed, a new one is allocated. Either way, it is returned by the function.

    Returns
    -----------
    np.ndArray
    """

    nvertices = g.vcount()

    # Initializes the matrix with zeros
    if out is None:
        # Container not informed. Creates a new one.
        out = np.zeros((nvertices, nvertices), dtype=float)
    else:
        # Else, it is assumed to be an allocated array of the same shape as above. Sets all elements to zero.
        out[:] = 0.

    # Iterate over nodes and neighbors to construct the "weighted adjacency matrix" (transition matrix)
    for i, v in enumerate(g.vs):
        grad_sum = 0.
        for v_neigh in v.neighbors():
            j = v_neigh.index
            grad = v_neigh["gradient"]
            out[i, j] = grad  # Sets the matrix entry to the gradient value.
            grad_sum += grad

        # Finally normalizes the line. A lonely node could have grad_sum == 0.
        if grad_sum > 0.:
            out[i] /= grad_sum

    return out


def calc_matrix_leading_eigenvector(mat, sparse=True):
    """
    Returns (and possibly constructs) a 1D numpy array with the principal eigenvector of a matrix, that is, the one
    with greatest (real part of the) eigenvalue.
    For a stochastic matrix corresponding to a random walk, entry i is the probability that an individual is found
    there during the STEADY STATE of the random walk.

    The eigenvector is normalized by its sum, so the sum of its components results in 1.

    Parameters
    ----------
    mat : np.ndArray
        The 2D square array with the matrix to be diagonalized.
    sparse : bool
        If True (default), assumes that mat is sparse and uses a specialized diagonalization method.

    Returns
    -------
    np.ndArray
        A 1D array with the normalized leading eigenvector.
    """

    if sparse:
        eigval, eigvec = scipy.sparse.linalg.eigs(mat.transpose(), k=1)
    else:
        eigval, eigvec = scipy.linalg.eig(mat.transpose())

    # Reshapes and rearranges the array, then normalizes by its sum (pre-assuming it is not zero, shouldn't be!).
    eigvec = eigvec.real.flatten().ravel()
    eigvec /= np.sum(eigvec)

    return eigvec


def distribute_agents_by_weights(nvertices, nagents, expidx, weights):
    """
    Initialize the location of the agents. The nagents per vertex is random, according to non-uniform probabilities
    for each vertex (given by 'weights'), but the agents id is NOT. The ids of a vertex will be all sequential

    Args:
    nvertices(int): number of vertices in the map
    nagents(int): number of agents
    expidx(int): experiment index
    weights(np.ndArray): probability of choosing each node

    Returns:
    list of list: each element corresponds to a vertex and contains the indices of the
    vertices
    """
    info('exp:{} Distributing agents in the map (weighted version)...'.format(expidx))

    # Creates an array with randomly chosen vertex ids. Each id is the destination of an agent.
    chosen_vertices = np.random.choice(nvertices, size=nagents, replace=True, p=weights)

    # From the previous choices of vertices, creates the 'particles' list with the agents that should be at each vertex.
    # Not very optimal, but shouldn't take long to execute.
    particles = [list() for _ in range(nvertices)]
    for agent_id, vertex_id in enumerate(chosen_vertices):
        particles[vertex_id].append(agent_id)

    return particles


def main():
    expidx = "1swtmiiu"  # Sei lá
    view_results = True

    # ---------------------------
    # Creates and initializes a graph and its gradient values
    g, coords = generate_graph("er", nvertices=1000, avgdegree=4,
                       latticethoroidal=-1, baoutpref=-1, wsrewiring=-1, wxalpha=-1,
                       expidx=expidx, randomseed=4, wxparamspath="config/waxman_params.csv", tmpdir="tmp_dir")
    g = initialize_gradients(g, coords, ngaussians=1, sigma=0.5, expidx=expidx)

    nvertices = g.vcount()
    nagents = 1 * nvertices  # Just to test.

    # -----------------
    # Transition matrix and its normalized leading eigenvector
    rw_transmat = calc_rw_transition_matrix(g)
    node_probabs = calc_matrix_leading_eigenvector(rw_transmat, sparse=True)

    # Initializes the agents on the graph according to the calculated node probabilites.
    particles = distribute_agents_by_weights(nvertices, nagents, expidx, weights=node_probabs)

    # ------------------
    # Optional - visiualization of the results as plots
    if view_results:
        info("Importing pyplot and plotting...")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1)

        # View the relationship between each node's gradient values and its RW eigenvector entry
        ax = axes[0]
        gradients = np.fromiter((node["gradient"] for node in g.vs), dtype=float)
        ax.plot(gradients, node_probabs, "bo", alpha=0.5)
        ax.set_xlabel("gradient")
        ax.set_ylabel("eigenvector")

        # View the actual number of agents with respect to the eigenvector (i.e., the probabilities)
        ax = axes[1]
        nagents_at_each_vertex = np.fromiter((len(a) for a in particles), dtype=int)
        ax.plot(node_probabs, nagents_at_each_vertex, "gs", alpha=0.5)
        ax.set_xlabel("eigenvector")
        ax.set_ylabel("number of agents")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
