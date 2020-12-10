"""Utility functions to process state."""
from typing import Tuple

import numpy as np
from numba import njit


# @jit
# def normalize(array_in):
#     """nx2 or mxnx2"""
#     if len(array_in.shape) == 2:
#         vec, fac = normalize_array(array_in)
#         return vec, fac
#     factors = []
#     vectors = []
#     for m in array_in:
#         vec, fac = normalize_array(m)
#         vectors.append(vec)
#         factors.append(fac)

#     return np.array(vectors), np.array(factors)


@njit
def vector_angles(vecs: np.ndarray) -> np.ndarray:
    """Calculate angles for an array of vectors
    :param vecs: nx2 ndarray
    :return: nx1 ndarray
    """
    ang = np.arctan2(vecs[:, 1], vecs[:, 0])  # atan2(y, x)
    return ang


@njit
def left_normal(vecs: np.ndarray) -> np.ndarray:
    vecs = np.fliplr(vecs) * np.array([-1.0, 1.0])
    return vecs


@njit
def right_normal(vecs: np.ndarray) -> np.ndarray:
    vecs = np.fliplr(vecs) * np.array([1.0, -1.0])
    return vecs


@njit
def normalize(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize nx2 array along the second axis
    input: [n,2] ndarray
    output: (normalized vectors, norm factors)
    """
    norm_factors = []
    for line in vecs:
        norm_factors.append(np.linalg.norm(line))
    norm_factors = np.array(norm_factors)
    normalized = vecs / np.expand_dims(norm_factors, -1)
    # get rid of nans
    for i in range(norm_factors.shape[0]):
        if norm_factors[i] == 0:
            normalized[i] = np.zeros(vecs.shape[1])
    return normalized, norm_factors


@njit
def desired_directions(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given the current state and destination, compute desired direction."""
    destination_vectors = state[:, 4:6] - state[:, 0:2]
    directions, dist = normalize(destination_vectors)
    return directions, dist


@njit
def vec_diff(vecs: np.ndarray) -> np.ndarray:
    """r_ab
    r_ab := r_a − r_b.
    """
    diff = np.expand_dims(vecs, 1) - np.expand_dims(vecs, 0)
    return diff


def each_diff(vecs: np.ndarray, keepdims=False) -> np.ndarray:
    """
    :param vecs: nx2 array
    :return: diff with diagonal elements removed
    """
    diff = vec_diff(vecs)
    # diff = diff[np.any(diff, axis=-1), :]  # get rid of zero vectors
    diff = diff[
        ~np.eye(diff.shape[0], dtype=bool), :
    ]  # get rif of diagonal elements in the diff matrix
    if keepdims:
        diff = diff.reshape(vecs.shape[0], -1, vecs.shape[1])

    return diff


@njit
def speeds(state: np.ndarray) -> np.ndarray:
    """Return the speeds corresponding to a given state."""
    #     return np.linalg.norm(state[:, 2:4], axis=-1)
    speed_vecs = state[:, 2:4]
    speeds_array = np.array([np.linalg.norm(s) for s in speed_vecs])
    return speeds_array


@njit
def center_of_mass(vecs: np.ndarray) -> np.ndarray:
    """Center-of-mass of a given group"""
    return np.sum(vecs, axis=0) / vecs.shape[0]


@njit
def minmax(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_min = np.min(vecs[:, 0])
    y_min = np.min(vecs[:, 1])
    x_max = np.max(vecs[:, 0])
    y_max = np.max(vecs[:, 1])
    return (x_min, y_min, x_max, y_max)

def turn_vector_around_other(vector, basev, referencev) :
    # Get angle between vector and referencev with respect to basev
    theta = np.arccos(np.dot(basev + referencev, vector)/(np.linalg.norm(basev + referencev)*np.linalg.norm(vector)))
    # Translate target, turn by rotation matrix and translate back
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    target = np.matmul(rot_mat,(vector - basev)) + basev
    return target

def smoke(state,R_sm,fire,dR_sm,dc_health,dc_panic,dt=1) :
    R_sm += dR_sm*dt
    n_people = len(state)
    x,y = [state[:,0],state[:,1]]
    x_smoke,y_smoke = [fire[:,0][0]+(fire[:,0][-1]-fire[:,0][0])/2, fire[:,1][0]+(fire[:,1][-1]-fire[:,1][0])/2]
    c_smoke,c_health,c_panic = [state[:,9],state[:,10],state[:,11]]
    index = np.where((x-x_smoke)**2+(y-y_smoke)**2<R_sm**2)
    d_rel = 1-np.sqrt(((x[index]-x_smoke)**2+(y[index]-y_smoke)**2))/R_sm
    c_smoke[index] = d_rel
    c_health[index] -= dc_health*d_rel*dt  
    c_health[np.where(c_health<0)] = 0
    c_panic[index] += dc_panic*d_rel*dt
    c_panic[np.where(c_panic>1)] = 1     
    state[:,9] = c_smoke
    state[:,10] = c_health
    state[:,11] = c_panic
    
    return state, R_sm