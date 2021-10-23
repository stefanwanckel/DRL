import numpy as np


def recover_homogenous_affine_transformation(p, p_prime):
    '''
    Find the unique homogeneous affine transformation that
    maps a set of 3 points to another set of 3 points in 3D
    space:

        p_prime == np.dot(p, R) + t

    where `R` is an unknown rotation matrix, `t` is an unknown
    translation vector, and `p` and `p_prime` are the original
    and transformed set of points stored as row vectors:

        p       = np.array((p1,       p2,       p3))
        p_prime = np.array((p1_prime, p2_prime, p3_prime))

    The result of this function is an augmented 4-by-4
    matrix `A` that represents this affine transformation:

        np.column_stack((p_prime, (1, 1, 1))) == \
            np.dot(np.column_stack((p, (1, 1, 1))), A)

    Source: https://math.stackexchange.com/a/222170 (robjohn)
    '''

    # construct intermediate matrix
    Q = p[1:] - p[0]
    #Q = p[1:]
    Q_prime = p_prime[1:] - p_prime[0]

    # calculate rotation matrix
    R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
               np.row_stack((Q_prime, np.cross(*Q_prime))))

    # calculate translation vector
    t = p_prime[0] - np.dot(p[0], R)

    # calculate affine transformation matrix
    return np.column_stack((np.row_stack((R, t)),
                            (0, 0, 0, 1)))


if __name__ == "__main__":
    lst_p1 = [[1, 2, 0], [2, 4, 1], [1, -1, -1]]
    lst_p1 = np.array(lst_p1)
    lst_p2 = [[0, 3, 2], [-1, 2, 4], [1, 3, -1]]
    lst_p2 = np.array(lst_p2)
    T = recover_homogenous_affine_transformation(lst_p1, lst_p2)
    T = np.transpose(T)
    np.printoptions(precision=2)
    T[abs(T) < 1e-10] = 0

    print(T)
