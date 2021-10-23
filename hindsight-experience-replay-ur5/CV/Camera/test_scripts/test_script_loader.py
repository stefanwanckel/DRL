import numpy as np
np.set_printoptions(precision=2)


def get_custom_T_matrix(points_cf_1, points_cf_2):

    Points_1 = np.array(points_cf_1)[:, :3]
    Points_2 = np.array(points_cf_2)[:, :3]
    nr_points = Points_1.shape[0]
 # construct p
    b = Points_1.flatten()

  # construct A
    A = np.zeros((nr_points*3, 9))
    for i in range(0, nr_points):
        A[0+i*3, :3] = Points_2[i, :]
        A[0+i*3, 6] = 1

        A[1+i*3, 1] = Points_2[i, 0]
        A[1+i*3, 3] = Points_2[i, 1]
        A[1+i*3, 4] = Points_2[i, 2]
        A[1+i*3, 7] = 1

        A[2+i*3, 2] = Points_2[i, 0]
        A[2+i*3, 4] = Points_2[i, 1]
        A[2+i*3, 5] = Points_2[i, 2]
        A[2+i*3, 8] = 1
    x = np.linalg.lstsq(A, b)
    T_cf1_cf2 = np.zeros((4, 4))
    T_cf1_cf2[0, :3] = x[:3]
    T_cf1_cf2[:3, 3] = x[6:]
    T_cf1_cf2[3, :] = np.array([0, 0, 0, 1])
    T_cf1_cf2[1, 0] = x[1]
    T_cf1_cf2[1, 1:3] = x[3:5]
    T_cf1_cf2[2, 0] = x[2]
    T_cf1_cf2[2, 1:3] = x[4:6]

    return T_cf1_cf2


if __name__ == "__main__":
    lst_p1 = [[1, 2, 0], [2, 4, 1], [1, -1, -1], [3, 2, 2]]
    lst_p2 = [[0, 3, 2], [-1, 2, 4], [1, 3, -1], [-2, 1, 2]]
    T = get_custom_T_matrix(lst_p1, lst_p2)
    np.printoptions(precision=2)
    print(T)
