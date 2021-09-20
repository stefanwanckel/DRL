import urkin
import numpy as np

n = 10000
sample_pos = [-1.6, -1.4, -2.28, -0.59, 1.60, 0.023]
for i in range(n):
    
    q = sample_pos
    ee = urkin.fk(q, urkin.UR5E_DH)
    sols = urkin.ik(ee, urkin.UR5E_DH)
    found = False
    for sol in sols:
        ok = True
        for j in range(6):
            #revolution-independent angle comparison:
            if np.abs(np.cos(sol[j]) - np.cos(q[j])) > 1e-9 \
                    or np.abs(np.sin(sol[j]) - np.sin(q[j])) > 1e-9:
                ok = False
                break
        if ok:
            found = True
    minNorm = np.zeros(6)
    bestSol = []
    for sol in sols:
        crrNorm = np.linalg.norm((np.asarray(sample_pos),np.asarray(sol)),axis=0,ord=2)
        if minNorm == np.zeros(6) or minNorm > crrNorm:
            minNorm = crrNorm
            bestSol = np.asarray(sol)

    if not found:
        print("forwards then inverse kinematics of joints", q, " had no solution")
    print("Best solution was: ", bestSol)          
print("ran", n, "tests")
