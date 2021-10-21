
"""
    FOllowing information was retrieved from rs-enumerate-devices -c in tools folder
    Intrinsic of "Color" / 1280x720 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y16}
     Width:      	1280
     Height:     	720
     PPX:        	634.047607421875
     PPY:        	355.952941894531
     Fx:         	921.931396484375
     Fy:         	919.706237792969
     Distortion: 	Inverse Brown Conrady
     Coeffs:     	0  	0  	0  	0  	0  
     FOV (deg):  	69.54 x 42.75

     the intrinsic camera matrix looks like this
     3x3 (2D embedded in affine space)
     M_a = [
            f_x  0   c_x
            0    f_y c_y
            0    0   1
            ]
     """
import numpy as np

PPX = 634.047607421875
PPY = 355.952941894531
Fx = 921.931396484375
Fy = 919.706237792969
M_a = np.zeros((3, 3))
M_a[0, 0] = Fx
M_a[0, 1] = 0
M_a[0, 2] = PPX
M_a[1, 0] = 0
M_a[1, 1] = Fy
M_a[1, 2] = PPY
M_a[2, 0] = 0
M_a[2, 1] = 0
M_a[2, 2] = 1

print(M_a)
np.save("../camera_info/D400_intrinsic_matrix",
        M_a, allow_pickle=True, fix_imports=True)
