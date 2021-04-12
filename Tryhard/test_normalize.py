def _normalize_scalar(var, old_min, old_max, new_min, new_max):
        """ Normalize scalar var from one range to another """
        return var / (old_max - old_min) * (new_max - new_min)


old_min  = -3.0
old_max = 4.0

new_min = -0.5
new_max = 0.5

var = -1.7

new_var = _normalize_scalar(var, old_min, old_max,new_min,new_max)
print (new_var)