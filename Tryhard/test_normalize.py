def _normalize_scalar(var, old_min, old_max, new_min, new_max):
        """ Normalize scalar var from one range to another """
        return var / (old_max - old_min) * (new_max - new_min)

def _normalize_scalar_wrong( var, old_min, old_max, new_min, new_max):
        """ Normalize scalar var from one range to another """
        return ((new_max - new_min) * (var - old_min) / (old_max - old_min)) + new_min


old_min  = -1
old_max = 1

new_min = -0.1
new_max = 0.1

var = -0.5

new_var = _normalize_scalar(var, old_min, old_max,new_min,new_max)
print (new_var)

new_var_wrong = _normalize_scalar_wrong(var, old_min, old_max,new_min,new_max)
print (new_var_wrong)