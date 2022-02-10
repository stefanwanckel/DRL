def map_sim_2_rg2(rg2_val):
    rg2_closed = -0.41
    rg2_open = 0.61

    sim_closed = 0.2
    sim_open = -0.45

    slope = (sim_closed - sim_open) / (rg2_closed - rg2_open)
    sim_val = sim_open + slope * (rg2_val - rg2_open)
    return sim_val


print(map_sim_2_rg2(-0.4))

print(map_sim_2_rg2(0))

print(map_sim_2_rg2(0.5))
