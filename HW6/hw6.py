import random as rng
from draw_mql import *

# Arm, leg, arm leg. Remember to flip the left from the right (left = 9 - 
# right value)
# draw_mql(9, 15, 0, 1, 1, 9, 9, 4, 3, 6, 7)


sex_cdf = {
    (2.0/3.0): 'male',
    1.0: 'female'
}

# 0 is hexagon (female) and 1 is circle (male)
bt_obs_m_cdf = {
    0.3: 0,
    1.0: 1
}

bt_obs_f_cdf = {
    0.7: 0,
    1.0: 1
}

# Body type is basically determined 1-1 from sex, so just do a mapping here
# 1 is circle, 0 is hex
sex_bt_map = {
    'male': 1,
    'female': 0
}

bt_obs_bt_map = {
    0: bt_obs_f_cdf,
    1: bt_obs_m_cdf
}

riaa_f_cdf = {
    (10.0/90.0): 5,
    (40.0/90.0): 6,
    1.0:         7
}

riaa_m_cdf = {
    (10.0/90.0): 3,
    (30.0/90.0): 4,
    (60.0/90.0): 5,
    (80.0/90.0): 6,
    1.0:         7
}

riaa_map = {
    'female': riaa_f_cdf,
    'male': riaa_m_cdf
}

rila_f_cdf = {
    (10.0/90.0): 3,
    (30.0/90.0): 4,
    (60.0/90.0): 5,
    (80.0/90.0): 6,
    1.0:         7
}

rila_m_cdf = {
    (50.0/90.0): 3,
    (80.0/90.0): 4,
    1.0:         5
}

rila_map = {
    'female': rila_f_cdf,
    'male': rila_m_cdf
}

# The following tables use modifier values rather than absolute values.
outer_angle__modifier_cdf = {
    (30.0/110.0): -2,
    (50.0/110.0): -1,
    (60.0/110.0): 0,
    (80.0/110.0): 1,
    1.0:          2
}

obs_err_1_modifier_cdf = {
    0.7: 0,
    0.9: 1,
    1.0: 2
}

obs_err_2_modifier_cdf = {
    0.3: -1,
    0.7: 0,
    0.9: 1,
    1.0: 2
}

obs_err_typ_modifier_cdf = {
    0.1: -2,
    0.3: -1,
    0.7: 0,
    0.9: 1,
    1.0: 2
}

obs_err_8_modifier_cdf = {
    0.1: -2,
    0.3: -1,
    0.7: 0,
    1.0: 1
}

obs_err_9_modifier_cdf = {
    0.1: -2,
    0.3: -1,
    1.0: 0
}

def sample_distr(cdf_table):
    rand = rng.random()
    prev_key = 0.0
    for key in cdf_table:
        if rand >= prev_key and rand < key:
            return cdf_table[key]

def get_sex():
    return sample_distr(sex_cdf)

# Expects 'male' or 'female'
def get_bt(sex):
    return sex_bt_map[sex]

# Expects 0 (hexagon) or 1 (circle)
def get_obs_bt(body_type):
    return sample_distr(bt_obs_bt_map[body_type])

# Expects 'male' or 'female'
def get_riaa(sex):
    return sample_distr(riaa_map[sex])

# Expects 1-9
def get_obs_riaa(riaa):
    return get_obs_angle_val(riaa)

# Expects 1-9
def get_obs_liaa(riaa):
    return get_obs_angle_val(flip_angle(riaa))

# Expects 1-9
def get_roaa(riaa):
    return riaa + sample_distr(outer_angle__modifier_cdf)

# Expects 1-9
def get_obs_roaa(roaa):
    return get_obs_angle_val(roaa)

# Expects 1-9
def get_obs_loaa(roaa):
    return get_obs_angle_val(flip_angle(roaa))

# Expects 'male' or 'female'
def get_rila(sex):
    return sample_distr(rila_map[sex])

def get_obs_rila(rila):
    return get_obs_angle_val(rila)

def get_obs_lila(rila):
    return get_obs_angle_value(flip_angle(rila))

def get_rola(rila):
    return rila + sample_distr(outer_angle__modifier_cdf)

def get_obs_rola(rola):
    return get_obs_angle_val(rola)

def get_obs_lola(rola):
    return get_obs_angle_val(flip_angle(rola))

# Expected range: 1-9
def get_obs_angle_val(x):
    if x == 1:
        table = obs_err_1_modifier_cdf
    elif x == 2:
        table = obs_err_2_modifier_cdf
    elif x == 8:
        table = obs_err_8_modifier_cdf
    elif x == 9:
        table = obs_err_9_modifier_cdf
    else:
        table = obs_err_typ_modifier_cdf
    return x + sample_distr(table)

def flip_angle(x):
    return 9 - x + 1

def sample_model():
    sex = get_sex()
    bt = get_bt(sex)
    obs_bt = get_obs_bt(bt)

    riaa = get_riaa(sex)
    obs_riaa = get_obs_angle_val(riaa)
    liaa = flip_angle(riaa)
    obs_liaa = get_obs_angle_val(liaa)

    roaa = get_roaa(riaa)
    obs_roaa = get_obs_angle_val(roaa)
    loaa = flip_angle(roaa)
    obs_loaa = get_obs_angle_val(loaa)

    rila = get_rila(sex)
    obs_rila = get_obs_angle_val(rila)
    lila = flip_angle(rila)
    obs_lila = get_obs_angle_val(lila)

    rola = get_rola(rila)
    obs_rola = get_obs_angle_val(rola)
    lola = flip_angle(rola)
    obs_lola = get_obs_angle_val(lola)

    return {
        'sex': sex,
        'bt': bt,
        'obs_bt': obs_bt,
        'riaa': riaa,
        'obs_riaa': obs_riaa,
        'liaa': liaa,
        'obs_liaa': obs_liaa,
        'roaa': roaa,
        'obs_roaa': obs_roaa,
        'loaa': loaa,
        'obs_loaa': obs_loaa,
        'rila': rila,
        'obs_rila': obs_rila,
        'lila': lila,
        'obs_lila': obs_lila,
        'rola': rola,
        'obs_rola': obs_rola,
        'lola': lola,
        'obs_lola': obs_lola,
    }

if __name__ == '__main__':
    rng.seed(42)
    for i in range(10):
        cur_mql = sample_model()
        draw_mql(9, 15, cur_mql['bt'],     cur_mql['riaa'],     cur_mql['rila'],     cur_mql['liaa'],     cur_mql['lila'],     cur_mql['roaa'],     cur_mql['rola'],     cur_mql['loaa'],     cur_mql['lola'])
        draw_mql(9, 15, cur_mql['obs_bt'], cur_mql['obs_riaa'], cur_mql['obs_rila'], cur_mql['obs_liaa'], cur_mql['obs_lila'], cur_mql['obs_roaa'], cur_mql['obs_rola'], cur_mql['obs_loaa'], cur_mql['obs_lola'])