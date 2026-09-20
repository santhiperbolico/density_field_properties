"""Rockstar and consistent-trees column index maps for catalog readers."""

ROCKSTAR_T_OVER_U_COLUMN = 37
UNIT_T_OVER_U_COLUMN = 56

UNIT_HLIST_COLUMNS = {
    "id": 1,
    "pid": 5,
    "Rvir": 11,
    "x": 17,
    "y": 18,
    "z": 19,
    "Spin": 26,
    "Rs_Klypin": 37,
    "M200b": 39,
    "ba": 46,
    "ca": 47,
    "t_over_u": UNIT_T_OVER_U_COLUMN,
}

ROCKSTAR_LIST_COLUMNS = {
    "halo_id": 0,
    "halo_x": 8,
    "halo_y": 9,
    "halo_z": 10,
    "halo_m200b": 20,
    "t_over_u": ROCKSTAR_T_OVER_U_COLUMN,
}
