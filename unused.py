# these functions generate vectors with step of step
def generate_l_pairs():
    L_values = np.arange(L_MIN, L_MAX + step, step)
    L_values[-1] = L_MAX  # so that the code stops at A_MAX instead of going overboard
    return [
        [[float(L_values[i]), A_MIN, B_MIN], [float(L_values[i + 1]), A_MIN, B_MIN]]
        for i in range(len(L_values) - 1)
    ]


def generate_a_pairs():
    A_values = np.arange(A_MIN, A_MAX + step, step)
    A_values[-1] = A_MAX  # so that the code stops at A_MAX instead of going overboard
    return [
        [[L_MIN, float(A_values[i]), B_MIN], [L_MIN, float(A_values[i + 1]), B_MIN]]
        for i in range(len(A_values) - 1)
    ]


def generate_b_pairs():
    B_values = np.arange(B_MIN, B_MAX + step, step)
    B_values[-1] = B_MAX  # so that the code stops at A_MAX instead of going overboard
    return [
        [[L_MIN, A_MIN, float(B_values[i])], [L_MIN, A_MIN, float(B_values[i + 1])]]
        for i in range(len(B_values) - 1)
    ]


# a func for determining is a single lab point in adobe rgb (delete later)
def is_within_adobe_rgb_single(L, a, b):

    xyz = Lab_to_XYZ([L, a, b])

    rgb = XYZ_to_RGB(
        xyz,
        illuminant_XYZ="D65",  # do we use d65 or d50?
        illuminant_RGB=RGB_COLOURSPACE_ADOBE_RGB1998.whitepoint,
        RGB_COLOURSPACE=RGB_COLOURSPACE_ADOBE_RGB1998,
        colourspace="Adobe RGB (1998)",
    )

    return np.all(rgb >= 0) and np.all(rgb <= 1)
