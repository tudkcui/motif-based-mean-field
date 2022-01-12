from scipy.special import comb


def encode_x_nary(N, j, base):
    xnary = []
    while j != 0:
        bit = j % base
        xnary.insert(0, bit)
        j = j // base
    return [0] * (N - len(xnary)) + xnary


def decode_x_nary(xnary, base):
    sum_x_nary = 0
    for j in range(len(xnary)):
        sum_x_nary += base ** j * xnary[len(xnary) - j - 1]
    return sum_x_nary


def multinomial(params):
    if len(params) == 1:
        return 1
    return comb(sum(params), params[-1]) * multinomial(params[:-1])