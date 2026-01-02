import numpy as np


def allmax(a):
    if len(a) == 0:
        return []
    all_ = [0]
    max_ = a[0]
    for i in range(1, len(a)):
        if a[i] > max_:
            all_ = [i]
            max_ = a[i]
        elif a[i] == max_:
            all_.append(i)
    return (max_, all_)


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


def kl(x, y):
    if (x == 0):
        if (y == 1.):
            return np.infty
        return np.log(1. / (1. - y))
    if (x == 1):
        if (y == 0.):
            return np.infty
        return np.log(1. / y)
    if (y == 0) or (y == 1):
        return np.infty
    return x * np.log(x / y) + (1. - x) * np.log((1. - x) / (1. - y))


def search_up(f, up, down, epsilon=0.0001):
    mid = (up + down) / 2
    if (up - down > epsilon):
        if f(mid):
            return search_up(f, up, mid)
        else:
            return search_up(f, mid, down)
    else:
        if f(up):
            return up
        return down


def search_down(f, up, down, epsilon=0.0001):
    mid = (up + down) / 2
    if (up - down > epsilon):
        if f(mid):
            return search_down(f, mid, down)
        else:
            return search_down(f, up, mid)
    else:
        if f(down):
            return down
        return up
