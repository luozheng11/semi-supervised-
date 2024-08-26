import numpy as np


def sigmoid_rampup(current, rampup_length):
    '''
    根据当前的训练步骤和 ramp-up 的长度来计算一个在训练初期逐渐增加的权重
    '''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def zero_cosine_rampdown(current, epochs):
    return float(.5 * (1.0 + np.cos((current - 1) * np.pi / epochs)))