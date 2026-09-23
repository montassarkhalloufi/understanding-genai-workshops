from itertools import product


def relu(value):
    return max(0, value)


def xor_network(left, right):
    first = relu(left + right)
    second = relu(left + right - 1)
    return first - 2 * second


for left, right in product((0, 1), repeat=2):
    prediction = xor_network(left, right)
    expected = int(left != right)
    assert prediction == expected
