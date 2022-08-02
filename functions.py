import numpy as np


# Naive implementation:
def calc_beta_naive(length, stk, ind):
    output = []

    for symbol in stk:
        beta_vector = []

        for i in range(length, len(symbol)):
            stock_slice = symbol[i - length:i]
            index_slice = ind[i - length:i]
            x_mean = np.mean(index_slice)
            y_mean = np.mean(stock_slice)
            beta = np.sum((index_slice - x_mean) * (stock_slice - y_mean)) / np.sum((index_slice - x_mean) ** 2)
            beta_vector.append(beta)

        output.append(beta_vector)

    return np.asarray(output).swapaxes(0, 1)


# Final Implementation:
def calc_beta(length, stk, ind):
    output = []

    for i in range(length, len(stk)):
        X = [ind[i - length:i]]
        X = np.vstack([[1] * length, X]).T
        Y = stk[i - length:i]
        beta_vector = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(Y))
        output.append(beta_vector[1])

    return np.asarray(output)


# Calculates Return from Log Return
def calc_return(lr, pct):
    if pct:
        return (np.exp(lr) - 1) * 100

    else:
        return np.exp(lr) - 1
