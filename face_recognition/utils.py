import numpy as np

# Operation utils


def batch_cosine_similarity(x1, x2):
    '''
        x1,x2 must be l2 normalized
    '''

    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.multiply(x1, x2)
    s = np.sum(mul, axis=1)

    return s


def dist2id(distance, y, th=0.5):
    # Get 
    idx = np.argmax(distance)
    
    if distance[idx] >= th:
        return y[idx]
    else:
        return None
