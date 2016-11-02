def GradientDescentNormalize(data, max_iter=1000, step_size=0.1, tol=1e-4, lamb=0.01):
    #data is matrix of arrays [[1, symmetry, intensity], +/-1]
    feature_size = data[0][0].size
    weight = np.zeros(feature_size)
    sample_size = data.size
    step = step_size
    for i in range(0, max_iter):
        old_weight = weight.copy()
        grad = np.zeros(feature_size)
        norm = 2 * lamb * old_weight
        for feature, target in data:
            numerator = target * feature
            denominator = (1 + np.exp(target * np.dot(weight.T, feature)))
            grad += (numerator / denominator)
        grad /= sample_size
        weight = old_weight + (step * grad) + norm
        if np.linalg.norm(weight - old_weight) < tol:
            print(np.linalg.norm(weight - old_weight))
            break
        if (i % 100 == 0):
            print("Iteration: {}, Weight: {}, Gradient: {}".format(i, weight, grad))
            print("{}/{}".format(numerator, denominator))
    print("WEIGHT: {}".format(weight))
    print("Iterations: {}".format(i))
    return weight
    
def CalcError(data, weight):
    #data is matrix of arrays [[1, symmetry, intensity], +/-1]
    err = 0
    for feature, target in data:
        err += np.log(1 + np.exp((-1) * target * np.dot(weight.T, feature)))
    return err / data.size
    
    
def PolynomialTransform(data, order):
    # Only works on 2d data (i.e. feature of [1, x, y])
    new_data_array = []
    for feature, target in data:
        new_feature = feature.tolist() # new feature starts w/ copy/first order poly
        for i in range(2, order + 1):
            exp = i
            while exp >= 0:
                new_feature.append((feature[1] ** exp) * (feature[2] ** (2 - exp)))
                exp -= 1
        feature_array = np.array(new_feature)
        new_data_array.append(np.array([feature_array, target]))
    new_data = np.array(new_data_array)
    return new_data
