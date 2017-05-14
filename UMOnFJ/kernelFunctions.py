
def rbf_kernel(X,Y,gamma):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    gram_matrix[i, j] = np.exp(-gamma * np.sum((np.absolute(x - y)) ** 2))
        return gram_matrix

def laplacean(X,Y,gamma):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    gram_matrix[i, j] = np.exp(-gamma * np.sum((np.absolute(x - y))))
        return gram_matrix

def sinc(X,Y):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x - y))
                    gram_matrix[i, j] = np.sinc(sum)
        return gram_matrix

def sinc2(X,Y):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x - y)) ** 2
                    gram_matrix[i, j] = np.sinc(sum)
        return gram_matrix

def quadratic(X,Y,c):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x - y)) ** 2
                    gram_matrix[i, j] = 1 - sum / (sum + c)
        return gram_matrix

def multiquadric(X,Y,c):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x - y)) ** 2
                    gram_matrix[i, j] = - np.sqrt(sum + c * c)
        return gram_matrix

def inverse_multiquadric(X,Y,c):
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    sum = np.sum(np.absolute(x - y)) ** 2
                    gram_matrix[i, j] = 1 / np.sqrt(sum + c * c)
        return gram_matrix
