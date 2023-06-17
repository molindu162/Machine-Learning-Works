import numpy as np
class Model:
    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s

    def initialize_params(self ,dim):
        w = np.zeros((dim, 1))
        b = 0.0
        return w, b
    def propagate(self, w, b, X, Y):
        m = X.shape[1]
        # forward prooagation
        A = self.sigmoid(np.dot(w.T, X) + b)
        # cost = np.sum(-Y * np.log(A) - (1 - Y) * np.log(1 - A)) / m
        cost = 0
        for i in range(m):
            cost += -Y[0][i] * np.log(A[0][i]) - (1 - Y[0][i]) * np.log(1 - A[0][i])

        cost /= m

        # Backward propagation
        dL_dw = np.dot(X, (A - Y).T) / m
        dL_db = np.sum(A - Y) / m

        derivatives = {
            "dL_dw": dL_dw,
            "dL_db": dL_db
        }

        return derivatives, cost

    def optimize(self,w, b, X, Y, learning_rate=0.005, num_iterations=1000, print_cost=False):
        costs = []
        for i in range(num_iterations):
            grads, cost = self.propagate(w, b, X, Y)
            dL_dw = grads["dL_dw"]
            dL_db = grads["dL_db"]
            w = w - learning_rate * dL_dw
            b = b - learning_rate * dL_db

            if (print_cost):
                if (i % 10 == 0):
                    costs.append(cost)
                    print(f"Cost at iteration {i} : {cost}")

        params = {
            "w": w,
            "b": b
        }

        derivatives = {
            "dL_dw": dL_dw,
            "dL_db": dL_db
        }

        return params, derivatives, costs

    def predict(self, w, b, X):
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        y_prediction = (A >= 0.5) * 1
        assert (y_prediction.shape == (1, X.shape[1]))
        return y_prediction

    def fit(self, X, Y):
        w, b = self.initialize_params(X.shape[0])
        params, derivatives, costs = self.optimize(w, b, X, Y)
        self.w = params["w"]
        self.b = params["b"]

