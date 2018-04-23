import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime

import argparse



def create_dataset(N, M, data_range=(-1, 7), func=np.sin):
    xs = np.random.uniform(*data_range, N)
    ys = func(xs)

    ys = ys + np.random.randn(*ys.shape)

    features = convert_features(xs, M)

    return xs, ys, features


def convert_features(xs, M):
    features = np.concatenate([
                   np.power(xs.reshape(-1, 1), i) for i in range(M)
               ],
               axis=1)
    return features

# def convert_features(xs, M, means, var):
#     return - np.power(xs.reshape(-1, 1) - means, 2) / (2 * np.power(var, 2))


def posterior(observed_xs, ys, prior_mean, prior_var, noise_var):
    dim = prior_var.shape[0]

    posterior_var = np.zeros((dim, dim))

    posterior_mean = np.zeros((1, dim))

    for x, y in zip(observed_xs, ys):
        x = x.reshape(1, -1)
        posterior_var += np.dot(x.T, x)
        posterior_mean += y * x

    posterior_var = noise_var * posterior_var + prior_var

    posterior_mean = np.dot(noise_var * posterior_mean + np.dot(prior_mean, prior_var),
                            np.linalg.inv(posterior_var))

    return posterior_mean, posterior_var


def predictive(xs, mean, var, noise_var):
    dim = var.shape[0]
    features = convert_features(xs, dim)

    mean_star = np.dot(features, mean.T).reshape(-1)
            
    var_star = np.diag(np.dot(np.dot(features, np.linalg.inv(var)), features.T))

    std_star = np.sqrt(var_star)
    return mean_star, mean_star - std_star, mean_star + std_star


def evidence(ys, priors, posteriors, noise_var):
    print("yn",noise_var * np.sum(np.power(ys, 2)))
    print("const", - np.log(noise_var) + np.log(2 * np.pi))
    print("prior", np.dot(np.dot(priors[0], priors[1]), priors[0].T) - np.log(np.linalg.det(priors[1])))
    print("post", - np.dot(np.dot(posteriors[0], posteriors[1]), posteriors[0].T))
    print("post_det", + np.log(np.linalg.det(posteriors[1])))
    return -0.5 * (noise_var * np.sum(np.power(ys, 2)) - np.log(noise_var) + np.log(2 * np.pi) + np.dot(np.dot(priors[0], priors[1]), priors[0].T) - np.log(np.linalg.det(priors[1])) - np.dot(np.dot(posteriors[0], posteriors[1]), posteriors[0].T) + np.log(np.linalg.det(posteriors[1])))


def hoge(x):
    return 0.01 * np.power(x, 2) + np.sin(x)


def main(args):
    N = args.num_data
    M = args.dim
    noise_var = args.noise_var
    data_range = (-1, 7)
    func = np.sin

    all_x = np.linspace(data_range[0] - 3, data_range[1] + 3, 100)

    xs, ys, features = create_dataset(N, M, data_range, func)

    fig, ax = plt.subplots()
    artists = []
    posterior_mean = np.zeros((1, M))
    posterior_var = np.identity(M)

    for i in range(1, N + 1):
        posterior_mean, posterior_var = posterior(features[i-1:i], ys[i-1:i],
                                                  posterior_mean, posterior_var,
                                                  noise_var)
        mean_star, min_star, max_star = predictive(all_x,
                                                   posterior_mean, posterior_var,
                                                   noise_var) 

        plt.ylim(-5, 20)
        plt.plot(all_x, func(all_x), 'r')
        b = plt.plot(xs[:i], ys[:i], 'ro')
        a = plt.plot(all_x, mean_star, 'b')
        c = plt.plot(all_x, min_star, 'c--')
        d = plt.plot(all_x, max_star, 'c--')
        # plt.fill_between(all_x, min_star, max_star, alpha=0.2)
        artists.append(a + b + c + d)

    ani = anime.ArtistAnimation(fig, artists, interval=500, repeat_delay=1000)
    posterior_mean, posterior_var = posterior(features, ys,
                                              np.zeros((1, M)), np.identity(M),
                                              noise_var)
    print(evidence(ys, (np.zeros((1, M)), np.identity(M)), (posterior_mean, posterior_var), noise_var))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian Logistic Regression')
    parser.add_argument('--num_data', type=int, default=10,
                        help='number of train data')
    parser.add_argument('--dim', type=int, default=5,
                        help='dimension')
    parser.add_argument('--noise_var', type=float, default=10,
                        help='noise variance')

    args = parser.parse_args()

    main(args)
