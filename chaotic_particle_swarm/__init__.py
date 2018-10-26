import numpy as np


class ChaosSwarm(object):
    def __init__(self, function, search_space, num_particles=3, w=0.9,
                 max_error=0.005):
        self.w_value = w
        self.num_particles = num_particles
        self.max_error = max_error
        self.search_space = search_space
        self.dimensions = len(search_space)
        self.function = function
        self.V, self.fitness, self.local_best = [], [], []
        self.c_1 = self.c_2 = 2
        self.w_min = 0.2
        self.w_max = 1.2

        # Initialize the positions of  the particles
        self.X = self.generate_particles(num_particles)

        # Initialize particles
        self.V, self.w = self.init_particles(self.X)
        # Run for first time
        current = self.evaluate(self.X)
        # Set the global best particle
        self.global_best = current.index(min(current))
        # Set the local best score of the particles
        self.local_best = list(zip(current, self.X))

    def generate_particles(self, num_particles):
        return [np.array([np.random.uniform(low, high)
                          for _, (low, high) in self.search_space])
                for _ in range(num_particles)]

    def evaluate(self, particles):
        return [self.function(
            **{k[0]: v for k, v in zip(self.search_space, particle)})
                for particle in particles]

    def init_particles(self, particles):
        num_particles = len(particles)
        V = np.random.uniform(0, 1, (num_particles, len(self.search_space)))
        w = [self.w_value for _ in range(num_particles)]
        return V, w

    def velocity(self, velocities):
        '''
        Implementation of equation 2 in the article.
        '''

        return [self.w[p_i] * p_v + self.c_1 * np.random.uniform() *
                self.local_diff(p_i) +
                self.c_2 * np.random.uniform() * self.global_diff(p_i)
                for p_i, p_v in enumerate(velocities)]

    def local_diff(self, p_i):
        return self.local_best[p_i][1] - self.X[p_i]

    def global_diff(self, p_i):
        return self.local_best[self.global_best][1] - self.X[p_i]

    def location(self, locations, velocities):
        new_locations = []
        for x, v in zip(locations, velocities):
            new_x = x + v
            for i in range(self.dimensions):
                search_var = self.search_space[i][1]
                dim_min, dim_max = search_var[0], search_var[1]
                if new_x[i] > dim_max:
                    new_x[i] = dim_max
                elif new_x[i] < dim_min:
                    new_x[i] = dim_min
            new_locations.append(new_x)
        return new_locations

    def get_local_best(self, current):
        """
        Set the local best positions for the particles based on the scores
        achieved in the last evaluation.
        """
        return [(cur_score, self.X[p_i])
                if cur_score < self.local_best[p_i][0]
                else self.local_best[p_i]
                for p_i, cur_score in enumerate(current)]

    def aiwf(self, scores):
        """
        Implements the Adaptive inertia weight factor.
        """
        mean_score = np.mean(scores)
        min_score = np.min(scores)

        return [self.new_w(mean_score, score, min_score) for score in scores]

    def new_w(self, mean_score, score, min_score):
        if score <= mean_score and min_score < mean_score:
            return self.w_min + (((self.w_max - self.w_min) * (score - min_score)) /
                    (mean_score - min_score))
        else:
            return self.w_max

    def chaotic_local_search(self, particle, score):
        # Apply the chaotic function to the particle.
        mins = [var[0] for name, var in self.search_space]
        maxs = [var[1] for name, var in self.search_space]

        for i in range(5):
            chaotic_particle = self.logistic_function(particle[:], mins, maxs)
            new_score = self.evaluate([chaotic_particle])
            if new_score < score:
                print('better one found', particle)
                return new_score, chaotic_particle
        return score, particle

    def logistic_function(self, particle, mins, maxs):
        cxs = self.part_to_cx(particle, mins, maxs)
        logistic = [4 * cx * (1 - cx) for cx in cxs]
        return self.cx_to_part(logistic, mins, maxs)

    def part_to_cx(self, particle, lows, highs):
        return [(x - low) / (high - low)
                for x, (low, high) in zip(particle, zip(lows, highs))]

    def cx_to_part(self, cxs, lows, highs):
        return [low + cx * (high - low)
                for cx, (low, high) in zip(cxs, zip(lows, highs))]

    def pso(self, iterations=50):
        error, i, similar = 1, 0, False
        while error > self.max_error or i < iterations and not \
                self.same_particles():
            # Recalculate the score
            current = self.evaluate(self.X)

            # Update the local best particle based on the current score
            self.local_best = self.get_local_best(current)

            # Set the weight factor based on the scores
            self.w = self.aiwf(current)

            # Set new velocity and positions of the particles
            self.V = self.velocity(self.V)
            self.X = self.location(self.X, self.V)

            # Set the global best particle
            best_index = current.index(min(current))
            self.global_best = best_index
            error = self.local_best[self.global_best][0]
            i += 1


    def same_particles(self):
        if len(set(tuple(p) for p in self.X)) == 1:
            return True
        return False

    def decrease_search_space(self, particle, r=0.25):
        mins = [var[0] for name, var in self.search_space]
        maxs = [var[1] for name, var in self.search_space]

        xmins = [max(mins[i], particle[i] - (r * (maxs[i] - mins[i]))) for i in
                 range(self.dimensions)]
        xmaxs = [min(maxs[i], particle[i] + (r * (maxs[i] - mins[i]))) for i in
                 range(self.dimensions)]
        return [(k[0], (xmins[i], xmaxs[i])) for i, k in
                enumerate(self.search_space)]

    def new_generation(self, old, amount):
        new_particles = self.generate_particles(amount)
        self.X = [*old, *new_particles]
        new_V, new_w = self.init_particles(new_particles)
        self.V = [*self.V[:len(old)], *new_V]
        self.w = [*self.w[:len(old)], *new_w]

    def run(self):
        error, i = 1, 0
        while error > self.max_error or i < 100:
            # Run the Particle Swarm.
            self.pso()

            # top is a list with tuples (score, particle)
            top = sorted(self.local_best, key=lambda x: x[0])[:int(self.num_particles / 5)]

            # Chaotic search
            top[0] = self.chaotic_local_search(top[0][1], top[0][0])

            self.local_best[self.global_best] = top[0]
            # Decrease search space
            self.search_space = self.decrease_search_space(top[0][1])

            # Construct the new particles
            num_new = int((4 * self.num_particles) / 5)
            self.new_generation([p[1] for p in top], num_new)
            error = self.local_best[self.global_best][0]
            i += 1
        return self.X[self.global_best]


if __name__ == '__main__':
    function = lambda x: x ** 2
    pso = ChaosSwarm(function, [('x', (-1000, 1000))], 30, max_error=0.001)
    result = pso.run()
    print(function(*result), result)
