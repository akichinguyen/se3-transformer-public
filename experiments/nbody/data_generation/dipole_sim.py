import numpy as np


class DipoleSim(object):
    def __init__(self, n_particle=5, box_size=1., loc_std=1., vel_norm=0.5,
                 interaction_strength=1., noise_var=0., dim=3):
        self.n_particle = n_particle
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.int_strength = interaction_strength
        self.noise_var = noise_var

        self._charge_types = np.array([-1., 0., 1.])
        self._delta_T = 0.001
        self.dim = dim

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matrix
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:] - B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        clamp = np.zeros([loc.shape[1]])
        if self.box_size > 1e-6:
            assert (np.all(loc < self.box_size * 3))
            assert (np.all(loc > -self.box_size * 3))

            over = loc > self.box_size
            loc[over] = 2 * self.box_size - loc[over]
            assert (np.all(loc <= self.box_size))

            # assert(np.all(vel[over]>0))
            vel[over] = -np.abs(vel[over])

            under = loc < -self.box_size
            loc[under] = -2 * self.box_size - loc[under]
            # assert (np.all(vel[under] < 0))
            assert (np.all(loc >= -self.box_size))
            vel[under] = np.abs(vel[under])

            clamp[over[0, :]] = 1
            clamp[under[0, :]] = 1

        return loc, vel, clamp

    def simulation(self, T=10000, sample_freq=10, charge_prob=[1. / 2, 0, 1. / 2]):
        T_save = int(T / sample_freq - 1)
        counter = 0
        charges = np.random.choice(self._charge_types, size=(self.n_particle, 1),
                                   p=charge_prob)
        edges = charges.dot(charges.transpose())
        loc = np.zeros((T_save, self.dim, self.n_particle))
        vel = np.zeros((T_save, self.dim, self.n_particle))
        di_moment = np.zeros((T_save, self.dim, self.n_particle))
        ang_vel = np.zeros((T_save, self.dim, self.n_particle))
        ang_vel_parallel = np.zeros((T_save, self.dim, self.n_particle))
        ang_vel_perpen = np.zeros((T_save, self.dim, self.n_particle))
        clamp = np.zeros((T_save, self.n_particle))
        loc_next = np.random.randn(self.dim, self.n_particle) * self.loc_std
        vel_next = np.random.randn(self.dim, self.n_particle)
        di_moment[0] = np.random.uniform(0.41, 0.58, (self.dim, self.n_particle))
        ang_vel[0] = np.random.randn(self.dim, self.n_particle)
        loc[0, :, :], vel[0, :, :], clamp[0, :] = self._clamp(loc_next, vel_next)
        ang_vel_parallel[0] = ((np.dot(ang_vel[0].transpose(), di_moment[0].transpose())/(di_moment[0].transpose() ** 2).sum(axis=1))* di_moment[0].transpose()).transpose()
        ang_vel_perpen[0] = ang_vel[0] - ang_vel_parallel[0]
        dist_ij_vector = loc[0].transpose().reshape(1, self.n_particle, self.dim) - loc[0].transpose().reshape(self.n_particle, 1, self.dim)
        dist_ij_len = np.sqrt(self._l2(loc[0].transpose(), loc[0].transpose()))
        dist_ij_unit = dist_ij_vector/dist_ij_len




if __name__ == '__main__':
    print((np.arange(6).reshape((2,1,3)) - np.arange(6).reshape((1,2,3))).shape)









