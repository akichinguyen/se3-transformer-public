import numpy as np
import math
import matplotlib.pyplot as plt

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
        for i in range(A.shape[0]):
            dist[i,i] = 0.0
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
        loc_next = np.random.uniform(-1., 1., (self.dim, self.n_particle)) * self.loc_std
        vel_next = np.random.randn(self.dim, self.n_particle)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        di_moment[0] = np.random.uniform(0.41, 0.58, (self.dim, self.n_particle))
        ang_vel[0] = np.random.randn(self.dim, self.n_particle)
        print(loc_next)
        loc[0, :, :], vel[0, :, :], clamp[0, :] = self._clamp(loc_next, vel_next)
        loc_next = loc[0]
        vel_next = vel[0]
        ang_vel_parallel[0] = (((ang_vel[0].transpose()* di_moment[0].transpose()).sum(axis=1).reshape(self.n_particle, 1) / ((
                    di_moment[0].transpose() ** 2).sum(axis=1)).reshape(self.n_particle, 1)) * di_moment[0].transpose()).transpose()
        ang_vel_perpen[0] = ang_vel[0] - ang_vel_parallel[0]
        dist_ij_vector = loc[0].transpose().reshape(1, self.n_particle, self.dim) - loc[0].transpose().reshape(
            self.n_particle, 1, self.dim)
        dist_ij_len = np.sqrt(self._l2(loc[0].transpose(), loc[0].transpose())).reshape(self.n_particle, self.n_particle, 1) + 0.00001
        dist_ij_unit = dist_ij_vector / dist_ij_len
        e_firstpart = charges.reshape(self.n_particle, 1, 1) * dist_ij_unit/ (dist_ij_len ** 2)
        pjrji = np.sum(di_moment[0].transpose().reshape(self.n_particle,1,self.dim) * dist_ij_unit, axis=-1).reshape(self.n_particle, self.n_particle, 1)
        e_secondpart = (3*pjrji*dist_ij_unit - di_moment[0].transpose().reshape(self.n_particle,1,self.dim))/(dist_ij_len ** 3)
        e = np.sum(e_firstpart + e_secondpart, axis=0)
        qjpi = charges.reshape(self.n_particle, 1, 1) * di_moment[0].transpose()
        qjrji = charges.reshape(self.n_particle, 1, 1) * dist_ij_unit
        pirji = np.sum(di_moment[0].transpose().reshape(1,self.n_particle,self.dim) * dist_ij_unit, axis=-1).reshape(self.n_particle, self.n_particle, 1)
        pe_firstpart = (qjpi - 3*qjrji*pirji)/(dist_ij_len ** 3)
        pipjrij = 3*di_moment[0].transpose()*pjrji
        pipj = np.dot(di_moment[0].transpose(), di_moment[0]).transpose().reshape(self.n_particle, self.n_particle, 1)
        rijpipj = 3*dist_ij_unit*(pipj - 5*pirji*pjrji)
        pjpirij = 7*di_moment[0].transpose().reshape(self.n_particle,1,self.dim) * pirji
        pe_secondpart = (pipjrij + rijpipj - pjpirij)/(dist_ij_len**4)
        pe = np.sum(pe_firstpart + pe_secondpart, axis=0)
        a = (charges*e + pe)
        alpha = np.cross(di_moment[0].transpose(), e) - np.cross(ang_vel_parallel[0].transpose(), ang_vel_perpen[0].transpose()) - np.cross(ang_vel_perpen[0].transpose(), ang_vel_parallel[0].transpose())
        dpidt = np.cross(alpha, di_moment[0].transpose()) + np.cross(ang_vel[0].transpose(), np.cross(ang_vel[0].transpose(), di_moment[0].transpose()))
        di_moment_next = di_moment[0].transpose()
        ang_vel_next = ang_vel[0].transpose()
        for i in range(1, T):
            loc_next += self._delta_T * vel_next + self._delta_T*self._delta_T*a.transpose()/2.0
            loc_next, vel_next, _ = self._clamp(loc_next, vel_next)
            dist_ij_vector = loc_next.transpose().reshape(1, self.n_particle, self.dim) - loc_next.transpose().reshape(
                self.n_particle, 1, self.dim)
            dist_ij_len = np.sqrt(self._l2(loc_next.transpose(), loc_next.transpose())).reshape(self.n_particle, self.n_particle, 1) + 0.00001
            dist_ij_unit = dist_ij_vector / dist_ij_len
            di_moment_next = di_moment_next + np.cross(ang_vel_next, self._delta_T*di_moment_next) + self._delta_T*self._delta_T*dpidt/2.0
            e_firstpart = charges.reshape(self.n_particle, 1, 1) * dist_ij_unit / (dist_ij_len ** 2)
            pjrji = np.sum(di_moment_next.reshape(self.n_particle, 1, self.dim) * dist_ij_unit,
                           axis=-1).reshape(self.n_particle, self.n_particle, 1)
            e_secondpart = (3 * pjrji * dist_ij_unit - di_moment_next.reshape(self.n_particle, 1,
                                                                                        self.dim)) / (dist_ij_len ** 3)
            e = np.sum(e_firstpart + e_secondpart, axis=0)
            qjpi = charges.reshape(self.n_particle, 1, 1) * di_moment_next
            qjrji = charges.reshape(self.n_particle, 1, 1) * dist_ij_unit
            pirji = np.sum(di_moment_next.reshape(1, self.n_particle, self.dim) * dist_ij_unit,
                           axis=-1).reshape(self.n_particle, self.n_particle, 1)
            pe_firstpart = (qjpi - 3 * qjrji * pirji) / (dist_ij_len ** 3)
            pipjrij = 3 * di_moment_next * pjrji
            pipj = np.dot(di_moment_next, di_moment_next.transpose()).transpose().reshape(self.n_particle, self.n_particle,1)
            rijpipj = 3 * dist_ij_unit * (pipj - 5 * pirji * pjrji)
            pjpirij = 7 * di_moment_next.reshape(self.n_particle, 1, self.dim) * pirji
            pe_secondpart = (pipjrij + rijpipj - pjpirij) / (dist_ij_len ** 4)
            pe = np.sum(pe_firstpart + pe_secondpart, axis=0)
            a_next = (charges*e + pe)
            vel_next = vel_next + 0.5* self._delta_T* (a.transpose() + a_next.transpose())
            loc_next, vel_next, _ = self._clamp(loc_next, vel_next)
            di_moment_next_len = (di_moment_next**2).sum(axis=-1).reshape(self.n_particle, 1)
            ang_vel_parallel_next = ((di_moment_next * (ang_vel_next + 0.5*self._delta_T*alpha)).sum(axis=-1).reshape(self.n_particle, 1)/di_moment_next_len)*di_moment_next
            ang_vel_perpen_next = ang_vel_next + 0.5*self._delta_T*alpha - ang_vel_parallel_next + 0.5*self._delta_T*np.cross(di_moment_next, e)
            alpha = np.cross(di_moment_next, e) - np.cross(ang_vel_parallel_next, ang_vel_perpen_next) - np.cross(ang_vel_perpen_next, ang_vel_parallel_next)
            ang_vel_next = ang_vel_parallel_next + ang_vel_perpen_next
            dpidt = np.cross(alpha, di_moment_next) + np.cross(ang_vel_next, np.cross(ang_vel_next, di_moment_next))
            a = a_next
            if i % sample_freq == 0:
                loc[counter, :, :], vel[counter, :, :]= loc_next, vel_next
                counter += 1
        return loc, vel



if __name__ == '__main__':
    sim = DipoleSim()
    loc, vel = sim.simulation(T=5000, sample_freq=100)
    plt.figure()
    axes = plt.gca(projection='3d')
    axes.set_xlim([-1., 1.])
    axes.set_ylim([-1., 1.])
    axes.set_zlim([-1., 1.])
    for i in range(loc.shape[-1]):
        plt.plot(loc[:, 0, i], loc[:, 1, i], loc[:, 2, i])
    plt.show()