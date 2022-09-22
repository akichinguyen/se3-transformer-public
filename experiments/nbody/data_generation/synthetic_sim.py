import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
from numpy.linalg import norm
from datetime import datetime, timedelta
import csv


def cross(left, right):
    x = ((left[1] * right[2]) - (left[2] * right[1]))
    y = ((left[2] * right[0]) - (left[0] * right[2]))
    z = ((left[0] * right[1]) - (left[1] * right[0]))
    return np.array([x, y, z])


def dot(left, right):
    assert (len(left) == len(right))
    res = sum([left[i] * right[i] for i in range(len(left))])
    return res


def rand_uvec_3d():
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.arccos(1 - 2 * np.random.uniform(0, 1))

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return [x, y, z]


class DipoleSim(object):
    def __init__(self, n_particle=5, box_size=1., loc_std=1., temp=0.1, mass=1.,
                 inert=0.1, mudip=1.0, max_force=100, delta_T=0.001,
                 noise_var=0., dim=3, sigma=1, epsilon_lj=1, epsilon_electic=1,
                 charges=None, name=None,
                 num_of_steps=1000000, type='dipole'):

        # np.random.seed(123456)

        self.n_particle = n_particle
        self.box_size = box_size
        self.temp = temp
        self.mass = mass
        self.inert = inert  # TARAS
        self.mudip = mudip  # TARAS
        self.sigma = sigma
        self.epsilon_lj = epsilon_lj
        self.epsilon_electic = epsilon_electic
        self.loc_std = loc_std
        self.vel_norm = (3 * self.temp / self.mass) ** 0.5  # TARAS
        self.ang_vel_norm = (2 * self.temp / self.inert) ** 0.5  # TARAS
        self.noise_var = noise_var
        self.max_force = max_force

        self._charge_types = np.array([-1., 0., 1.])
        self._delta_T = delta_T
        self.dim = dim
        self.num_of_steps = num_of_steps

        if charges is None:
            self.charges = np.random.choice(self._charge_types, size=(self.n_particle, 1),
                                            p=[0.5, 0, 0.5])
        else:
            self.charges = charges

        self.cur_state = {'loc': np.zeros((self.dim, self.n_particle)),
                          'vel': np.zeros((self.dim, self.n_particle)),
                          'ang_vel': np.zeros((self.dim, self.n_particle)),
                          'orientation': np.zeros((self.dim, self.n_particle)),
                          'di_moment': np.full((self.dim, self.n_particle), np.sqrt(1 / 3)),
                          'clamp': np.zeros([self.n_particle]),
                          }


        self.time_total = timedelta()

        self.K_functions = {'_K_ang': self._K_ang, '_K_lin': self._K_lin}
        self.U_functions = {'_U_LJ': self._U_LJ, '_U_dd': self._U_dd, '_U_dc': self._U_dc, '_U_cd': self._U_cd,
                            '_U_cc': self._U_cc}
        self.T_functions = {'_T_dd': self._T_dd, '_T_dc': self._T_dc}
        self.F_functions = {'_F_LJ': self._F_LJ, '_F_cc': self._F_cc, '_F_dc': self._F_dc, '_F_cd': self._F_cd,
                            '_F_dd': self._F_dd}

        if type == 'charged':
            self.K_functions = {'_K_lin': self._K_lin}
            self.U_functions = {'_U_LJ': self._U_LJ, '_U_cc': self._U_cc}
            self.T_functions = {}
            self.F_functions = {'_F_LJ': self._F_LJ, '_F_cc': self._F_cc}

        self.times = {el: timedelta() for el in
                      list(self.K_functions.keys()) + list(self.U_functions.keys()) +
                      list(self.T_functions.keys()) + list(self.F_functions.keys()) +
                      ['_clamp']
                      }

        if name is None:
            self.name = f"media/num_of_steps={self.num_of_steps}_n={self.n_particle}_boxsize={self.box_size}"
        else:
            self.name = name

    def get_time_analysis(self):
        for key in list(self.times.keys()):
            print(f"{key} time: {self.times[key]}")
        print(f"time_total = {self.time_total}\n")

    def _initialize_zero_state(self):
        self.cur_state['loc'] = self._initialize_start_loc()
        self.cur_state['vel'] = np.random.randn(self.dim, self.n_particle)
        self.cur_state['orientation'] = np.random.randn(self.dim, self.n_particle)
        #        self.cur_state['ang_vel'] = np.random.randn(self.dim, self.n_particle)

        v_cm = np.average(self.cur_state['vel'], axis=1).reshape(-1, 1)
        self.cur_state['vel'] = self.cur_state['vel'] - v_cm
        v_norm = np.sqrt(np.sum(self.cur_state['vel'] ** 2).sum(axis=0) / self.n_particle)
        self.cur_state['vel'] = self.cur_state['vel'] * self.vel_norm / v_norm

        # v_a_cm = np.average ( self.cur_state['ang_vel'], axis = 1 ).reshape(-1, 1)
        # self.cur_state['ang_vel'] = self.cur_state['ang_vel'] - v_a_cm
        # v_a_norm = np.sqrt(np.sum(self.cur_state['ang_vel'] ** 2).sum(axis=0)/self.n_particle)
        # self.cur_state['ang_vel'] = self.cur_state['ang_vel'] * self.ang_vel_norm / v_a_norm

        orientation_norm = np.sqrt((self.cur_state['orientation'] ** 2).sum(axis=0)).reshape(1, -1)
        self.cur_state['orientation'] = self.cur_state['orientation'] / orientation_norm

    def set_state(self, state):
        self.n_particle = len(state['loc'][0])
        self.dim = len(state['loc'])
        self.cur_state = state

    def set_charges(self, charges):
        self.charges = charges

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
            dist[i, i] = 0.0
        return dist

    def _clamp(self):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        start = datetime.now()
        fun_name = '_clamp'

        loc = self.cur_state['loc']
        vel = self.cur_state['vel']
        clamp = np.zeros([loc.shape[1]])
        if self.box_size > 1e-6:
            assert (np.all(loc < self.box_size * 3))
            assert (np.all(loc > -self.box_size * 3))

            over = loc > self.box_size
            loc[over] = 2 * self.box_size - loc[over]
            assert (np.all(loc <= self.box_size))

            vel[over] = -np.abs(vel[over])

            under = loc < -self.box_size
            loc[under] = -2 * self.box_size - loc[under]
            assert (np.all(loc >= -self.box_size))
            vel[under] = np.abs(vel[under])

            clamp[over[0, :]] = 1
            clamp[under[0, :]] = 1

        self.cur_state['loc'] = loc
        self.cur_state['vel'] = vel
        self.cur_state['clamp'] = clamp

        self.times[fun_name] += datetime.now() - start

    def _initialize_start_loc(self):
        max_try = 100

        start_loc = np.full((self.dim, self.n_particle), float(self.box_size * 2))
        for i in range(self.n_particle):
            try_num = 0
            cur_loc = np.random.uniform(-1 * self.box_size, self.box_size, self.dim)
            while min(abs(norm(cur_loc.T - start_loc.T, axis=1, ord=2))) < self.sigma:
                cur_loc = np.random.uniform(-1 * self.box_size, self.box_size, self.dim)
                try_num += 1
                assert (try_num < max_try)
            start_loc[:, i] = cur_loc
        return start_loc

    def _energy(self):
        with np.errstate(divide='ignore'):
            K_energies = {key: 0 for key in list(self.K_functions.keys())}
            U_energies = {key: 0 for key in list(self.U_functions.keys())}
            total_energy = {'E': [], 'U': [], 'K': []}

            for key, fun in list(self.K_functions.items()):
                start = datetime.now()
                K_energies[key] += fun()
                self.times[key] += datetime.now() - start

            for i in range(self.n_particle - 1):
                for j in range(i + 1, self.n_particle):
                    for key, fun in list(self.U_functions.items()):
                        start = datetime.now()
                        U_energies[key] += fun(i, j)
                        self.times[key] += datetime.now() - start

            total_energy['K'] = sum(list(K_energies.values()))
            total_energy['U'] = sum(list(U_energies.values()))
            total_energy['E'] = total_energy['U'] + total_energy['K']
            return (U_energies, K_energies, total_energy)

    def _K_lin(self):
        K_lin = 0.5 * (self.mass * self.cur_state['vel'] ** 2).sum()
        return K_lin

    def _K_ang(self):
        K_ang = 0.5 * (self.inert * self.cur_state['ang_vel'] ** 2).sum()
        return K_ang

    def _U_LJ(self, i, j):
        r_v = self.cur_state['loc'][:, i] - self.cur_state['loc'][:, j]
        r_cut = 2 ** (1 / 6) * self.sigma
        r = np.linalg.norm(r_v)
        if r > r_cut:
            return 0
        U_lj = 4 * self.epsilon_lj * (
                (np.power(self.sigma / r, 12) - np.power(self.sigma / r, 6)) -
                (np.power(self.sigma / r_cut, 12) - np.power(self.sigma / r_cut, 6))
        )
        return U_lj

    def _U_dd(self, i, j):
        u1_v = self.cur_state['orientation'][:, i]
        u2_v = self.cur_state['orientation'][:, j]
        r_v = self.cur_state['loc'][:, i] - self.cur_state['loc'][:, j]
        r = np.linalg.norm(r_v)
        mu1, mu2 = self.mudip, self.mudip
        U_dd = mu1 * mu2 / (np.power(r, 3)) * \
               (np.dot(u1_v, u2_v) - 3 * np.dot(r_v, u1_v) * np.dot(r_v, u2_v) / np.power(r, 2))

        return U_dd

    def _U_dc(self, i, j):
        u1_v = self.cur_state['orientation'][:, i]
        r_v = self.cur_state['loc'][:, i] - self.cur_state['loc'][:, j]
        r = np.linalg.norm(r_v)
        mu1 = self.mudip
        q2 = self.charges[j]
        U_dc = mu1 * q2 / (4 * np.pi * self.epsilon_electic * np.power(r, 3)) * np.dot(r_v, u1_v)
        return U_dc[0]

    def _U_cd(self, i, j):
        return -self._U_dc(j, i)

    def _U_cc(self, i, j):
        q1 = self.charges[i]
        q2 = self.charges[j]
        r_v = self.cur_state['loc'][:, i] - self.cur_state['loc'][:, j]
        r = np.linalg.norm(r_v)
        U_cc = q1 * q2 / r
        return U_cc[0]

    def _F_LJ(self, i, j):
        r_v = self.cur_state['loc'][:, i] - self.cur_state['loc'][:, j]
        r_cut = 2 ** (1. / 6) * self.sigma
        r = np.linalg.norm(r_v)
        if r > r_cut:
            return 0
        f_lj = 24 * self.epsilon_lj / r ** 2 * (2 * (self.sigma / r) ** 12 - (self.sigma / r) ** 6) * r_v
        return f_lj

    def _F_dd(self, i, j):
        u1_v = self.cur_state['orientation'][:, i]
        u2_v = self.cur_state['orientation'][:, j]
        r_v = self.cur_state['loc'][:, i] - self.cur_state['loc'][:, j]
        r = np.linalg.norm(r_v)
        mu1, mu2 = self.mudip, self.mudip
        f_dd = 3 * mu1 * mu2 / (np.power(r, 4)) * \
               (
                       (np.dot(u1_v, u2_v) - 5 / np.power(r, 2) * np.dot(r_v, u1_v) * np.dot(r_v, u2_v)) / r * r_v + \
                       (u1_v * np.dot(u2_v, r_v) + u2_v * np.dot(u1_v, r_v)) / r
               )
        return f_dd

    def _F_cc(self, i, j):
        q1 = self.charges[i]
        q2 = self.charges[j]
        r_v = self.cur_state['loc'][:, i] - self.cur_state['loc'][:, j]
        r = np.linalg.norm(r_v)
        f_cc = q1 * q2 / r ** 2 * (r_v / r)
        return f_cc

    def _F_dc(self, i, j):
        q2 = self.charges[j]
        u1 = self.cur_state['orientation'][:, i]
        r_v = self.cur_state['loc'][:, i] - self.cur_state['loc'][:, j]
        mu1 = self.mudip
        r = np.linalg.norm(r_v)
        f_dc = mu1 * q2 * u1 / r ** 3 - 3 * mu1 * q2 * dot(u1, r_v) * r_v / r ** 5
        return f_dc

    def _F_cd(self, i, j):
        return -self._F_dc(j, i)

    def _total_F(self):
        total_F = np.zeros((self.n_particle, self.n_particle, self.dim))
        for i in range(self.n_particle - 1):
            for j in range(i + 1, self.n_particle):
                cur_F = np.zeros(self.dim)
                for key, fun in list(self.F_functions.items()):
                    start = datetime.now()
                    cur_F += fun(i, j)
                    self.times[key] += datetime.now() - start

                total_F[i, j] = -cur_F
                total_F[j, i] = cur_F

        total_F = total_F.sum(axis=0)
        for j in range(self.n_particle):
            if np.power(total_F[j], 2).sum() > self.max_force ** 2:
                print(f"normed F from {np.power(total_F[j], 2).sum() ** 0.5}")
                total_F[j] = np.sqrt(self.max_force ** 2 / np.power(total_F[j], 2).sum()) * total_F[j]
                print(total_F[j])
        return total_F.T

    def _T_dd(self, i, j):
        u1_v = self.cur_state['orientation'][:, i]
        u2_v = self.cur_state['orientation'][:, j]
        r_v = self.cur_state['loc'][:, i] - self.cur_state['loc'][:, j]
        r = np.linalg.norm(r_v)
        mu1, mu2 = self.mudip, self.mudip
        T_dd = -1 * mu1 * mu2 / np.power(r, 3) * (u2_v - 3 * np.dot(u2_v, r_v) / np.power(r, 2) * r_v)
        return T_dd

    def _T_dc(self, i, j):
        q2 = self.charges[j]
        u1 = self.cur_state['orientation'][:, i]
        r_v = self.cur_state['loc'][:, i] - self.cur_state['loc'][:, j]
        mu1 = self.mudip
        r = np.linalg.norm(r_v)
        t_dc = mu1 * q2 * r_v / r ** 3

        return t_dc

    def _total_T(self):
        total_T = np.zeros((self.n_particle, self.dim))
        for i in range(self.n_particle):
            cur_T = 0
            for j in range(self.n_particle):
                if i == j:
                    continue
                for key, fun in list(self.T_functions.items()):
                    start = datetime.now()
                    cur_T += fun(i, j)
                    self.times[key] += datetime.now() - start

            total_T[i] = cur_T
        for j in range(self.n_particle):
            if np.power(total_T[j], 2).sum() > self.max_force ** 2:
                print(f"normed T from {np.power(total_T[j], 2).sum() ** 0.5}")
                total_T[j] = np.sqrt(self.max_force ** 2 / np.power(total_T[j], 2).sum()) * total_T[j]
                print(total_T[j])
        return total_T.T

    def write_to_lammps(self, fp, i):
        fp.write("{:<s}\n".format("ITEM: TIMESTEP"))
        fp.write("{:<d}\n".format(i))
        fp.write("{:<s}\n".format("ITEM: NUMBER OF ATOMS"))
        fp.write("{:<d}\n".format(self.n_particle))
        fp.write("{:<s}\n".format("ITEM: BOX BOUNDS ff ff ff"))
        fp.write("{0:<15.8f}{1:<15.8f}\n".format(-self.box_size, self.box_size))
        fp.write("{0:<15.8f}{1:<15.8f}\n".format(-self.box_size, self.box_size))
        fp.write("{0:<15.8f}{1:<15.8f}\n".format(-self.box_size, self.box_size))
        fp.write("{:<s}\n".format("ITEM: ATOMS id type q x y z mux muy muz"))

        for j in range(self.n_particle):
            if self.charges[j, 0] >= 0:
                atype = 1
            else:
                atype = 2

            fp.write("{0:7d}{1:>4d}{2:>10.4f}{3:>15.6f}{4:>15.6f}{5:>15.6f}{6:>15.6f}{7:>15.6f}{8:>15.6f}\n"
                     .format(j + 1, atype, self.charges[j, 0]
                             , self.cur_state['loc'][0, j], self.cur_state['loc'][1, j], self.cur_state['loc'][2, j]
                             , self.cur_state['orientation'][0, j], self.cur_state['orientation'][1, j],
                             self.cur_state['orientation'][2, j]))

        fp.flush()

    def write_restart(self, step, fname):
        fp = open(fname, "w")
        fp.write("LAMMPS data file. Step {:<d}\n".format(step))
        fp.write("\n")
        fp.write("{:<d} atoms\n".format(self.n_particle))
        fp.write("{:<d} atom types\n".format(2))
        fp.write("\n")
        fp.write("{0:<15.8f} {1:<15.8f} xlo xhi\n".format(-self.box_size, self.box_size))
        fp.write("{0:<15.8f} {1:<15.8f} ylo yhi\n".format(-self.box_size, self.box_size))
        fp.write("{0:<15.8f} {1:<15.8f} zlo zhi\n".format(-self.box_size, self.box_size))
        fp.write("\n")
        fp.write("Masses\n")
        fp.write("\n")
        fp.write("{0:<d} {1:<15.8f}\n".format(1, self.mass))
        fp.write("{0:<d} {1:<15.8f}\n".format(2, self.mass))
        fp.write("\n")
        fp.write("Atoms # hybrid\n")
        fp.write("\n")
        for j in range(self.n_particle):
            if self.charges[j, 0] >= 0:
                atype = 1
            else:
                atype = 2

            fp.write(
                "{0:7d}{1:>4d}{2:>18.12f}{3:>18.12f}{4:>18.12f}{5:>12.8f}{6:>12.8f}{7:>12.8f}{8:>18.12f}{9:>18.12f}{10:>18.12f}{11:>4d}{12:>4d}{13:>4d}\n"
                .format(j + 1, atype, self.cur_state['loc'][0, j], self.cur_state['loc'][1, j],
                        self.cur_state['loc'][2, j]
                        , self.sigma, self.mass * 6 / (np.pi * self.sigma ** 3), self.charges[j, 0]
                        , self.cur_state['orientation'][0, j], self.cur_state['orientation'][1, j],
                        self.cur_state['orientation'][2, j]
                        , 0, 0, 0))
        fp.write("\n")

        fp.write("Velocities\n")
        fp.write("\n")
        angvel = np.cross(self.cur_state['orientation'], self.cur_state['ang_vel'], axisa=0, axisb=0).transpose(1, 0)
        for j in range(self.n_particle):
            fp.write("{0:7d}{1:>18.12f}{2:>18.12f}{3:>18.12f}{4:>18.12f}{5:>18.12f}g{6:>18.12f}\n"
                     .format(j + 1, self.cur_state['vel'][0, j], self.cur_state['vel'][1, j],
                             self.cur_state['vel'][2, j]
                             , angvel[0, j], angvel[1, j], angvel[2, j]))

        fp.flush()
        fp.close()

    def read_restart(self, fname):
        fp = open(fname, "r")
        fp.readline()
        fp.readline()
        arr = fp.readline().split()
        self.n_particle = int(arr[0])
        fp.readline()
        fp.readline()
        arr = fp.readline().split()
        self.box_size = np.absolute(float(arr[0]))
        for i in range(10):
            fp.readline()
        for j in range(self.n_particle):
            arr = fp.readline().split()
            self.cur_state['loc'][0, j] = arr[2]
            self.cur_state['loc'][1, j] = arr[3]
            self.cur_state['loc'][2, j] = arr[4]
            self.charges[j] = arr[7]
            self.cur_state['orientation'][0, j] = arr[8]
            self.cur_state['orientation'][1, j] = arr[9]
            self.cur_state['orientation'][2, j] = arr[10]

        for i in range(3):
            fp.readline()

        for j in range(self.n_particle):
            arr = fp.readline().split()
            self.cur_state['vel'][0, j] = arr[1]
            self.cur_state['vel'][1, j] = arr[2]
            self.cur_state['vel'][2, j] = arr[3]
            self.cur_state['ang_vel'][0, j] = arr[4]
            self.cur_state['ang_vel'][1, j] = arr[5]
            self.cur_state['ang_vel'][2, j] = arr[6]

        self.cur_state['ang_vel'] = np.cross(self.cur_state['ang_vel'], self.cur_state['orientation'], axisa=0,
                                             axisb=0).transpose(1, 0)
        fp.close()


    def simulation(self, num_of_steps=0, sample_freq=100, sim_num=1, read_state=False,
                   restart_start_fname="restart_start.data", restart_current_fname="restart_current.data",
                   restart_finish_fname="restart_finish.data", restart_init_fname="restart_init.data"):
        if not num_of_steps:
            num_of_steps = self.num_of_steps
        start = datetime.now()
        steps_save = int(num_of_steps / sample_freq)
        counter = 0

        energy = {key: [] for key in ['E', 'U', 'K'] + list(self.U_functions.keys()) + list(self.K_functions.keys())}
        loc = np.zeros((steps_save, self.dim, self.n_particle))
        vel = np.zeros((steps_save, self.dim, self.n_particle))
        clamp = np.zeros((steps_save, self.n_particle))
        orientation = np.zeros((steps_save, self.dim, self.n_particle))
        di_moment = np.full((steps_save, self.dim, self.n_particle), np.sqrt(1 / 3))
        ang_vel = np.zeros((steps_save, self.dim, self.n_particle))
        self._initialize_zero_state()

        if read_state:
            try:
                self.read_restart(restart_init_fname)
            except:
                print(f"no such file {restart_init_fname}")

        i = 0
        self.write_restart(i, restart_start_fname)

        ang_vel[0] = self.cur_state['ang_vel']
        self._clamp()
        loc[0, :, :], vel[0, :, :] = self.cur_state['loc'], self.cur_state['vel']

        torques = self._total_T()
        forces = self._total_F()

        a_next = forces / self.mass
        alpha_next = (torques - (torques * self.cur_state['orientation']).sum(axis=0) * self.cur_state[
            'orientation']) / self.inert \
                     - self.cur_state['orientation'] * (self.cur_state['ang_vel'] ** 2).sum(axis=0)  # TARAS

        fname = f"logs/traj_{sim_num}.lammptrj"
        fp = open(fname, "w")
        energies_fname = f"logs/energies_{sim_num}.csv"
        energies_fp = open(energies_fname, "w")
        writer = csv.writer(energies_fp, delimiter="\t")
        writer.writerow(energy.keys())
        energies_fp.close()
        energies_fp = open(energies_fname, "a")
        energies_writer = csv.DictWriter(energies_fp, fieldnames=energy.keys(), delimiter="\t")

        for i in range(num_of_steps):
            a_old = a_next
            alpha_old = alpha_next

            self.cur_state['loc'] = self.cur_state['loc'] + self.cur_state[
                'vel'] * self._delta_T + a_old * self._delta_T ** 2 / 2

            self.cur_state['orientation'] = self.cur_state['orientation'] + self.cur_state[
                'ang_vel'] * self._delta_T + alpha_old * self._delta_T ** 2 / 2  # TARAS

            for j in range(self.n_particle):
                self.cur_state['orientation'][:, j] /= np.linalg.norm(self.cur_state['orientation'][:, j])

            forces = self._total_F()
            torques = self._total_T()

            a_next = forces / self.mass

            self.cur_state['vel'] = self.cur_state['vel'] + (a_old + a_next) / 2 * self._delta_T

            self.cur_state['ang_vel'] = self.cur_state['ang_vel'] + alpha_old / 2 * self._delta_T  # TARAS
            self._clamp()

            alpha_next = (torques - (torques * self.cur_state['orientation']).sum(axis=0) * self.cur_state[
                'orientation']) / self.inert \
                         - self.cur_state['orientation'] * (
                                     (self.cur_state['ang_vel'] + alpha_old / 2 * self._delta_T) ** 2).sum(
                axis=0)  # TARAS

            self.cur_state['ang_vel'] = self.cur_state['ang_vel'] + alpha_next / 2 * self._delta_T  # TARAS

            k_lin = self._K_lin()
            k_ang = self._K_ang()
            temp_lin = k_lin / 3 * 2 / (self.n_particle)
            temp_ang = k_ang / (self.n_particle)

            if (i % 100 == 0) and (i < self.num_of_steps):
                vel_lin = np.sqrt(self.temp / temp_lin)
                vel_ang = np.sqrt(self.temp / temp_ang)
                self.cur_state['vel'] *= vel_lin  # rescale velocities thermostat
                self.cur_state['ang_vel'] *= vel_ang  # rescale angular velocities thermostat

            if i % sample_freq == 0:
                self.write_to_lammps(fp, i)

                # self.write_restart(i, restart_current_fname)

                cur_energy = self._energy()
                cur_energy_dict = {}
                for key, value in list(cur_energy[2].items()) + list(cur_energy[0].items()) + list(
                        cur_energy[1].items()):
                    energy[key].append(value)
                    cur_energy_dict[key] = value
                energies_writer.writerow(cur_energy_dict)
                loc[counter, :, :], vel[counter, :, :] = self.cur_state['loc'], self.cur_state['vel']
                orientation[counter, :, :], ang_vel[counter, :, :] = self.cur_state['orientation'], self.cur_state[
                    'ang_vel']
                clamp[counter, :] = self.cur_state['clamp']
                counter += 1

        self.write_restart(i, restart_finish_fname)
        energies_fp.close()

        self.time_total = datetime.now() - start
        # self.get_time_analysis()

        return loc, vel, energy, orientation, ang_vel, self.charges, clamp


if __name__ == '__main__':
    box_size = 4
    num_of_steps = 1000000
    n_particle = 4

    name = f"media/num_of_steps={num_of_steps}_n={n_particle}_boxsize={box_size}"
    print(f"run for {name}")

    sim = DipoleSim(n_particle=n_particle, box_size=box_size,
                    num_of_steps=num_of_steps, type='dipole', dim=3,
                    temp=0.348, delta_T=0.001,
                    mass=1., inert=0.1,
                    mudip=1.0, max_force=100,
                    noise_var=0.,
                    loc_std=1., sigma=1,
                    epsilon_lj=1, epsilon_electic=1,
                    charges=None, name=name
                    )

    sim.simulation(sample_freq=100, read_state=False,
                   restart_start_fname="restart_start.data", restart_current_fname="restart_current.data",
                   restart_finish_fname="restart_finish.data", restart_init_fname="restart_init.data")

