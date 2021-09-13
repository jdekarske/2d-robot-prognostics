#!/usr/bin/env python

from src.controller import build_optimal_controller
from src.robotmodel import *
import json
import sys
import time
import copy
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sm
print("---")
print("IGNORE THIS WARNING")


print("---")
print()

# Constants #
#############

SIMULATION_TIME_STEP = 0.005

# https://github.com/ros-industrial/universal_robot/blob/kinetic-devel/ur_e_description/urdf/ur5e.urdf.xacro
# model constants are used for controller design
# TODO config file this
upper_arm_radius_constant = 0.054
upper_arm_mass_constant = 8.393
fore_arm_length_constant = 0.392
fore_arm_radius_constant = 0.060
fore_arm_mass_constant = 2.275
upper_arm_length_constant = 0.425
upper_arm_com_length_constant = 0.2125
end_effector_mass_constant = 0
fore_arm_com_length_constant = 0.11993

model_constants = {
    upper_arm_length: upper_arm_length_constant,  # [m]
    upper_arm_com_length: upper_arm_length_constant,  # [m]
    upper_arm_mass: upper_arm_mass_constant,  # [kg]
    # [kg*m^2]
    upper_arm_inertia: 0.5 * upper_arm_mass_constant * upper_arm_radius_constant ** 2,
    fore_arm_length: fore_arm_length_constant,  # [m]
    end_effector_mass: end_effector_mass_constant,  # [kg]
    fore_arm_com_length: fore_arm_com_length_constant,  # [m]
    fore_arm_mass: fore_arm_mass_constant,  # [kg]
    # [kg*m^2]
    fore_arm_inertia: 0.5 * fore_arm_mass_constant * fore_arm_radius_constant ** 2,
    shoulder_degradation: 0,  # not actually a rate, viscous
    elbow_degradation: 0,
    g: 0  # 9.806
}

# Initialize systems #
######################

# use the model system to design the controller
model_sys = System(kane)
model_sys.constants = model_constants
model_sys.generate_ode_function()

# Change the constants for real-world parameters
real_constants = copy.deepcopy(model_constants)

real_constants.update({
    shoulder_degradation: 1,
    elbow_degradation: 1,
    g: 9.806
})

real_sys = System(kane)
real_sys.constants = real_constants
real_sys.generate_ode_function()


# Build controller #
####################

equilibrium_point = np.zeros(len(coordinates + speeds))
equilibrium_dict = dict(zip(coordinates + speeds, equilibrium_point))

K, Nbar = build_optimal_controller(kane, model_sys.constants, equilibrium_dict)

# also design a gravity compensator
fore_arm_grav_compensator = sm.lambdify(
    [theta1, theta2], fore_arm_grav_compensation_expr.subs(real_sys.constants))
upper_arm_grav_compensator = sm.lambdify(
    [theta1, theta2], upper_arm_grav_compensation_expr.subs(real_sys.constants))


def full_simulation(chunks, chunk_time, desired_positions, shoulder_degradation_val, elbow_degradation_val, shoulder_degradation_SD, elbow_degradation_SD):
    chunk_times = np.arange(0.0, chunk_time, SIMULATION_TIME_STEP)

    all_positions = np.tile(desired_positions, (1, chunks))

    chunk_initial_conditions = np.vstack(
        (desired_positions[:, 0].reshape(2, 1), [[0], [0]]))  # and zero velocities

    all_t = np.empty((chunk_times.size * chunks, 1))  # times
    all_y = np.empty((chunk_times.size * chunks, 4))  # 4 states
    all_command = np.empty((chunk_times.size * chunks, 2))  # 2 inputs

    for i in range(chunks):
        # desired positions input must have chunks+1 number of positions where the first is the initial position
        r = all_positions[:, i+1]
        chunk_times = np.arange(
            chunk_time * i, chunk_time * (i+1), SIMULATION_TIME_STEP)
        output_indices = np.array([0, chunk_times.size]) + chunk_times.size * i

        # update model parameters
        # Increase degradation each chunk, must be greater than 0!!!
        real_constants[elbow_degradation] = np.max([0.0, (elbow_degradation_val + np.random.normal(0, elbow_degradation_SD))])
        real_constants[shoulder_degradation] = np.max([0.0, (shoulder_degradation_val + np.random.normal(0, shoulder_degradation_SD))])

        # make a new system (because it is required when changing constants)
        real_sys = System(kane)
        real_sys.constants = real_constants
        real_sys.generate_ode_function()

        def controller(x, t):
            # LQR Controller
            u = np.asarray(np.add(-np.matmul(K, x).reshape(2, 1),
                                  np.matmul(Nbar, r).reshape(2, 1))).flatten()
            # Feed Forward Gravity Compensation
            u[0] -= upper_arm_grav_compensator(x[0], x[1])
            u[1] -= fore_arm_grav_compensator(x[0], x[1])
            return u

        sol = solve_ivp(lambda x, t, r, p: real_sys.evaluate_ode_function(t, x, r, p),
                        (chunk_times[0], chunk_times[-1]), chunk_initial_conditions.flatten(), method='LSODA', t_eval=chunk_times, args=(controller, real_sys.constants))

        all_t[output_indices[0]:output_indices[1],
              :] = sol.t.reshape(sol.t.size, 1)
        all_y[output_indices[0]:output_indices[1], :] = np.transpose(sol.y)

        commands = np.apply_along_axis(
            lambda a: controller(a, 0), 1, np.transpose(sol.y))
        all_command[output_indices[0]:output_indices[1], :] = commands

        chunk_initial_conditions = sol.y[:, -1]

    return (all_t, all_y, all_command)

# Config File #
###############


def print_help():
    print("Usage: python {} config_file.json".format(sys.argv[0]))
    return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide path to file")
        print_help()
        exit()
    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print_help()
        exit()

    config_file_path = sys.argv[1]
    print("using config file: {}".format(config_file_path))


with open(config_file_path) as f:
    config = json.load(f)

# Run the Simulation #
######################

print("starting simulation, if there is no output in 10 seconds, ctrl+c and restart script")

for elbow_deg in config['elbow_degradations']:
    for shoulder_deg in config['shoulder_degradations']:
        t = time.time() # for profiling
        fname = "shoulder{:d}-elbow{:d}.csv".format(int(shoulder_deg*100), int(elbow_deg*100))
        with open(fname,"a") as write_file:
            for i in range(config['iterations']):
                all_t, all_y, all_command = full_simulation(config['chunks'], config['chunk_time'], np.deg2rad(
                    np.array(config['desired_positions'])), shoulder_deg, elbow_deg, config['shoulder_degradations_SD'], config['elbow_degradations_SD'])

                # TODO we'll add observation noise if we need it
                # signal_range = np.min(np.ptp(all_y[:, :2], axis=0))
                # random_addition = np.random.normal(0, signal_range/1000, all_y.shape)
                # noise_y = all_y + random_addition

                np.savetxt(write_file, np.hstack((all_t, all_y, all_command)), fmt='%2.5f')

        elapsed = time.time() - t
        print(fname + " completed " + str(config['iterations']) + " iterations in " + str(elapsed) + " seconds.")

with open("header.txt", "w") as text_file:
    text_file.write("time\ttheta1\ttheta2\tomega1\tomega2\ttorque1\ttorque2")
