#!/usr/bin/env python
from sympy import symbols, simplify, pi
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, KanesMethod
from sympy.physics.vector import init_vprinting
from pydy.system import System
import numpy as np


# # Reference Frames
inertial_frame = ReferenceFrame('I')
upper_arm_frame = ReferenceFrame('U')
fore_arm_frame = ReferenceFrame('F')
# theta1 - shoulder, theta2 - elbow

theta1, theta2 = dynamicsymbols('theta1, theta2')
upper_arm_frame.orient(inertial_frame, 'Axis', (theta1, inertial_frame.z))
fore_arm_frame.orient(upper_arm_frame, 'Axis', (theta2, upper_arm_frame.z))


# # Points

# Joints
shoulder = Point('S')
upper_arm_length = symbols('l_U')

elbow = Point('E')
elbow.set_pos(shoulder, upper_arm_length * upper_arm_frame.y)

end_effector = Point('EE')
fore_arm_length = symbols('l_F')
end_effector.set_pos(elbow, fore_arm_length * fore_arm_frame.y)

# Center of Masses
upper_arm_com_length, fore_arm_com_length = symbols('d_U, d_F')

upper_arm_mass_center = Point('U_o')
upper_arm_mass_center.set_pos(
    shoulder, upper_arm_com_length * upper_arm_frame.y)

fore_arm_mass_center = Point('F_o')
fore_arm_mass_center.set_pos(elbow, fore_arm_com_length * fore_arm_frame.y)


# # Kinematical Differential Equations

# the generalized speeds are the angular velocities of the joints
omega1, omega2 = dynamicsymbols('omega1, omega2')

kinematical_differential_equations = [omega1 - theta1.diff(),
                                      omega2 - theta2.diff()]


# # Velocities

upper_arm_frame.set_ang_vel(inertial_frame, omega1 * inertial_frame.z)
fore_arm_frame.set_ang_vel(upper_arm_frame, omega2 * upper_arm_frame.z)
fore_arm_frame.ang_vel_in(inertial_frame)

shoulder.set_vel(inertial_frame, 0)
upper_arm_mass_center.v2pt_theory(shoulder, inertial_frame, upper_arm_frame)

elbow.v2pt_theory(shoulder, inertial_frame, upper_arm_frame)

fore_arm_mass_center.v2pt_theory(elbow, inertial_frame, fore_arm_frame)

end_effector.v2pt_theory(elbow, inertial_frame, fore_arm_frame)


# # Inertia

# Mass
upper_arm_mass, fore_arm_mass, end_effector_mass = symbols('m_U, m_F, m_EE')

# Inertia
upper_arm_inertia, fore_arm_inertia = symbols('I_Uz, I_Fz')

upper_arm_inertia_dyadic = inertia(upper_arm_frame, 0, 0, upper_arm_inertia)

upper_arm_central_inertia = (upper_arm_inertia_dyadic, upper_arm_mass_center)

fore_arm_inertia_dyadic = inertia(fore_arm_frame, 0, 0, fore_arm_inertia)

fore_arm_central_inertia = (fore_arm_inertia_dyadic, fore_arm_mass_center)

# rigid bodies
upper_arm = RigidBody('Upper Arm', upper_arm_mass_center, upper_arm_frame,
                      upper_arm_mass, upper_arm_central_inertia)

fore_arm = RigidBody('Upper Leg', fore_arm_mass_center, fore_arm_frame,
                     fore_arm_mass, fore_arm_central_inertia)


# # Kinetics

# gravity
g = symbols('g')
upper_arm_grav = (upper_arm_mass_center, -
                  upper_arm_mass * g * inertial_frame.y)

fore_arm_grav = (fore_arm_mass_center, -fore_arm_mass * g * inertial_frame.y)

end_effector_grav = (end_effector, -end_effector_mass * g * inertial_frame.y)

# joint torques

shoulder_torque, elbow_torque = dynamicsymbols('T_s, T_e')

upper_arm_torque = (upper_arm_frame,
                    shoulder_torque * inertial_frame.z - elbow_torque *
                    inertial_frame.z)

fore_arm_torque = (fore_arm_frame,
                   elbow_torque * inertial_frame.z)


# # Equations of Motion

coordinates = [theta1, theta2]

speeds = [omega1, omega2]

kane = KanesMethod(inertial_frame,
                   coordinates,
                   speeds,
                   kinematical_differential_equations)

loads = [
    upper_arm_grav,
    fore_arm_grav,
    upper_arm_torque,
    fore_arm_torque]

bodies = [upper_arm, fore_arm]

fr, frstar = kane.kanes_equations(bodies, loads)

mass_matrix = kane.mass_matrix_full
forcing_vector = kane.forcing_full

# Gravity compensation
fore_arm_grav_compensation_expr = fore_arm_mass_center.pos_from(
    elbow).dot(inertial_frame.x) * fore_arm_mass * -g
upper_arm_grav_compensation_expr = upper_arm_mass_center.pos_from(
    shoulder).dot(inertial_frame.x) * upper_arm_mass * -g + fore_arm_mass_center.pos_from(
    shoulder).dot(inertial_frame.x) * fore_arm_mass * -g

# TODO
# exponential degradation
# piecewise degradation modes
# torque joint limits
# 2 joints, 3 FM ea (100% 60% 30%)
# just the torque data
# can it lift the mass
# add some noise
# do with 5kg
