#!/usr/bin/env python
from sympy import symbols, simplify, pi
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody, Particle, KanesMethod
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

end_effector_point = Point('EE')
fore_arm_length = symbols('l_F')
end_effector_point.set_pos(elbow, fore_arm_length * fore_arm_frame.y)

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

end_effector_point.v2pt_theory(elbow, inertial_frame, fore_arm_frame)

end_effector_x_expr = end_effector_point.pos_from(shoulder).dot(inertial_frame.x)
end_effector_y_expr = end_effector_point.pos_from(shoulder).dot(inertial_frame.y)

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

fore_arm = RigidBody('Fore Arm', fore_arm_mass_center, fore_arm_frame,
                     fore_arm_mass, fore_arm_central_inertia)

# and a particle

end_effector = Particle('End Effector', end_effector_point, end_effector_mass)

# # Kinetics

# gravity
g = symbols('g')
upper_arm_grav = (upper_arm_mass_center, -
                  upper_arm_mass * g * inertial_frame.y)

fore_arm_grav = (fore_arm_mass_center, -fore_arm_mass * g * inertial_frame.y)

end_effector_grav = (end_effector_point, -
                     end_effector_mass * g * inertial_frame.y)

# degradation torques
shoulder_degradation, elbow_degradation = symbols('R_s, R_e')

shoulder_degradation_torque = shoulder_degradation * -omega1  # viscous friction
# shoulder_degradation_torque = shoulder_degradation * -np.sign(omega1)  # kinetic friction
# shoulder_degradation_torque = (1 + shoulder_degradation_rate)**sm.dynamicssymbols._t * -omega1 # continuous-exponential (untested)

elbow_degradation_torque = elbow_degradation * -omega2  # viscous friction
# elbow_degradation_torque = elbow_degradation * -np.sign(omega2)  # kinetic friction
# elbow_degradation_torque = (1 + elbow_degradation_rate)**sm.dynamicssymbols._t * -omega2 # continuous-exponential (untested)


# joint torques

shoulder_torque, elbow_torque = dynamicsymbols('T_s, T_e')

upper_arm_torque = (upper_arm_frame,
                    (shoulder_torque + shoulder_degradation_torque) * inertial_frame.z - (elbow_torque + elbow_degradation_torque) *
                    inertial_frame.z)

fore_arm_torque = (fore_arm_frame,
                   (elbow_torque + elbow_degradation_torque) * inertial_frame.z)


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
    end_effector_grav,
    upper_arm_torque,
    fore_arm_torque]

bodies = [upper_arm, fore_arm, end_effector]

fr, frstar = kane.kanes_equations(bodies, loads)

mass_matrix = kane.mass_matrix_full
forcing_vector = kane.forcing_full

# Gravity compensation
fore_arm_grav_compensation_expr = fore_arm_mass_center.pos_from(
    elbow).dot(inertial_frame.x) * fore_arm_mass * -g
upper_arm_grav_compensation_expr = upper_arm_mass_center.pos_from(
    shoulder).dot(inertial_frame.x) * upper_arm_mass * -g + fore_arm_mass_center.pos_from(
    shoulder).dot(inertial_frame.x) * fore_arm_mass * -g

def reachable(model, x, y):
    l1 = model[upper_arm_length]
    l2 = model[fore_arm_length]
    r = l1 + l2 
    if y > 0:
        y_reachable = r**2-x**2
        return y_reachable >= y
    else:
        y_reachable = x**2-r**2
        return y_reachable <= y

def inverseKinematics(model, x, y, armconfig=1):
    if not reachable(model, x, y):
        raise RuntimeError(f"specified coordinates ({x}, {y}) not reachable.")
    # something is weird about numpy atan2 I think
    tmp = x
    x=y
    y=tmp
    
    # Modern Robotics pg 221
    l1 = model[upper_arm_length]
    l2 = model[fore_arm_length]
    
    D = (np.power(x,2)+np.power(y,2)-np.power(l1,2)-np.power(l2,2))/(2*l1*l2)
    theta2 = np.arctan2(armconfig*np.sqrt(1-np.power(D,2)), D)
    
    theta1 = np.arctan2(y,x) - np.arctan2(l2*np.sin(theta2), l1+l2*np.cos(theta2))
    return [theta1, theta2]

# TODO
# torque joint limits
