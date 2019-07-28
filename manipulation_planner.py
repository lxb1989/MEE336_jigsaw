from __future__ import print_function
import math
from math import pi
import numpy as np


class manipulation_planner():
    def __init__(self, home_location, home_orientation, heave_height):
        self.home_location = home_location
        self.home_orientation = home_orientation
        self.heave_height = heave_height
    
    def eular2rotation_vector(self, x, y, z, axes='xyz'):
        # eular -> matrix -> axis angle
        
        z = np.matrix([
        [math.cos(z), -math.sin(z), 0],
        [math.sin(z), math.cos(z), 0],
        [0, 0, 1]
        ])

        y = np.matrix([
        [math.cos(y), 0, math.sin(y)],
        [0, 1, 0],
        [-math.sin(y), 0, math.cos(y)]
        ])

        x = np.matrix([
        [1, 0, 0],
        [0, math.cos(x), -math.sin(x)],
        [0, math.sin(x), math.cos(x)]
        ])
        
        if axes == 'xyz': 
            R = x*y*z
        else:
            R = z*y*x
        # eval('R = '+axes[0]+'*'+ axes[1] +'*'+axes[2])
        theta = math.acos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
        multi = 1 / (2 * math.sin(theta))

        rx = multi * (R[2, 1] - R[1, 2])
        ry = multi * (R[0, 2] - R[2, 0])
        rz = multi * (R[1, 0] - R[0, 1])
        return (rx, ry, rz, theta)

    def manipulate_piece(self, which_piece, board_location, board_angle, piece_location, piece_angle, mani_step1_distance, mani_step2_distance):

        manipulation_flow = [[self.home_location, self.home_orientation, 'release']]
        if which_piece == 0:
            pick_angle = self.eular2rotation_vector(pi, 0, -piece_angle)
            place_angle = self.eular2rotation_vector(pi, 0, -board_angle)
        elif which_piece == 1:
            pick_angle = self.eular2rotation_vector(pi, 0, -piece_angle + pi/2)
            place_angle = self.eular2rotation_vector(pi, board_angle, -10*pi/180, axes='xzy')
        elif which_piece == 2:
            pick_angle = self.eular2rotation_vector(pi, 0, -piece_angle - pi/4)
            place_angle = self.eular2rotation_vector(pi, board_angle - pi/4, -10*pi/180, axes='xzy')
        elif which_piece == 3:
            pick_angle = self.eular2rotation_vector(pi, 0, -piece_angle + pi/2)
            place_angle = self.eular2rotation_vector(pi, board_angle, -10*pi/180, axes='xzy')
        else:
            raise Exception('param \'which_piece\' error!')
        
        # 1 move above piece
        location = [piece_location[0], piece_location[1], piece_location[2] + self.heave_height]    
        rotation_angle = pick_angle
        suction_cup = 'none'
        manipulation_flow = manipulation_flow + [[location, rotation_angle, suction_cup]]

        # 2 pick 
        location = piece_location
        suction_cup = 'suck'
        manipulation_flow = manipulation_flow + [[location, rotation_angle, suction_cup]]
        
        # 3 leave
        location = [piece_location[0], piece_location[1], piece_location[2] + self.heave_height]
        suction_cup = 'none'
        manipulation_flow = manipulation_flow + [[location, rotation_angle, suction_cup]]

        # 5 move above board
        if which_piece == 0:
            location = [board_location[0], board_location[1], board_location[2] + self.heave_height]
            rotation_angle = place_angle

        elif which_piece == 1:
            location = [board_location[0] + 0.01 * math.cos(board_angle), 
                board_location[1] + 0.01 * math.sin(board_angle), board_location[2] + self.heave_height]
            rotation_angle = place_angle

        elif which_piece == 2:
            location = [location[0] - 0.01 * math.cos(board_angle) + 0.01 * math.sin(board_angle), 
                location[1] - 0.01 * math.sin(board_angle) + 0.01 * math.cos(board_angle), location[2] + self.heave_height]
            rotation_angle = place_angle

        elif which_piece == 3:
            location = [board_location[0] + 0.01 * math.sin(board_angle),
                board_location[1] + 0.01 * math.cos(board_angle), board_location[2] + self.heave_height]
            rotation_angle = place_angle

        suction_cup = 'none'
        manipulation_flow = manipulation_flow + [[location, rotation_angle, suction_cup]]

        # 6 descend
        if which_piece == 0:
            location = board_location

        elif which_piece == 1:
            location[2] = board_location[2]
            
        elif which_piece == 2:
            location[2] = location[2]

        elif which_piece == 3:
            location = board_location[2]


        suction_cup = 'none'
        manipulation_flow = manipulation_flow + [[location, rotation_angle, suction_cup]]

        # 7 mani_step1

        if which_piece == 0:
            new_board_location = [board_location[0] - mani_step1_distance * math.sin(board_angle), 
                board_location[1] - mani_step2_distance * math.cos(board_angle), board_location[2]]
            location = new_board_location

        elif which_piece == 1:
            location = [location[0] - mani_step1_distance * math.cos(board_angle + pi/2), 
                location[1] - mani_step1_distance * math.sin(board_angle + pi/2), location[2]]

        elif which_piece == 2:
            location = [location[0] + mani_step1_distance * math.sin(board_angle), 
                location[1] + mani_step1_distance * math.cos(board_angle), location[2]]

        elif which_piece == 3:
            location = [location[0] + mani_step1_distance * math.cos(board_angle), 
                location[1] + mani_step1_distance * math.sin(board_angle), board_location[2]]

        suction_cup = 'none'
        manipulation_flow = manipulation_flow + [[location, rotation_angle, suction_cup]]

        # 8 mani_step2
        if which_piece == 0:
            new_board_location = [new_board_location[0] - mani_step1_distance * math.cos(board_angle), 
                new_board_location[1] - mani_step1_distance * math.sin(board_angle), new_board_location[2]]
            location = new_board_location

        elif which_piece == 1:
            location = [location[0] + mani_step2_distance * math.cos(board_angle + pi), 
                location[1] + mani_step2_distance * math.sin(board_angle + pi), location[2]]

        elif which_piece == 2:
            location = [location[0] - mani_step2_distance * math.cos(board_angle), 
                location[1] - mani_step2_distance * math.sin(board_angle), location[2]]

        elif which_piece == 3:
            location = [location[0] - mani_step2_distance * math.sin(board_angle), 
                location[1] - mani_step2_distance * math.cos(board_angle), location[2]]

        suction_cup = 'release'
        manipulation_flow = manipulation_flow + [[location, rotation_angle, suction_cup]]

        # 9 leave
        if which_piece == 0:
            location = [new_board_location[0], new_board_location[1], new_board_location[2] + self.heave_height]

        elif which_piece == 1:
            location[2] = location[2] + self.heave_height

        elif which_piece == 2:
            location[2] = location[2] + self.heave_height

        elif which_piece == 3:
            location[2] = location[2] + self.heave_height

        suction_cup = 'none'
        manipulation_flow = manipulation_flow + [[location, rotation_angle, suction_cup]]

        # 10 home
        manipulation_flow = manipulation_flow + [[self.home_location, self.home_orientation, 'none']]

        return manipulation_flow
    