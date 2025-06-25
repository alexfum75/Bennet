import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys
from scipy import signal

# Plot trajectory, velocity and angles of Bennet comet
# Compare result with https://theskylive.com/cometbennett-info
# use NASA API https://ssd.jpl.nasa.gov/horizons/app.html
# color https://matplotlib.org/stable/gallery/color/named_colors.html

class Horizon:
    def __init__(self):
        self.dates = []
        self.x = []
        self.y = []
        self.z = []
        self.v_x = []
        self.v_y = []
        self.v_z = []

    def _parse_pos_line(self, line):
        sx, sy, sz = line[4:26], line[30:52], line[56:78]
        return float(sx), float(sy), float(sz)

    def _parse_velocity_line(self, line):
        vx, vy, vz = line[5:26], line[30:52], line[56:78]
        return float(vx), float(vy), float(vz)

    def get_ephemeris(self, filename, state_vector):
        with open(filename) as fi:
            while not fi.readline().strip() == "$$SOE":
                pass
            while True:
                line = fi.readline().rstrip()
                if line == "$$EOE":
                    break
                if "A.D." in line:
                    date = line[25:36]
                    dt = datetime.datetime.strptime(date, '%Y-%b-%d')
                    #if state_vector == 'position':
                    #    new_format_date = str(dt.year) + str(dt.month).zfill(2) + str(dt.day).zfill(2)
                    #    self.dates.append(int(new_format_date))
                    #else:
                    self.dates.append(date)
                elif line.startswith(" X =") and (state_vector == 'position'):
                    sx, sy, sz = self._parse_pos_line(line)
                    self.x.append(sx)
                    self.y.append(sy)
                    self.z.append(sz)
                elif line.startswith(" VX=")and (state_vector == 'velocity'):
                    svx, svy, svz = self._parse_velocity_line(line)
                    self.v_x.append(svx * 1731.46)
                    self.v_y.append(svy * 1731.46)
                    self.v_z.append(svz * 1731.46) # from  AU/day in km/s

    def get_position(self):
        return self.dates, self.x, self.y, self.z

    def get_velocity(self):
        return self.dates, self.v_x, self.v_y, self.v_z

    def get_module_velocity(self):
        v_square = np.sqrt (np.square(self.v_x) + np.square(self.v_y) + np.square(self.v_z))
        return self.dates, v_square

    def get_module_x_velocity(self):
        v_square = self.v_x
        return self.dates, v_square

    def get_module_y_velocity(self):
        v_square = self.v_y
        return self.dates, v_square

    def get_module_z_velocity(self):
        v_square = self.v_z
        return self.dates, v_square

def config(filename):
    _bodies = None
    with open(filename) as stream:
        try:
            _bodies = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return _bodies

def calculate_angle(coord_p_1, coord_p_2, coord_p_3):
    p_1 = np.array(coord_p_1)
    p_2 = np.array(coord_p_2)
    p_3 = np.array(coord_p_3)

    # calculate vectors
    vec_p1p2 = p_2 - p_1
    vec_p1p3 = p_3 - p_1
    vec_p2p1 = p_1 - p_2
    vec_p2p3 = p_3 - p_2
    vec_p3p1 = p_1 - p_3
    vec_p3p2 = p_2 - p_3

    # calculate vertex p1
    cos_p1 = np.dot(vec_p1p2, vec_p1p3) / (np.linalg.norm(vec_p1p2) * np.linalg.norm(vec_p1p3))
    p1_rad = np.arccos(cos_p1)
    p1_deg = np.degrees(p1_rad)

    # calculate vertex p2
    cos_p2 = np.dot(vec_p2p1, vec_p2p3) / (np.linalg.norm(vec_p2p1) * np.linalg.norm(vec_p2p3))
    p2_rad = np.arccos(cos_p2)
    p2_deg = np.degrees(p2_rad)

    # calculate vertex p3
    cos_p3 = np.dot(vec_p3p1, vec_p3p2) / (np.linalg.norm(vec_p3p1) * np.linalg.norm(vec_p3p2))
    p3_rad = np.arccos(cos_p3)
    p3_deg = np.degrees(p3_rad)
    return p1_deg, p2_deg, p3_deg


def plotTrajectory (bodies):
    angles = []
    angles.append((20,90))
    angles.append((90, 0))
    angles.append((45, 45))
    angles.append((20, 160))
    angles.append((-5, 20))
    angles.append((-5, -130))
    angles.append((20, -90))
    angles.append((10, -90))

    print(f"Plotting position")
    for angle in angles:
        fig = plt.figure(figsize=(20,16))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(angle[0], angle[1])

        legend_list = []
        print(f"Working with projection angle: {angle}")
        for body_index in range (0, len(bodies['bodies'])):
            for body, param in bodies['bodies'][body_index].items():
                print(f"Working on body: {body}")
                legend_list.append(body)
                body_lower_name = body.lower()
                pos_body_dec = []

                horizon = Horizon()
                horizon.get_ephemeris(f"./Horizons/{body_lower_name}.txt", 'position')
                pos_body = horizon.get_position()

                pos_body_dec.append(pos_body[0])
                if 'decimate' in param:
                    pos_body_dec.append(signal.decimate(pos_body[1], param['decimate']))
                else:
                    pos_body_dec.append(pos_body[1])
                if 'decimate' in param:
                    pos_body_dec.append(signal.decimate(pos_body[2], param['decimate']))
                else:
                    pos_body_dec.append(pos_body[2])
                if 'decimate' in param:
                    pos_body_dec.append(signal.decimate(pos_body[3], param['decimate']))
                else:
                    pos_body_dec.append(pos_body[3])

                # bullet size
                if body == 'Sun':
                    size = [49 for element in range(len(pos_body_dec[1]))]
                else:
                    size = [4 for element in range(len(pos_body_dec[1]))]
                ax.scatter(pos_body_dec[1], pos_body_dec[2], pos_body_dec[3], '.', color=param['color'], sizes=size)

                for i in range(0, len(pos_body[1])):
                    if (i % 10) == 0:
                        ax.text(pos_body[1][i], pos_body[2][i], pos_body[3][i], pos_body[0][i], fontsize=7)

            ax.set_xlabel('x (UA)', fontsize = 20)
            ax.set_ylabel('y (UA)', fontsize = 20)
            ax.set_zlabel('z (UA)', fontsize = 20)

        legend = ax.legend(legend_list,  loc = 'center left', fontsize = 20)
        plt.title(f'Trajectories of the comet Bennet from ({angle[0]}, {angle[1]})', fontsize = 40)
        plt.savefig(f'Bennet_trajectory_{angle[0]}_{angle[1]}.png')
        plt.show()


def plotVelocity (bodies):
    print(f"Plotting velocity")
    body = ''
    body_lower_name = ''
    found_target = False
    for body_index in range(0, len(bodies['bodies'])):
        for _body, _param in bodies['bodies'][body_index].items():
            if 'velocity' not in _param:
                _param['velocity'] = 'False'
            body = _body
            body_lower_name = _body.lower()
            found_target = True
            break
        if found_target:
            break

    legend_list = []
    fig = plt.figure(figsize=(20, 18))

    print(f"Working on body: {body}")
    horizon = Horizon()
    horizon.get_ephemeris(f"./Horizons/{body_lower_name}.txt", 'velocity')

    vel_body = horizon.get_module_velocity()
    min_size = min(len(vel_body[0]), len(vel_body[1]))
    plt.plot(vel_body[0][0:min_size], vel_body[1][0:min_size],'X')
    legend_list.append('|v(x,y,z)| Comet Bennet')
    plt.title(f'Velocity of comet Bennet', fontsize = 40)
    plt.xlabel('Date', fontsize = 20)
    plt.ylabel('Velocity (km/s)', fontsize=20)
    tick_label = ['X' if (i % 5) != 0 else str(val) for i, val in enumerate(vel_body[0][0:min_size])]

    vel_x_body = horizon.get_module_x_velocity()
    plt.plot(vel_x_body[0][0:min_size], vel_x_body[1][0:min_size],'X')
    legend_list.append('v(x) Comet Bennet')

    vel_y_body = horizon.get_module_y_velocity()
    plt.plot(vel_y_body[0][0:min_size], vel_y_body[1][0:min_size],'X')
    legend_list.append('v(y) Comet Bennet')

    vel_z_body = horizon.get_module_z_velocity()
    plt.plot(vel_z_body[0][0:min_size], vel_z_body[1][0:min_size],'X')
    legend_list.append('v(z) Comet Bennet')

    plt.xticks(tick_label, tick_label, rotation=45, fontsize='20')
    plt.legend(legend_list,  loc = 'upper right', fontsize = 20)
    plt.grid(True)
    plt.savefig(f'Comet_Bennet_velocity.png')
    plt.show()

def plotAngle (bodies):
    print(f"Plotting angles")
    const_conversion = 1.496e+8
    earth_coord_list = []
    sun_coord_list = []
    bennet_coord_list = []
    time = []

    for body_index in range (0, len(bodies['bodies'])):
        for body, param in bodies['bodies'][body_index].items():
            body_lower_name = body.lower()
            if body_lower_name != 'earth' and body_lower_name != 'bennet'and body_lower_name != 'sun':
                continue
            print(f"Working on body: {body}")
            horizon = Horizon()
            horizon.get_ephemeris(f"./Horizons/{body_lower_name}.txt", 'position')
            pos_body = horizon.get_position()
            time = pos_body[0]
            for i in range(0, len(pos_body[1])):
                coord = [pos_body[1][i] * const_conversion, pos_body[2][i] * const_conversion, pos_body[3][i] * const_conversion]
                if body_lower_name == 'earth':
                    earth_coord_list.append(coord)
                if body_lower_name == 'bennet':
                    bennet_coord_list.append(coord)
                if body_lower_name == 'sun':
                    sun_coord_list.append(coord)

    fig = plt.figure(figsize=(20, 18))
    p1_list = []
    p2_list = []
    p3_list = []

    legend_list = ['Angle at vertex E', 'Angle at vertex B', 'Angle at vertex S']
    for i in range(len(earth_coord_list)):
        p1, p2, p3 = calculate_angle(earth_coord_list[i], bennet_coord_list[i], sun_coord_list[i])
        p1_list.append(p1)
        p2_list.append(p2)
        p3_list.append(p3)
    min_size = min(len(time), len(p1_list))
    tick_label = ['.' if (i % 5) != 0 else str(val) for i, val in enumerate(time[0:min_size])]

    plt.title(f'Anomalies between Sun, Earth and comet Bennet', fontsize = 40)
    plt.plot(time[0:min_size], p1_list[0:min_size],'*')
    plt.plot(time[0:min_size], p2_list[0:min_size],'*')
    plt.plot(time[0:min_size], p3_list[0:min_size],'*')
    plt.xticks(tick_label, tick_label, rotation=45, fontsize='20')
    plt.legend(legend_list, loc = 'upper right', fontsize = 20)
    plt.grid(True)
    plt.savefig(f'Bennet_Earth_Sun_angle.png')
    plt.show()


if __name__ == "__main__":
    bodies = config("bodies.yaml")
    if bodies is None:
        sys.exit("Bodies not found.")

    plotAngle(bodies)
    plotTrajectory(bodies)
    plotVelocity (bodies)
    exit(0)
