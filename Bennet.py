import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys
from scipy import signal

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
                    if state_vector == 'position':
                        new_format_date = str(dt.year) + str(dt.month).zfill(2) + str(dt.day).zfill(2)
                        self.dates.append(int(new_format_date))
                    else:
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
        #v_square = np.sqrt (np.square(self.v_x))
        v_square = self.v_x
        return self.dates, v_square

    def get_module_y_velocity(self):
        #v_square = np.sqrt (np.square(self.v_y))
        v_square = self.v_y
        return self.dates, v_square

    def get_module_z_velocity(self):
        #v_square = np.sqrt (np.square(self.v_z))
        v_square = self.v_z
        return self.dates, v_square

def config(filename):
    bodies = None
    with open(filename) as stream:
        try:
            bodies = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return bodies

if __name__ == "__main__":
    bodies = config("bodies.yaml")
    if bodies is None:
        sys.exit("Bodies not found.")

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
        color_list = []
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

                #if body_lower_name != 'earth':
                for i in range(0, len(pos_body[1])):
                    if (i % 10) == 0:
                        ax.text(pos_body[1][i], pos_body[2][i], pos_body[3][i], pos_body[0][i], fontsize=7)

            ax.set_xlabel('x (UA)', fontsize = 20)
            ax.set_ylabel('y (UA)', fontsize = 20)
            ax.set_zlabel('z (UA)', fontsize = 20)

        legend = ax.legend(legend_list,  loc = 'center left', fontsize = 20)
        plt.title(f'Trajectories of the comet Bennet from ({angle[0]}, {angle[1]})', fontsize = 40)
        plt.savefig(f'bennet_trajectory_{angle[0]}_{angle[1]}.png')
        plt.show()

    print(f"Plotting velocity")
    legend_list = []
    fig = plt.figure(figsize=(20, 18))
    for body_index in range (0, len(bodies['bodies'])):
        for body, param in bodies['bodies'][body_index].items():
            body_lower_name = body.lower()
            if body_lower_name != 'bennet':
                continue
            print(f"Working on body: {body}")
            pos_body_dec = []

            horizon = Horizon()
            horizon.get_ephemeris(f"./Horizons/{body_lower_name}.txt", 'velocity')

            vel_body = horizon.get_module_velocity()
            plt.plot(vel_body[0][0:8137], vel_body[1][0:8137],'X')
            legend_list.append('|v(x,y,z)| Comet Bennet')
            plt.title(f'Velocity of comet Bennet', fontsize = 40)
            plt.xlabel('Date', fontsize = 20)
            plt.ylabel('Velocity (km/s)', fontsize=20)
            tick_label = ['x' if (i % 5) != 0 else str(val) for i, val in enumerate(vel_body[0][0:8137])]

            vel_x_body = horizon.get_module_x_velocity()
            plt.plot(vel_x_body[0][0:8137], vel_x_body[1][0:8137],'X')
            legend_list.append('v(x) Comet Bennet')

            vel_y_body = horizon.get_module_y_velocity()
            plt.plot(vel_y_body[0][0:8137], vel_y_body[1][0:8137],'X')
            legend_list.append('v(y) Comet Bennet')

            vel_z_body = horizon.get_module_z_velocity()
            plt.plot(vel_z_body[0][0:8137], vel_z_body[1][0:8137],'X')
            legend_list.append('v(z) Comet Bennet')

            plt.xticks(tick_label, tick_label, rotation=45, fontsize='20')

    plt.legend(legend_list,  loc = 'upper right', fontsize = 20)
    plt.grid(True)
    plt.savefig(f'Comet_Bennet_velocity.png')
    plt.show()
