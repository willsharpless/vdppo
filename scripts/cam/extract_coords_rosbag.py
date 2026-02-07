import pathlib
import matplotlib.pyplot as plt
from datetime import timedelta
import datetime
import numpy as np

import cyclopts
from scipy.interpolate import make_splrep
import ipdb
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


app = cyclopts.App()


@app.default()
def main(bag_path: pathlib.Path):
    # Explicitly create a type store for legacy ROS2 bags without message definitions.
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    T_sec = []
    T_pos = []

    dt0 = None

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        # for connection in reader.connections:
        #     print(connection.topic)
        # exit(0)

        connections = [x for x in reader.connections if x.topic == '/cf20/odom']
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            pos_msg = msg.pose.pose.position
            pos = np.array([pos_msg.x, pos_msg.y, pos_msg.z])
            T_pos.append(pos)

            stamp = msg.header.stamp
            sec, nanosec = stamp.sec, stamp.nanosec
            dt = timedelta(seconds=sec, microseconds=nanosec // 1000)

            if dt0 is None:
                dt0 = dt

            T_sec.append((dt - dt0).total_seconds())

    T_sec = np.array(T_sec)
    T_pos = np.stack(T_pos, axis=0)

    # Downsample.
    ds_factor = 3

    T_sec = T_sec[::ds_factor]
    T_pos = T_pos[::ds_factor]

    # Construct a smoothing spline to compute velocity estimates.
    s = 1e-5
    print("splrep_x..")
    spl_x = make_splrep(T_sec, T_pos[:, 0], s=s)
    print("splrep_x.. done!")

    spl_y = make_splrep(T_sec, T_pos[:, 1], s=s)
    spl_z = make_splrep(T_sec, T_pos[:, 2], s=s)

    # t_final = min(T_sec[-1], 130.0)
    t_final = T_sec[-1]

    S_sec = np.linspace(T_sec[0], t_final, num=len(T_sec))

    S_vel_x = spl_x.derivative()(S_sec)
    S_vel_y = spl_y.derivative()(S_sec)
    S_vel_z = spl_z.derivative()(S_sec)

    S_vel = np.stack([S_vel_x, S_vel_y, S_vel_z], axis=-1)
    S_speed = np.linalg.norm(S_vel, axis=-1)

    # ----------------------------------------------
    # frames = arr = np.array([339, 546, 872, 1323, 1594, 2006, 2346, 3466, 3886, 4170, 4415, 4705, 5022, 5372, 5796])
    frames = np.array([
    370, 775, 1061, 1341, 1650,
    1941, 2224, 2484, 2910, 3208,
    3519, 4315, 4860, 5324, 6180,
    6522, 6823, 7139, 7464, 7905
    ])
    fps = 50
    times = frames / fps
    times = times - times[0]

    # offset = 15.5  cf20
    offset = 12.8
    times = times + offset
    # -----------------------------------------------

    # Get the position at each of the times
    pos_x = spl_x(times)
    pos_y = spl_y(times)
    pos_z = spl_z(times)
    positions = np.stack([pos_x, pos_y, pos_z], axis=-1)
    print("Positions at specified times:")
    print(repr(positions))

    # # Compute the velocities by finite differences
    # Tm1_vel = np.linalg.norm(T_pos[1:] - T_pos[:-1], axis=-1) / (T_sec[1:] - T_sec[:-1])

    fig, ax = plt.subplots()
    ax.plot(S_sec, S_speed)
    # ax.set(xlabel="Time (s)", ylabel="Velocity (m/s)", title="CF20 Velocity from Odometry")
    ax.set(xlabel="Time (s)", ylabel="Velocity (m/s)", title="CF3 Velocity from Odometry")

    for t in times:
        ax.axvline(x=t, color='r', linestyle='--')

    plt.show()


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        app()