import pathlib
import re
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

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        # for connection in reader.connections:
        #     print(connection.topic)

        connections = [x for x in reader.connections if x.topic == '/delivery_targets']
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            msg_data: str = msg.data
            prefix = "targets="
            assert msg_data.startswith(prefix)
            msg_data = msg_data[len(prefix):]

            # Should be a list of two arrays:
            # [array([[x1, y1],\n [x2, y2],\n ...]), array([[z1, z2, ...]])]
            # Extract everything between the array(...)
            matches = re.findall(r"(array\(.*?\))", msg_data, re.DOTALL)
            matches = list(matches)
            assert len(matches) == 2

            # Process xy array
            targets0 = eval(f"np.{matches[0]}")
            targets1 = eval(f"np.{matches[1]}")

            print("targets0 = np.{}".format(repr(targets0)))
            print("targets1 = np.{}".format(repr(targets1)))


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        app()
