import ipdb
import pathlib
import cyclopts
import pickle

app = cyclopts.App()

@app.default()
def main(pkl_path: pathlib.Path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    ipdb.set_trace()

if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        app()