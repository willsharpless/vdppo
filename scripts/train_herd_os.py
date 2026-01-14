import cyclopts
import ipdb

app = cyclopts.App()


@app.default()
def main():
    ...



if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
