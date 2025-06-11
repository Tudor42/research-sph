from application.server.server import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50007)
    parser.add_argument("--password", type=str, required=True, help="Password clients must provide in init")

    args = parser.parse_args()
    main(args.password, args.host, args.port)