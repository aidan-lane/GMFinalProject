import argparse
import cmd
from concurrent import futures
import logging
import shlex

import grpc

from gen import joyride_pb2
from gen import joyride_pb2_grpc


def get_joyride(stub, start, end, time):
    """ Gets a JoyRide

    Given a starting location, end location and time (in minutes), gets a JoyRide 
    computed by the server.

    Usage:
        get_joyride(stub, "1999 Burdett Avenue, Troy NY", "River Street, Troy NY", 30)

    Args:
        stub: gRPC stub
        start: address of starting location
        end:  address of ending location
        time: number of minutes the ride should last

    Returns:
        Returns a valid route from the gRPC server
    """

    # Stream nodes from stub and calculate path
    nodes = []
    directions = []

    stream = stub.GetJoyRide(joyride_pb2.RideRequest(start=start, end=end, time=time))
    for response in stream:
        if response == grpc.aio.EOF:
            break

        nodes.append(response.node)
        if response.message != "":
            directions.append(response.message)
            print(response.message)

    return nodes

def update_ratings(stub, path):
    print("How did you feel about your joyride? Enter a rating between 1-10")
    rating = -1
    while True:
        rating = int(input("Rating: "))

        if rating in range(1,3):
            final_rating = 0
        elif rating in range(3,6):
            final_rating = 1
        elif rating in range(6,11):
            final_rating = 2
        else:
            print("Invalid rating entered...")
            continue
        break

    stub.GetRideRating(joyride_pb2.RideRating(rating=final_rating, path=path))
    print("Rating saved! Thanks for your feedback")


class AppShell(cmd.Cmd):
    prompt = ">>> "
    intro = "Type help or ? to list commands.\n"

    def __init__(self, stub):
        super(AppShell, self).__init__()
        self.stub = stub

    def precmd(self, line):
        return line.lower().strip()

    def do_quit(self, _):
        "Exits the application"
        return True

    def do_authors(self, _):
        "Prints author and related information for this project"

        authors = ["Aidan Lane", "Caitlin Crowley","Stephen Zenack"]
        for author in sorted(authors):
          print(author)

    def do_joyride(self, line):
        "Request a new JoyRide"
        parser = argparse.ArgumentParser()
        parser.add_argument("start", help="Starting location address", type=str)
        parser.add_argument("end", help="End destination address", type=str)
        parser.add_argument("time", help="Time in minutes the trip should last", type=int)

        try:
            args = parser.parse_args(shlex.split(line))
        except argparse.ArgumentError as e:
            print(e.message, e.args)
        except SystemExit:
            return  # Stop rest of command but allow user to continue

        nodes = get_joyride(self.stub, args.start, args.end, args.time * 60)

        # Get user rating for the ride
        update_ratings(self.stub, nodes)


def run(ip, port):
    """ Runs gRPC client

    Attempts to connect to a gRPC server instance using the given ip and port.

    Args:
        ip: Remote ip address string
        port: port number in string format
    """
    with grpc.insecure_channel("{}:{}".format(ip, port)) as channel:
        print("Connected to gRPC server {}:{}".format(ip, port))
        stub = joyride_pb2_grpc.JoyRideStub(channel)

        shell = AppShell(stub)
        while True:
            try:
                welcome = """
                         __            ____  _     __   
                        / /___  __  __/ __ \(_)___/ /__ 
                   __  / / __ \/ / / / /_/ / / __  / _ \\
                  / /_/ / /_/ / /_/ / _, _/ / /_/ /  __/
                  \____/\____/\__, /_/ |_/_/\__,_/\___/ 
                             /____/                     \n"""
                print(welcome)
                shell.cmdloop()
                break
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("ip", help="IP address to connect to", type=str)
    parser.add_argument("port", help="gRPC server port", type=str)

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(e.message, e.args)

    run(args.ip, args.port)
