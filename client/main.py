import argparse
import cmd
from concurrent import futures
import logging
import math
import shlex

import grpc
import networkx as nx
import osmnx as ox

from gen import joyride_pb2
from gen import joyride_pb2_grpc


def get_center(coords):
    """ Gets the center coordinate (lat, lng) of a list of coordinate tuples.

    Args:
        coords: list of coordinate tuples

    Returns:
        Coordinate tuple of center point
    """
    n = len(coords)
    if n == 1:
        return coords[0]
    
    x, y, z = 0, 0, 0

    for c in coords:
        lat = c[0] * math.pi / 180
        lng = c[1] * math.pi / 180

        x += math.cos(lat) * math.cos(lng)
        y += math.cos(lat) * math.sin(lng)
        z += math.sin(lat)

    x /= n
    y /= n
    z /= n

    clng = math.atan2(y, x)
    clat = math.atan2(z, math.sqrt(x * x + y * y))

    return clat * 180 / math.pi, clng * 180 / math.pi


def get_distance(c1, c2):
    """ Returns approximate distance between two pairs of coordinates in km.

    Args:
        c1: coordinate pair 1
        c2: coordinate pair 2

    Returns:
        distance in km between c1 and c2
    """
    # approximate radius of earth in km
    R = 6373.0

    lat1 = math.radians(c1[0])
    lng1 = math.radians(c1[1])
    lat2 = math.radians(c2[0])
    lng2 = math.radians(c2[1])

    dlon = lng2 - lng1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def load_subsection(start, end):
    """ Loads subsection of roadnetwork graph

    To save computation time when rendering the map for the user, we calculate a
    subsection of the graph based on the center point of the two addresses.

    Args:
        start: Start address string
        end: End address string

    Returns:
        A NetworkX MultiDiGraph
    """

    c1 = ox.geocode(start)
    c2 = ox.geocode(end)

    # Here we want to create a tight bounding box around the start and end points.
    # This is accomplished by first finding the center point. Next, a distance is
    # calcuated so that both points fit within the bounding box with some additional
    # margin space.
    margin_scale = 0.1
    dis = get_distance(c1, c2)*1000  # Convert km to m
    dis = dis/2 + dis*margin_scale

    G = ox.graph_from_point(get_center([c1, c2]), dist=dis, network_type="drive", simplify=True)

    return G


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
    with futures.ThreadPoolExecutor() as executor:
        future = executor.submit(load_subsection, start, end)

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

        _ = future.result()

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
                print()
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
