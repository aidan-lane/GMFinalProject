from concurrent import futures
import logging
import math
from __future__ import print_function
import sys

import grpc
import networkx as nx
import osmnx as ox

from gen import joyride_pb2
from gen import joyride_pb2_grpc


def get_center(coords):
    """ Gets the center coordinate (lat, lng) of a list of coordinate tuples.
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
    subsection of the graph base don the center point of the two addresses.

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
    margin_scale = 0.05
    dis = get_distance(c1, c2)*1000  # Convert km to m
    dis = dis/2 + dis*margin_scale

    G = ox.graph_from_point(get_center([c1, c2]), dist=dis, network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

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

        response = stub.GetJoyRide(joyride_pb2.RideRequest(start=start, end=end, time=time))
        G = future.result()
    
    return response


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
        response = get_joyride(stub, "10 King Street, Troy NY", "95 14th Street, Troy NY", 0)
        
    print("Received:", response.message)


if __name__ == "__main__":
    logging.basicConfig()

    if len(sys.argv) != 3:
        print("Please specify a valid IP address and port")
        sys.exit(0)

    ip = sys.argv[1]
    port = sys.argv[2]
    run(ip, port)
