import argparse
from concurrent import futures
import logging
import sys
import time

from decouple import config
import grpc
import networkx as nx
import osmnx as ox

from gen import joyride_pb2
from gen import joyride_pb2_grpc


# Osmnx config options
ox.config(use_cache=True, log_console=True)


class Joyride(joyride_pb2_grpc.JoyRideServicer):
    """ Servicer for the JoyRide gRPC service.
    """

    def __init__(self, graph):
        self.G = graph

    def GetJoyRide(self, request, context):
        # Get latitude, longitude of the start and end points
        start = ox.geocode(request.start)
        end = ox.geocode(request.end)
        starty, startx = start
        endy, endx = end

        # Find closest node (intersection) to each
        start_node = ox.nearest_nodes(self.G, startx, starty)
        end_node = ox.nearest_nodes(self.G, endx, endy)

        # Find shortest path weighted on pre-computed travel time
        #TODO(aidan) handle case where shortest path time is greater than requested time
        route = nx.shortest_path(G, start_node, end_node, weight="travel_time")

        return joyride_pb2.RideReply(message="{}, {}, {}".format(
            request.start, request.end, request.time))


def serve(port, graph):
    """ Starts gRPC server instance

    Starts the defined service class on the specified port, using data from the given
    graph.

    Args:
        port: port number as string
        graph: MultiDigraph to be queried by this service
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    joyride_pb2_grpc.add_JoyRideServicer_to_server(Joyride(graph), server)
    server.add_insecure_port("[::]:{}".format(port))

    server.start()
    print("Started server on port {}".format(port))
    server.wait_for_termination()


def load_data(address, r):
    """ Loads initial road network data

    Since we are dealing with a large amount of data from all over the world, we must
    specify a subset of this entire road network graph so that this application remains
    viable.

    Args:
        address: central point of graph
        r: radius in meters around address to be loaded

    Returns
        NetworkX.MultiDiGraph

    """
    start_time = time.time()
    G = ox.graph_from_address(address, dist=r, simplify=False, network_type="drive")
    print("Finished loading graph in {:.4f} seconds.\n".format(time.time() - start_time))

    # Calculate travel time based on length, speed (km/h) for each edge
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    G = ox.add_edge_bearings(G)
    print()

    return G


if __name__ == "__main__":
    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("port", help="Port for gRPC instance to be started on", type=str)
    parser.add_argument("location", 
        help="Location/address for graph to be centered around", type=str)
    parser.add_argument("radius", 
        help="Radius around central point to be included in network", type=int)

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(e.message, e.args)

    G = load_data(args.location, int(args.radius * 1.6 * 1000))  # Convert miles to meters
    serve(args.port, G)
