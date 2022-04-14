import argparse
from concurrent import futures
import logging
import time

from decouple import config
import grpc
import networkx as nx
import osmnx as ox

from gen import joyride_pb2
from gen import joyride_pb2_grpc


# Osmnx config options
ox.config(use_cache=True, log_console=False)


class Joyride(joyride_pb2_grpc.JoyRideServicer):
    """ Servicer for the JoyRide gRPC service.
    """

    def __init__(self, graph):
        self.G = graph

    def GetJoyRide(self, request, context):
        print("Getting Joyride between {} and {}.".format(request.start, request.end))

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
        length, route = nx.bidirectional_dijkstra(G, start_node, end_node, weight="travel_time")
        last_name = None
        last_bearing = None

        for i in range(0, len(route)):
            if i == len(route) - 1:
                yield joyride_pb2.RideReply(node=route[-1], message="")
                continue
            
            edge = G.get_edge_data(route[i], route[i + 1])[0]
            bearing = int(edge["bearing"])
            street_name = edge["name"]
            direction = cardinal_direction(bearing)

            if not last_bearing:
                turn = " "
            else:
                turn = bearing_turn(last_bearing, bearing)

            if street_name == last_name:
                yield joyride_pb2.RideReply(node=route[i], message="")
                continue

            msg = "Turn{}on {} and head {}".format(turn, street_name, direction)
            last_name = street_name
            last_bearing = bearing

            yield joyride_pb2.RideReply(node=route[i], message=msg)


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
    G = ox.graph_from_address(address, dist=r, network_type="drive", simplify=False)
    print("Finished loading graph in {:.4f} seconds.".format(time.time() - start_time))

    # Calculate travel time based on length, speed (km/h) for each edge
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    G = ox.add_edge_bearings(G)
    print()

    return G


def bearing_turn(b1, b2):
  if ((((b1 - b2 + 540) % 360) - 180) > 0):
    return " left "
  else:
    return " right "


def cardinal_direction(b):
    dirs = ["North", "North-East", "East", "South-East", "South", "South-West", "West", 
        "North-West"]

    degree = 337.5

    for dir in dirs:
        if b >= degree and b < degree + 45:
            return dir

        if degree + 45 >= 360:
            degree = 22.5
        else:
            degree += 45
    
    return None


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
