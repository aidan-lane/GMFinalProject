import sys
import logging
import time
from concurrent import futures

from decouple import config
import networkx as nx
import osmnx as ox
import grpc
import googlemaps

from gen import route_guide_pb2
from gen import route_guide_pb2_grpc


# Osmnx config options
ox.config(use_cache=True, log_console=True)

# Google Maps API Key (geo-encoding)
gmaps = googlemaps.Client(key=config("GMAPS_KEY"))


class RouteGuideServicer(route_guide_pb2_grpc.RouteGuideServicer):
    """ Provides methods that implement functionality of route guide server.
    """

    def __init__(self):
        #self.G = graph
        pass

    def GetJoyride(self, request, context):
        print("hello!")
        res = gmaps.geocode(request.start)
        res2 = gmaps.geocode(request.end)
        print(res)
        return route_guide_pb2.Ride(node1=1,node2=3)


def start_server(port, G):
    """ Starts the gRPC server listening on the specified port
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    route_guide_pb2_grpc.add_RouteGuideServicer_to_server(RouteGuideServicer(), server)

    server.add_insecure_port("[::]:{}".format(port))
    server.start()
    print("Started server on port: {}".format(port))
    server.wait_for_termination()


def load_data(address, r):
    print("Loading graph data...")
    start_time = time.time()
    G = ox.graph_from_address(address, dist=r)
    print("Finished loading graph in {:.4f} seconds.\n".format(time.time() - start_time))

    # Calculate travel time based on length, speed (km/h) for each edge
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    print()

    return G


if __name__ == "__main__":
    logging.basicConfig()

    if len(sys.argv) != 4:
        print("Please use correct arguments. [port] [location] [radius (in miles)]")
        sys.exit(0)

    port = sys.argv[1]
    location = sys.argv[2]
    radius = int(sys.argv[3])

    G = load_data(location, radius * 1.6 * 1000)  # Convert miles to meters
    start_server(port, G)
