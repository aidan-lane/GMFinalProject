import sys
import logging
import time
from concurrent import futures

from decouple import config
import networkx as nx
import osmnx as ox
import grpc
import googlemaps

from gen import joyride_pb2
from gen import joyride_pb2_grpc


# Osmnx config options
ox.config(use_cache=True, log_console=True)

# Google Maps API Key (geo-encoding)
gmaps = googlemaps.Client(key=config("GMAPS_KEY"))


class Joyride(joyride_pb2_grpc.JoyRideServicer):

    def __init__(self, graph):
        self.G = graph

    def GetJoyRide(self, request, context):
        loc = gmaps.geocode(request.start)[0]["geometry"]["location"]
        startx = loc["lng"]
        starty = loc["lat"]

        loc2 = gmaps.geocode(request.end)[0]["geometry"]["location"]
        endx = loc2["lng"]
        endy = loc2["lat"]

        start_node = ox.nearest_nodes(self.G, startx, starty)
        end_node = ox.nearest_nodes(self.G, endx, endy)

        print(start_node, " ", end_node)
        print(self.G[start_node], "\n")
        print(self.G[end_node], "\n")

        return joyride_pb2.RideReply(message="{}, {}, {}".format(request.start, request.end, request.time))


def serve(port, graph):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    joyride_pb2_grpc.add_JoyRideServicer_to_server(Joyride(graph), server)
    server.add_insecure_port("[::]:{}".format(port))
    server.start()
    print("Started server on port {}".format(port))
    server.wait_for_termination()


def load_data(address, r):
    print("Loading graph data...")
    start_time = time.time()
    G = ox.graph_from_address(address, dist=r, simplify=False, network_type="drive", retain_all=True)
    print("Finished loading graph in {:.4f} seconds.\n".format(time.time() - start_time))

    # Calculate travel time based on length, speed (km/h) for each edge
    #G = ox.add_edge_speeds(G)
    #G = ox.add_edge_travel_times(G)
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

    G = load_data(location, int(radius * 1.6 * 1000))  # Convert miles to meters
    serve(port, G)
