import sys
import logging
from concurrent import futures

import networkx as nx
import osmnx as ox
import grpc

from gen import route_guide_pb2
from gen import route_guide_pb2_grpc


class RouteGuideServicer(route_guide_pb2_grpc.RouteGuideServicer):
    """ Provides methods that implement functionality of route guide server.
    """

    def __init__(self):
        pass

    def GetJoyride(self, request, context):
        return super().GetJoyride(request, context)


def start_server(port):
    """ Starts the gRPC server listening on the specified port
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    route_guide_pb2_grpc.add_RouteGuideServicer_to_server(RouteGuideServicer(), server)

    server.add_insecure_port("[::]:{}".format(port))
    server.start()
    print("Started server on port: {}".format(port))
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()

    if len(sys.argv) != 2:
        print("Please specify a port in program arguments")
        sys.exit(0)

    port = sys.argv[1]
    start_server(port)
