import sys
import logging

import grpc

from gen import route_guide_pb2
from gen import route_guide_pb2_grpc


def guide_get_joyride(stub):
    ride = stub.GetJoyride(route_guide_pb2.Ride(start="", end="", minutes=0))


def start_client(ip, port):
    """ Attempts to create a gRPC client and connect to given IP and port
    """
    with grpc.insecure_channel("{}:{}".format(ip, port)) as channel:
        print("Started client. Connected to {}:{}".format(ip, port))
        stub = route_guide_pb2_grpc.RouteGuideStub(channel)
        guide_get_joyride(stub)


if __name__ == "__main__":
    logging.basicConfig()

    if len(sys.argv) != 3:
        print("Please specify a valid IP address and port")
        sys.exit(0)

    ip = sys.argv[1]
    port = sys.argv[2]
    start_client(ip, port)