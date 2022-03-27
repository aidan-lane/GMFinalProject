from __future__ import print_function
import sys
import logging

import grpc

from gen import joyride_pb2
from gen import joyride_pb2_grpc


def run(ip, port):
    with grpc.insecure_channel("{}:{}".format(ip, port)) as channel:
        print("Connected to gRPC server {}:{}".format(ip, port))
        stub = joyride_pb2_grpc.JoyRideStub(channel)
        response = stub.GetJoyRide(joyride_pb2.RideRequest(start="10 South Point Terrace, Kinnelon NJ", end="6 Stonybrook Trail, Kinnelon NJ", time=0))
    print("Received: " + response.message)


if __name__ == "__main__":
    logging.basicConfig()

    if len(sys.argv) != 3:
        print("Please specify a valid IP address and port")
        sys.exit(0)

    ip = sys.argv[1]
    port = sys.argv[2]
    run(ip, port)