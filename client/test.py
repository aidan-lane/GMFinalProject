import argparse
import cmd
from concurrent import futures
import logging
import shlex
import datetime

import grpc

from gen import joyride_pb2
from gen import joyride_pb2_grpc

import main

def testJoyride(stub):
    start = "1999 Burdett Avenue, Troy NY"
    end = "River Street, Troy NY"
    passed = 0
    
    # check that a path is returned
    ret = main.get_joyride(stub, start, end, 10)
    if len(ret) > 0:
        passed += 1
    print("\nPath is returned check: {} nodes returned starting from {} ending at {}.".format(len(ret), start, end))

    # check that longer time periods are longer paths
    ret2 = main.get_joyride(stub, start, end, 30)
    print("\n{} nodes returned for 30 minute long path from {} to {}.".format(len(ret2), start, end))
    long = 0
    for n in range(10, 30):
        ret1 = main.get_joyride(stub, start, end, n)
        print("\n{} nodes returned for {} minute long path from {} to {}.".format(len(ret1), n, start, end))

        if len(ret1) < len(ret2):
            long += 1

    print("\nLonger time periods return longer path check: {}/20 paths passed successfully".format(long))
    passed += long

    # check deterministic vs random
    ret1 = main.get_joyride(stub, start, end, 10)
    ret2 = main.get_joyride(stub, start, end, 10)
    print("\nPath 1 returned {} nodes. Path 2 returned {} nodes. Both paths had same input".format(len(ret1), len(ret2)))
    if ret1 == ret2:
        print("Output of algorithm is deterministic. Paths were equal")
    else:
        print("Output of algorithm was random. Paths are not equal")

    # runtime check
    runtime = 0
    
    # original start and end points
    for n in range(10, 90):
        before = datetime.datetime.now()
        ret = main.get_joyride(stub, start, end, n)
        after = datetime.datetime.now()
        t = after - before
        print("\nA joyride of {} minutes took {} seconds to calculate".format(n, t.total_seconds()))
        if t.total_seconds() <= 1:
            runtime += 1
    
    # try for various start and end points
    start = "107 Sunset Terrace, Troy NY"
    end = "99 Congress St, Troy NY"
    print("\nNew start and end locations are {} and {}.".format(start, end))
    for n in range(10, 90):
        before = datetime.datetime.now()
        ret = main.get_joyride(stub, start, end, n)
        after = datetime.datetime.now()
        t = after - before
        print("\nA joyride of {} minutes took {} seconds to calculate".format(n, t.total_seconds()))
        if t.total_seconds() <= 1:
            runtime += 1
    
    start = "1900 Peoples Ave, Troy NY"
    end = "51 South Pearl St, Albany NY"
    print("\nNew start and end locations are {} and {}.".format(start, end))
    for n in range(10, 90):
        before = datetime.datetime.now()
        ret = main.get_joyride(stub, start, end, n)
        after = datetime.datetime.now()
        t = after - before
        print("\nA joyride of {} minutes took {} seconds to calculate".format(n, t.total_seconds()))
        if t.total_seconds() <= 1:
            runtime += 1

    start = "893 Broadway, Albany NY"
    end = "277 Congress St, Troy NY"
    print("\nNew start and end locations are {} and {}.".format(start, end))
    for n in range(10, 90):
        before = datetime.datetime.now()
        ret = main.get_joyride(stub, start, end, n)
        after = datetime.datetime.now()
        t = after - before
        print("\nA joyride of {} minutes took {} seconds to calculate".format(n, t.total_seconds()))
        if t.total_seconds() <= 1:
            runtime += 1

    print("In total, {}/320 tests had a runtime under 1 second".format(runtime))
    passed += runtime

    return passed / 341

if __name__ == "__main__":
    # default ip and port
    ip = "localhost"
    port = 25565

    with grpc.insecure_channel("{}:{}".format(ip, port)) as channel:
        # create stub to run tests on
        print("Connected to gRPC server {}:{}".format(ip, port))
        stub = joyride_pb2_grpc.JoyRideStub(channel)

        # tests
        pct = testJoyride(stub)
        print("Algorithm passed {} percent of tests run".format(pct))