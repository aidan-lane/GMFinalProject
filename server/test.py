import argparse
from ast import Mult
from concurrent import futures
from multiprocessing import Manager
import threading
import logging
import math
import time

from decouple import config
import grpc
import networkx as nx
import osmnx as ox
from osmnx import plot
import psycopg2
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from gen import joyride_pb2
from gen import joyride_pb2_grpc

from networkx import NetworkXNoPath

import main
import geopy
import random
from shapely.geometry import MultiPoint

def testGet_center():
    passed = 0

    # 2 points
    points = []
    for _ in range(2):
        a = random.randint(-90, 90)
        b = random.randint(-180, 180)
        points.append(a, b)
    ptobj = MultiPoint(points)
    gt = ptobj.centroid
    ret = main.get_center(points)
    print("Input: {}".format(points))
    print("True centroid: {}, function return: {}".format(gt, ret))
    if gt == ret:
        passed += 1

    # 3 points
    points = []
    for _ in range(3):
        a = random.randint(-90, 90)
        b = random.randint(-180, 180)
        points.append(a, b)
    ptobj = MultiPoint(points)
    gt = ptobj.centroid
    ret = main.get_center(points)
    print("Input: {}".format(points))
    print("True centroid: {}, function return: {}".format(gt, ret))
    if gt == ret:
        passed += 1

    # 4 points
    points = []
    for _ in range(4):
        a = random.randint(-90, 90)
        b = random.randint(-180, 180)
        points.append(a, b)
    ptobj = MultiPoint(points)
    gt = ptobj.centroid
    ret = main.get_center(points)
    print("Input: {}".format(points))
    print("True centroid: {}, function return: {}".format(gt, ret))
    if gt == ret:
        passed += 1

    # 5 points
    points = []
    for _ in range(5):
        a = random.randint(-90, 90)
        b = random.randint(-180, 180)
        points.append(a, b)
    ptobj = MultiPoint(points)
    gt = ptobj.centroid
    ret = main.get_center(points)
    print("Input: {}".format(points))
    print("True centroid: {}, function return: {}".format(gt, ret))
    if gt == ret:
        passed += 1

    return passed / 4

def testGet_distance():
    # ground truth distance
    gt = [[[[geopy.distance.geodesic((a, b), (c, d)).km for d in range(-180, 180, 8)] for c in range(-80, 80, 8)] for b in range(-180, 180, 8)] for a in range(-80, 80, 8)]

    # use larger step with adjusted bounds so we run less iterations
    res = [[[[0 for _ in range(-180, 180, 8)] for _ in range(-80, 80, 8)] for _ in range(-180, 180, 8)] for _ in range(-80, 80, 8)]
    for a in range(-80, 80, 8):
        for b in range(-180, 180, 8):
            for c in range(-80, 80, 8):
                for d in range(-180, 180, 8):
                    ret = main.get_distance((a, b), (c, d))
                    res[(a + 80) // 8][(b + 180) // 8][(c + 80) // 8][(d + 180) // 8] = ret

    if res == gt:
        print("All returned results correspond to ground truth distances.")
        pct = 1
    else:
        print("Error in get_distance function. Not all results correspond to ground truth.")
        passed = 0
        for a in range(len(res)):
            for b in range(len(res[a])):
                for c in range(len(res[a][b])):
                    for d in range(len(res[a][b][c])):
                        if gt[a][b][c][d] == res[a][b][c][d]:
                            passed += 1
                        else:
                            print("Input: ({}, {}), ({}, {})".format(a * 8 + 80, b * 8 + 180, c * 8 + 80, d * 8 + 180))
                            print("True distance: {}, function return: {}".format(gt[a][b][c][d], res[a][b][c][d]))
        pct = passed / (len(res) ** 2 * len(res[a]) ** 2)

    return pct

def testBearing_turn():
    # ground truth turn
    gt = [[" right " if 0 <= j - i <= 180 else " left " for j in range(360)] for i in range(360)]

    # all possibilities
    res = [[0 for _ in range(360)] for _ in range(360)]
    for i in range(360):
        for j in range(360):
            ret = main.bearing_turn(i, j)
            res[i][j] = ret

    if res == gt:
        print("All returned results correspond to ground truth turns.")
        pct = 1
    else:
        print("Error in bearing_turn function. Not all results correspond to ground truth.")
        passed = 0
        for i in range(len(res)):
            for j in range(len(res[i])):
                if gt[i] == res[i]:
                    passed += 1
                else:
                    print("Input: {}, {}.".format(i, j))
                    print("True turn: {}, function return: {}".format(gt[i][j], res[i][j]))
        pct = passed / (len(res) ** 2)

    return pct

def testCardinal_direction():
    # list of degrees and ground truth directions that correspond to each other
    degs = [d for d in range(0, 360, 45)]
    gt = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]
    # all directions
    res = []
    for n in degs:
        print("Testing cardinal direction for {} degrees.".format(n))
        ret = main.cardinal_direction(n)
        print("Direction returned: {}".format(ret))
        res.append(ret)

    if res == gt:
        print("All returned results correspond to ground truth directions.")
        pct = 1
    else:
        print("Error in cardinal_direction function. Not all results correspond to ground truth.")
        passed = 0
        for i in range(len(res)):
            print("Input: {}".format(degs[i]))
            print("True direction: {}, function return: {}".format(gt[i], res[i]))
            if gt[i] == res[i]:
                passed += 1
        pct = passed / len(res)

    return pct   
    

if __name__ == "__main__":
    # tests
    pcts = []
    pct = testCardinal_direction()
    print("Cardinal direction passed {} percentage of its tests.".format(pct))
    pcts.append(pct)
    pct = testBearing_turn()
    print("Bearing turn passed {} percentage of its tests.".format(pct))
    pcts.append(pct)
    pct = testGet_distance()
    print("Get distance passed {} percentage of its tests.".format(pct))
    pcts.append(pct)
    pct = testGet_center()
    print("Get center passed {} percentage of its tests.".format(pct))
    pcts.append(pct)
    print("Overall average test success: {}".format(sum(pcts) / len(pcts)))