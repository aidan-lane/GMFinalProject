import argparse
from concurrent import futures
from multiprocessing import Manager
import logging
import time
import sys

from decouple import config
import grpc
import networkx as nx
import osmnx as ox
import psycopg2

from gen import joyride_pb2
from gen import joyride_pb2_grpc

from networkx import NetworkXNoPath


# Osmnx config options
ox.config(use_cache=True, log_console=False)

conn = psycopg2.connect(user=config("POSTGRES_USER"),
                              password=config("POSTGRES_PASSWORD"),
                              host=config("POSTGRES_HOST"),
                              port="5432",
                              database=config("POSTGRES_DB"))
cursor = conn.cursor()

add_node_query = """
    INSERT INTO personalization (node, interest, rating)
    VALUES (%s, %s, %s)
    ON CONFLICT (node)
    DO UPDATE SET interest = EXCLUDED.interest + 1;
"""

add_rating_query = """
    INSERT INTO personalization (node, interest, rating)
    VALUES (%s, %s, %s)
    ON CONFLICT (node)
    DO UPDATE SET rating = EXCLUDED.rating + 1;
"""

get_nodes_query = """
    SELECT * FROM personalization;
"""

###########
# Helpers #
###########

def add_node_interest(batch, P):
    """ Helper function to add a batch of (node, interest) rows to dictionary
        for page-rank.
    """
    for row in batch:
        P[row[0]] = row[1]

def incr_dict(map, key):
    """ Helper to increment a node's interest value in a dictionary
    """
    if key not in map:
        map[key] = 0
    map[key] += 1


class Joyride(joyride_pb2_grpc.JoyRideServicer):
    """ Servicer for the JoyRide gRPC service.
    """

    def __init__(self, graph, P, R):
        self.G = graph
        # P is a personalization dictionary for augmenting our page-rank algorithm.
        # When a user queries for a destination, the node's interest score is incremented
        # and saved to the database. It is not guaranteed that the dictionary will be
        # completely loaded once queries can be received for the sake of time.
        self.P = P
        self.R = R

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

        # Add start and end node data point to database (Point-of-interest)
        cursor.executemany(add_node_query, [(start_node, 1, 0), (end_node, 1, 0)])
        conn.commit()
        incr_dict(self.P, start_node)
        incr_dict(self.P, end_node)

        # Find shortest path weighted on pre-computed travel time
        total_time, route = nx.bidirectional_dijkstra(G, start_node, end_node, weight="travel_time")

        target_time = request.time
        current_time = total_time
        
        
        time_margin = target_time * 0.05
        step = 0
        eps = 100
        print("wtf")

        while current_time < target_time - time_margin and step < eps:
            next_route = []
            last_node = start_node
            for i in range(1, len(route) - 1,2):
                if current_time >= target_time - time_margin:
                    break

                node = route[i]
                node_attrib = G.nodes[node]
                edges = list(G.edges(node, data=True))

                left_node = route[i - 1]
                right_node = route[i + 1]
                if(left_node == right_node):
                    continue
                left_edge = G.get_edge_data(last_node, node)
                right_edge = G.get_edge_data(node, right_node)

                G.remove_node(node)
                found_path = True

                try:
                    new_time, new_route = nx.bidirectional_dijkstra(G, last_node, right_node, weight="travel_time")
                except NetworkXNoPath:
                    found_path = False

                G.add_nodes_from([(node, node_attrib)])
                G.add_edges_from(edges)

                if found_path and len(new_route) > 2 and new_time + current_time < target_time + time_margin:
                    last_node = new_route[-1]

                    if not left_edge:
                        pass
                    else:
                        current_time -= left_edge[0]["travel_time"]
                    if not right_edge:
                        pass
                    else:
                        current_time -= right_edge[0]["travel_time"]
                    current_time += new_time
                    next_route.extend(new_route[:-1])
                else:
                    next_route.append(left_node)
                    if not next_route:
                        next_route.append(last_node)
                    last_node = right_node
            route = next_route
            step += 1
            print("Current Time:",current_time)
            
                
        # Generate directions and node data and yield to gRPC client
        last_name = None
        last_bearing = None        # Generate directions and node data and yield to gRPC client
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
    
    #def UpdateRating()


def serve(port, graph):
    """ Starts gRPC server instance

    Starts the defined service class on the specified port, using data from the given
    graph.

    Args:
        port: port number as string
        graph: MultiDigraph to be queried by this service
    """

    P = Manager().dict()  # Personalization dictionary for page-rank
    max_workers = 20
    batch_size = 1000

    # Load personalization table into in-memory dictionary.
    # We can do this in parallel as the key, a node is guaranteed to be unique
    # by the table scheme (i.e. is a primary key).
    with futures.ProcessPoolExecutor(max_workers) as executor:
        cursor.execute(get_nodes_query)
        tasks = []
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            tasks.append(executor.submit(add_node_interest, rows, P))
        
        for t in tasks:
            t.result()

    # Do page-rank
    ranks = nx.pagerank(graph)
    print("Completed Page-Rank")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    joyride_pb2_grpc.add_JoyRideServicer_to_server(Joyride(graph, P, ranks), server)
    server.add_insecure_port("[::]:{}".format(port))

    server.start()
    print("Started server on port {}".format(port))
    server.wait_for_termination()
    conn.close()


def load_data(address, r):
    """ Loads initial road network data

    Since we are dealing with a large amount of data from all over the world, we must
    specify a subset of this entire road network graph so that this application remains
    viable.

    Args:
        address: central point of graph
        r: radius in meters around address to be loaded

    Returns:
        NetworkX.MultiDiGraph

    """
    start_time = time.time()
    G = ox.graph_from_address(address, dist=r, network_type="drive", simplify=True)
    print("Finished loading graph in {:.4f} seconds.".format(time.time() - start_time))

    # Calculate travel time based on length, speed (km/h) for each edge
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    G = ox.add_edge_bearings(G)
    print()

    return G


def bearing_turn(b1, b2):
    """ Calculates turn direction between two bearings

    Bearings represents degrees on a compass (0-359). In order to determine if a turn is
    left or right, the angle between b1 and b2 is calculated to determine the direction.

    Args:
        b1: bearing 1
        b2: bearing 2

    Returns:
        A string, ' left ' or ' right '
    """
    if ((((b1 - b2 + 540) % 360) - 180) > 0):
      return " left "
    else:
      return " right "


def cardinal_direction(b):
    """ Calculate the cardinal direction for a given bearing.

    Ex: 0 is 'North

    Args:
        b: bearing (in degrees)

    Returns:
        A string representing cardinal direction
    """
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
