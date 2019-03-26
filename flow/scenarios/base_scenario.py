"""Contains the base scenario class."""

from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
import time

try:
    # Import serializable if rllab is installed
    from rllab.core.serializable import Serializable
except ImportError:
    Serializable = object


class Scenario(Serializable):
    """Base scenario class.

    Initializes a new scenario. Scenarios are used to specify features of
    a network, including the positions of nodes, properties of the edges
    and junctions connecting these nodes, properties of vehicles and
    traffic lights, and other features as well. These features can later be
    acquired  from this class via a plethora of get methods (see
    documentation).

    This class uses network specific features to generate the necessary xml
    files needed to initialize a sumo instance. The methods of this class are
    called by the base scenario class.

    The xml files can be created in one of three ways:

    * Custom networks can be generated by defining the properties of the
      network's directed graph. This is done by defining the nodes and edges
      properties using the ``specify_nodes`` and ``specify_edges`` methods,
      respectively, as well as other properties via methods including
      ``specify_types``, ``specify_connections``, etc... For more on this,
      see the tutorial on creating custom scenarios or refer to some of the
      available scenarios.

    * Scenario data can be collected from an OpenStreetMap (.osm) file. The
      .osm file is specified in the NetParams object. For example:

        >>> from flow.core.params import NetParams
        >>> net_params = NetParams(osm_path='/path/to/osm_file.osm')

      In this case, no ``specify_nodes`` and ``specify_edges`` methods are
      needed. However, a ``specify_routes`` method is still needed to specify
      the appropriate routes vehicles can traverse in the network.

    * Scenario data can be collected from an sumo-specific network (.net.xml)
      file. This file is specified in the NetParams object. For example:

        >>> from flow.core.params import NetParams
        >>> net_params = NetParams(netfile='/path/to/netfile.net.xml')

      In this case, no ``specify_nodes`` and ``specify_edges`` methods are
      needed. However, a ``specify_routes`` method is still needed to specify
      the appropriate routes vehicles can traverse in the network.

    This class can be instantiated once and reused in multiple experiments.
    Note that this function stores all the relevant parameters. The
    generate() function still needs to be called separately.
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Instantiate the base scenario class.

        Attributes
        ----------
        name : str
            A tag associated with the scenario
        vehicles : flow.core.params.VehicleParams
            see flow/core/params.py
        net_params : flow.core.params.NetParams
            see flow/core/params.py
        initial_config : flow.core.params.InitialConfig
            see flow/core/params.py
        traffic_lights : flow.core.params.TrafficLightParams
            see flow/core/params.py
        """
        # Invoke serializable if using rllab
        if Serializable is not object:
            Serializable.quick_init(self, locals())

        self.orig_name = name  # To avoid repeated concatenation upon reset
        self.name = name + time.strftime('_%Y%m%d-%H%M%S') + str(time.time())

        self.vehicles = vehicles
        self.net_params = net_params
        self.initial_config = initial_config
        self.traffic_lights = traffic_lights

        if net_params.netfile is None and net_params.osm_path is None:
            # specify the attributes of the nodes
            self.nodes = self.specify_nodes(net_params)
            # collect the attributes of each edge
            self.edges = self.specify_edges(net_params)
            # specify the types attributes (default is None)
            self.types = self.specify_types(net_params)
            # specify the connection attributes (default is None)
            self.connections = self.specify_connections(net_params)
        else:
            self.nodes = None
            self.edges = None
            self.types = None
            self.connections = None

        # specify routes vehicles can take
        self.routes = self.specify_routes(net_params)

        # optional parameters, used to get positions from some global reference
        self.edge_starts = self.specify_edge_starts()
        self.internal_edge_starts = self.specify_internal_edge_starts()
        self.intersection_edge_starts = []  # this will be deprecated

    # TODO: convert to property
    def specify_edge_starts(self):
        """Define edge starts for road sections in the network.

        This is meant to provide some global reference frame for the road
        edges in the network.

        By default, the edge starts are specified from the network
        configuration file. Note that, the values are arbitrary but do not
        allow the positions of any two edges to overlap, thereby making them
        compatible with all starting position methods for vehicles.

        Returns
        -------
        list of (str, float)
            list of edge names and starting positions,
            ex: [(edge0, pos0), (edge1, pos1), ...]
        """
        return None

    # TODO: convert to property
    def specify_internal_edge_starts(self):
        """Define the edge starts for internal edge nodes.

        This is meant to provide some global reference frame for the internal
        edges in the network.

        These edges are the result of finite-length connections between road
        sections. This methods does not need to be specified if "no-internal-
        links" is set to True in net_params.

        By default, all internal edge starts are given a position of -1. This
        may be overridden; however, in general we do not worry about internal
        edges and junctions in large networks.

        Returns
        -------
        list of (str, float)
            list of internal junction names and starting positions,
            ex: [(internal0, pos0), (internal1, pos1), ...]
        """
        return [(':', -1)]

    # TODO: convert to property
    def specify_nodes(self, net_params):
        """Specify the attributes of nodes in the network.

        Parameters
        ----------
        net_params : flow.core.params.NetParams
            see flow/core/params.py

        Returns
        -------
        list of dict

            A list of node attributes (a separate dict for each node). Nodes
            attributes must include:

            * id {string} -- name of the node
            * x {float} -- x coordinate of the node
            * y {float} -- y coordinate of the node

        Other attributes may also be specified. See:
        http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Node_Descriptions
        """
        raise NotImplementedError

    # TODO: convert to property
    def specify_edges(self, net_params):
        """Specify the attributes of edges connecting pairs on nodes.

        Parameters
        ----------
        net_params : flow.core.params.NetParams
            see flow/core/params.py

        Returns
        -------
        list of dict

            A list of edges attributes (a separate dict for each edge). Edge
            attributes must include:

            * id {string} -- name of the edge
            * from {string} -- name of node the directed edge starts from
            * to {string} -- name of the node the directed edge ends at

            In addition, the attributes must contain at least one of the
            following:

            * "numLanes" {int} and "speed" {float} -- the number of lanes and
              speed limit of the edge, respectively
            * type {string} -- a type identifier for the edge, which can be
              used if several edges are supposed to possess the same number of
              lanes, speed limits, etc...

        Other attributes may also be specified. See:
        http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Edge_Descriptions
        """
        raise NotImplementedError

    # TODO: convert to property
    def specify_types(self, net_params):
        """Specify the attributes of various edge types (if any exist).

        Parameters
        ----------
        net_params : flow.core.params.NetParams
            see flow/core/params.py

        Returns
        -------
        list of dict
            A list of type attributes for specific groups of edges. If none are
            specified, no .typ.xml file is created.

        For information on type attributes, see:
        http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Type_Descriptions
        """
        return None

    # TODO: convert to property
    def specify_connections(self, net_params):
        """Specify the attributes of connections.

        These attributes are used to describe how any specific node's incoming
        and outgoing edges/lane pairs are connected. If no connections are
        specified, sumo generates default connections.

        Parameters
        ----------
        net_params : flow.core.params.NetParams
            see flow/core/params.py

        Returns
        -------
        list of dict
            A list of connection attributes. If none are specified, no .con.xml
            file is created.

        For information on type attributes, see:
        http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Connection_Descriptions
        """
        return None

    # TODO: convert to property
    def specify_routes(self, net_params):
        """Specify the routes vehicles can take starting from any edge.

        The routes are specified as lists of edges the vehicle must traverse,
        with the first edge corresponding to the edge the vehicle begins on.
        Note that the edges must be connected for the route to be valid.

        Currently, only one route is allowed from any given starting edge.

        Parameters
        ----------
        net_params : flow.core.params.NetParams
            see flow/core/params.py

        Returns
        -------
        dict
            Key = name of the starting edge
            Element = list of edges a vehicle starting from this edge must
            traverse.
        """
        raise NotImplementedError

    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        """Generate a user defined set of starting positions.

        Parameters
        ----------
        cls : flow.core.kernel.scenario.KernelScenario
            flow scenario kernel, with all the relevant methods implemented
        net_params : flow.core.params.NetParams
            network-specific parameters
        initial_config : flow.core.params.InitialConfig
            see flow/core/params.py
        num_vehicles : int
            number of vehicles to be placed on the network

        Returns
        -------
        list of tuple (float, float)
            list of start positions [(edge0, pos0), (edge1, pos1), ...]
        list of int
            list of start lanes
        list of float
            list of start speeds
        """
        raise NotImplementedError

    def __str__(self):
        """Return the name of the scenario and the number of vehicles."""
        return 'Scenario ' + self.name + ' with ' + \
               str(self.vehicles.num_vehicles) + ' vehicles.'
