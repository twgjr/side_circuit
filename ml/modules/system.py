from common import CommonObject, Wire, Junction, Junction, Node
from elements import *


class System(CommonObject):
    """create a system by defining wires between nodes
    and elements and subsystems in the constructor.
    """

    def __init__(
        self,
        idx: str = "",
        interfaces: list[str] = [],
        objects: list[CommonObject] = [],
    ) -> None:
        super().__init__(idx)
        self.__node_dict: dict[str, Node] = {}
        self.__ext_interfaces: dict[str, Interface] = {}
        self.__int_interfaces: dict[str, Interface] = {}
        for interface in interfaces:
            self.__ext_interfaces[interface] = Interface(interface)
            self.__ext_interfaces[interface].parent = self
            self.__int_interfaces[interface] = Interface(interface)
            self.__int_interfaces[interface].parent = self
        self.__v_dict: dict[str, V] = {}
        self.__r_dict: dict[str, R] = {}
        self.__x_dict: dict[str, "System"] = {}
        for obj in objects:
            obj.parent = self
            if isinstance(obj, V):
                self.__v_dict[str(obj)] = obj
            elif isinstance(obj, R):
                self.__r_dict[str(obj)] = obj
            elif isinstance(obj, System):
                self.__x_dict[str(obj)] = obj
            else:
                raise TypeError(f"invalid object type {type(obj)}")
        self.__wires: dict[str, Wire] = {}

    # region magic methods
    def __contains__(self, item) -> bool:
        if isinstance(item, Junction):
            return item in self.__ext_interfaces or item in self.__int_interfaces

        if isinstance(item, V):
            return item in self.__v_dict

        if isinstance(item, R):
            return item in self.__r_dict

        if isinstance(item, System):
            return item in self.__x_dict

        if isinstance(item, Wire):
            return item in self.__wires

        return False

    def __call__(self, idx: str) -> "System":
        system_copy = self.copy()
        system_copy.idx = idx
        return system_copy

    # endregion

    # region properties
    @property
    def X(self) -> dict[str, "System"]:
        return self.__x_dict

    @property
    def V(self) -> dict[str, V]:
        return self.__v_dict

    @property
    def R(self) -> dict[str, R]:
        return self.__r_dict

    @property
    def Nodes(self) -> dict[str, Node]:
        return self.__node_dict

    @property
    def Wires(self) -> dict[str, Wire]:
        return self.__wires
    
    @property
    def ei(self) -> dict[str, Interface]:
        """external interfaces"""
        return self.__ext_interfaces

    @property
    def ii(self) -> dict[str, Interface]:
        """internal interfaces"""
        return self.__int_interfaces

    # endregion

    # region public methods
    def num_subsystems(self) -> int:
        return len(self.__x_dict)

    def num_v(self) -> int:
        return len(self.__v_dict)

    def num_r(self) -> int:
        return len(self.__r_dict)

    def num_wires(self) -> int:
        return len(self.__wires)

    def num_nodes(self) -> int:
        return len(self.__node_dict)

    def num_interfaces(self) -> int:
        if len(self.__ext_interfaces) != len(self.__int_interfaces):
            raise ValueError(f"number of external and internal interfaces do not match")
        return len(self.__ext_interfaces)

    def copy(self) -> "System":
        elements = []
        for v in self.__v_dict.values():
            elements.append(v.copy())
        for r in self.__r_dict.values():
            elements.append(r.copy())
        sys = System(self.idx, list(self.__ext_interfaces.keys()), elements)
        for key, node in self.__node_dict.items():
            sys.__node_dict[key] = node.copy()
        sys.__copy_wires(self.__wires)
        return sys

    def link(self, hi: Interface, lo: Interface) -> None:
        """controls all wire creation"""
        hi_nodes = hi.nodes()
        lo_nodes = lo.nodes()

        if len(hi_nodes) == 0 and len(lo_nodes) == 0:
            if len(hi.neighbors()) > 0 or len(lo.neighbors()) > 0:
                # connect hi, lo and their neighbors to a common node
                node = self.__add_node()
                self.__add_wire_between(hi, node)
                self.__add_wire_between(lo, node)
                for neighbor in hi.neighbors():
                    if neighbor is not node:
                        self.__add_wire_between(node, neighbor)
                for neighbor in lo.neighbors():
                    if neighbor is not node:
                        self.__add_wire_between(node, neighbor)
                return

            # connect them directly
            self.__add_wire_between(hi, lo)
            return

        if len(hi_nodes) == 0 and len(lo_nodes) == 1:
            # connect hi to lo's node
            self.__add_wire_between(hi, lo_nodes[0])
            return

        if len(hi_nodes) == 1 and len(lo_nodes) == 0:
            # connect lo to hi's node
            self.__add_wire_between(lo, hi_nodes[0])
            return

        if len(hi_nodes) == 1 and len(lo_nodes) == 1:
            # merge the nodes
            hi_node = hi_nodes[0]
            lo_node = lo_nodes[0]
            # merge hi_node into lo_node
            for neighbor in hi_node.neighbors():
                if neighbor is lo_node:
                    raise ValueError(f"{hi} connected to two nodes")
                self.__remove_wire_between(hi_node, neighbor)
                self.__remove_node(hi_node)
                self.__add_wire_between(neighbor, lo_node)
            return

        raise ValueError(f"Error connecting {hi} to {lo}")

    # endregion

    # region private methods
    def __get_matching(self, wire: Wire, junction: Junction) -> Junction:
        """Retrieve the interface or node from this system that has a matching key"""

        parent = junction.parent

        if isinstance(junction, Node):
            return self.Nodes[str(junction)]
        
        if isinstance(parent, System):
            super_system = wire.parent
            if isinstance(super_system, System):
                if super_system is junction.parent:
                    return self.ii[str(junction)]
                else:
                    return self.ei[str(junction)]

        if isinstance(parent, V):
            v = self.V[str(parent)]
            if isinstance(v, V):
                return v[str(junction)]

        if isinstance(parent, R):
            r = self.R[str(parent)]
            if isinstance(r, R):
                return r[str(junction)]

        raise TypeError(f"invalid interface {junction}")

    def __add_node(self) -> Node:
        node = Node(f"node{self.num_nodes()}")
        self.__node_dict[str(node)] = node
        node.parent = self
        return node

    def __remove_node(self, node: Node) -> None:
        del self.__node_dict[str(node)]

    def __add_wire(self, wire: Wire) -> None:
        self.__wires[str(wire)] = wire
        wire.parent = self

    def __get_wire_between(self, hi: Junction, lo: Junction) -> Wire:
        return self.__wires[Wire.make_idx(hi, lo)]

    def __remove_wire_between(self, hi: Junction, lo: Junction) -> None:
        wire = self.__get_wire_between(hi, lo)
        del self.__wires[str(wire)]
        hi.remove_wire(wire)
        lo.remove_wire(wire)

    def __add_wire_between(self, hi: Junction, lo: Junction) -> None:
        wire = Wire(hi, lo)
        hi.add_wire(wire)
        lo.add_wire(wire)
        self.__add_wire(wire)

    def __copy_wires(self, wires: dict[str, Wire]) -> None:
        for key, wire in wires.items():
            # get interfaces from this system matching other system itx ids
            hi = self.__get_matching(wire, wire.hi)
            lo = self.__get_matching(wire, wire.lo)
            # create new wire
            self.__add_wire_between(hi, lo)

    # endregion
