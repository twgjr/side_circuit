# system
from modules.system import *
from modules.elements import *

# spice simulator
import PySpice.Logging.Logging as Logging

logger = Logging.setup_logging()
from PySpice.Spice.Netlist import Circuit as PySpiceCircuit
from PySpice.Unit import *


class Simulator:
    def __init__(self, system: System) -> None:
        self.system = system

    def system_to_pyspice(self, title: str) -> PySpiceCircuit:
        self.system.check_complete()
        circuit = PySpiceCircuit(title=title)
        for node in self.system.nodes:
            if isinstance(node, System):
                continue
            if isinstance(node, CircuitNode):
                continue
            if isinstance(node, Voltage):
                if isinstance(node.config, DC):
                    circuit.V(
                        node.deep_id(),
                        node.p.neighbor(),
                        node.n.neighbor(),
                        node.config.value,
                    )
                elif isinstance(node.config, Pulse):
                    circuit.PulseVoltageSource(
                        node.deep_id(),
                        node.p.neighbor(),
                        node.n.neighbor(),
                        initial_value=node.config.initial_value,
                        pulsed_value=node.config.pulsed_value,
                        pulse_width=node.config.duty / node.config.freq,
                        period=1 / node.config.freq,
                    )
            elif isinstance(node, Resistor):
                circuit.R(
                    node.deep_id(), node.p.neighbor(), node.n.neighbor(), node.value
                )
            elif isinstance(node, Capacitor):
                circuit.C(
                    node.deep_id(), node.p.neighbor(), node.n.neighbor(), node.value
                )
            elif isinstance(node, Inductor):
                circuit.L(
                    node.deep_id(), node.p.neighbor(), node.n.neighbor(), node.value
                )
            else:
                raise NotImplementedError(f"Node {node} not implemented")
        return circuit
