import random
from app.system import System
from app.element import (
    Element,
    Voltage,
    Resistor,
    Current,
    Capacitor,
    Inductor,
    VoltageControlledSwitch,
    CurrentControlledSwitch,
    VoltageControlledVoltageSource,
    CurrentControlledVoltageSource,
    CurrentControlledCurrentSource,
    VoltageControlledCurrentSource,
)


class RandomSubSystem(System):
    def __init__(self, max_elements: int = 6) -> None:
        super().__init__()
        self.generate(max_elements)

    def generate(self, max_elements: int) -> None:
        # choose a random number of elements to add to the system
        num_elements = random.randint(1, max_elements)

        # randomly choose elements to create
        for _ in range(num_elements):
            element = self.choose_random_element()
            self.add_element(element)

        # connect the elements together
        for element in self.elements:
            # build temporary list of ports from elements already in the sub-system
            ports_to_select = []
            for element in self.elements:
                ports_to_select.extend(list(element.terminals.values()))

            # loop through the ports of the element and attempt to connect them
            for port_key in element.terminals.keys():
                to_port = element.terminals[port_key]
                # choose a random port to connect to from the list of ports
                from_port = random.choice(ports_to_select)
                self.connect(from_port, to_port)
                ports_to_select.remove(from_port)

    def choose_random_element(self) -> Element:
        # choose a random element from the list of elements
        element = random.choice(
            [
                Voltage,
                Resistor,
                Current,
                Capacitor,
                Inductor,
                VoltageControlledSwitch,
                CurrentControlledSwitch,
                VoltageControlledVoltageSource,
                CurrentControlledVoltageSource,
                CurrentControlledCurrentSource,
                VoltageControlledCurrentSource,
            ]
        )
        # instantiate the element
        return element()
