import random
from system import System
from elements import (
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
        # loop through the number of elements and choose a random element
        for _ in range(num_elements):
            element = self.choose_random_element()
            self.elements.append(element)
            
        for element in self.elements:            
            # build temporary list of ports to choose from
            ports_to_select = []
            for element in self.elements:
                ports_to_select.extend(list(element.ports.values()))
            
            # loop through the ports of the element and attempt to connect them
            for port_key in element.ports.keys():
                to_port = element.ports[port_key]
                if len(ports_to_select) == 0 or random.random() < 0.2:
                    # make the element port a sub-system port
                    self.ports[port_key] = to_port
                else:
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
