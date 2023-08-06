/*
CIRCUITS is a library for creating and analyzing electrical circuits.  It primarily
follows the same structure as the SPICE circuit simulator.  It can access the ngspice 
API.
*/

import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/node.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/models/circuit/wire.dart';

enum Quantity { I, V, P }

class Circuit {
  List<Node> nodes = [];
  List<Device> devices = [];
  List<Wire> wires = [];

  Circuit();

  bool isEmpty() => devices.isEmpty && nodes.isEmpty;

  void _addDevice(Device element) => devices.add(element);
  void addDeviceOf(DeviceKind kind) {
    switch (kind) {
      case DeviceKind.V:
        _addDevice(IndependentSource(circuit: this, kind: DeviceKind.V));
        break;
      case DeviceKind.I:
        _addDevice(IndependentSource(circuit: this, kind: DeviceKind.I));
        break;
      case DeviceKind.R:
        _addDevice(Resistor(this));
        break;
      default:
        throw ArgumentError("Invalid device kind: $kind");
    }
  }

  void removeDevice(Device element) => devices.remove(element);
  void removeDeviceAt(int index) => devices.removeAt(index);
  void _addNode(Node node) => nodes.add(node);
  void newNode() => _addNode(Node(this));
  void removeNode(Node node) => nodes.remove(node);
  void removeNodeAt(int index) => nodes.removeAt(index);
  void addTerminal(Device device) => device.terminals.add(Terminal(device, ""));
  void removeTerminal(Terminal terminal) {
    terminal.device.terminals.remove(terminal);
    if (terminal.node != null) {
      terminal.node?.removeTerminal(terminal);
    }
  }

  int numNodes() => nodes.length;
  int numDevices() => devices.length;

  int maxDeviceIdOf(DeviceKind kind) {
    int maxId = 0;
    for (Device device in devices) {
      if (device.kind == kind && device.id > maxId) {
        maxId = device.id;
      }
    }
    return maxId;
  }

  int maxNodeId() {
    int maxId = 0;
    for (Node node in nodes) {
      if (node.id > maxId) {
        maxId = node.id;
      }
    }
    return maxId;
  }

  void connect(Terminal terminal, Node node) {
    terminal.node = node;
    node.addTerminal(terminal);
    wires.add(Wire(terminal, node));
  }

  void disconnect(Terminal terminal) {
    terminal.node?.removeTerminal(terminal);
    terminal.node = null;
    wires.removeWhere((wire) => wire.terminal == terminal);
  }

  Circuit copy() {
    Circuit newCircuit = Circuit();
    for (Node node in nodes) {
      node.circuit = newCircuit;
      newCircuit._addNode(node);
    }
    for (Device device in devices) {
      device.circuit = newCircuit;
      newCircuit._addDevice(device);
    }
    return newCircuit;
  }
}
