/*
CIRCUITS is a library for creating and analyzing electrical circuits.  It primarily
follows the same structure as the SPICE circuit simulator.  It can access the ngspice 
API.
*/

import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/node.dart';
import 'package:app/models/circuit/terminal.dart';

enum Quantity { I, V, P }

class Circuit {
  List<Node> nodes = [];
  List<Device> devices = [];

  Circuit();

  bool isEmpty() => devices.isEmpty && nodes.isEmpty;

  void _addElement(Device element) => devices.add(element);
  void addDeviceOf(DeviceKind kind) {
    switch (kind) {
      case DeviceKind.V:
        _addElement(IndependentSource(circuit: this, kind: DeviceKind.V));
        break;
      case DeviceKind.I:
        _addElement(IndependentSource(circuit: this, kind: DeviceKind.I));
        break;
      case DeviceKind.R:
        _addElement(Resistor(this));
        break;
      default:
        throw ArgumentError("Invalid element kind: $kind");
    }
  }

  void removeDevice(Device element) => devices.remove(element);
  void removeDeviceAt(int index) => devices.removeAt(index);
  void _addNode(Node node) => nodes.add(node);
  void newNode() => _addNode(Node(this));
  void removeNode(Node node) => nodes.remove(node);
  void removeNodeAt(int index) => nodes.removeAt(index);
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
  }

  Circuit copy() {
    Circuit newCircuit = Circuit();
    for (Node node in nodes) {
      node.circuit = newCircuit;
      newCircuit._addNode(node);
    }
    for (Device device in devices) {
      device.circuit = newCircuit;
      newCircuit._addElement(device);
    }
    return newCircuit;
  }
}
