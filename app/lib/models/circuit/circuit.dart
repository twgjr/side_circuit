import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/node.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/library/device_library.dart';
import 'package:app/models/visual/wire.dart';
import 'package:flutter/material.dart';

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
      case DeviceKind.BLOCK:
        _addDevice(IndependentSource(circuit: this, kind: DeviceKind.BLOCK));
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
  Wire startWireAt(int terminalIndex, int deviceIndex) {
    final terminal = devices[deviceIndex].terminals[terminalIndex];
    Wire wire = Wire(terminal: terminal);
    terminal.wire = wire;
    wires.add(wire);
    return wire;
  }

  void addVertexToWireAt(Offset offset, double angleIncrement, Wire wire) {
    wire.addVertexAt(offset, angleIncrement);
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

  Circuit copy({required bool deep}) {
    final newCircuit = Circuit();
    if (deep) {
      for (Device device in devices) {
        final newDevice = device.copyWith(circuit: newCircuit);
        newCircuit.devices.add(newDevice);
      }
      for (Node node in nodes) {
        final newNode = node.copyWith(circuit: newCircuit);
        newCircuit.nodes.add(newNode);
      }
    } else {
      newCircuit.devices = devices;
      newCircuit.nodes = nodes;
    }
    return newCircuit;
  }

  void replaceDeviceWith(Device newDevice, int atIndex) {
    for (Terminal terminal in newDevice.terminals) {
      terminal.wire = null;
    }
    devices[atIndex] = newDevice;
  }
}
