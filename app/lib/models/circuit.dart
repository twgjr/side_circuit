import 'package:app/models/device.dart';
import 'package:app/models/node.dart';
import 'package:app/models/terminal.dart';
import 'package:app/library/device_library.dart';
import 'package:app/models/vertex.dart';
import 'package:app/models/net.dart';
import 'package:app/models/wire.dart';
import 'package:app/models/segment.dart';
import 'package:flutter/material.dart';

enum Quantity { I, V, P }

class Circuit {
  List<Device> devices = [];
  List<Net> nets = [];

  Circuit();

  bool isEmpty() => devices.isEmpty && nets.isEmpty;

  void _addDevice(Device element) => devices.add(element);
  void addDeviceOf(DeviceKind kind) {
    switch (kind) {
      case DeviceKind.V:
        _addDevice(IndependentSource(kind: DeviceKind.V));
        break;
      case DeviceKind.BLOCK:
        _addDevice(IndependentSource(kind: DeviceKind.BLOCK));
        break;
      case DeviceKind.R:
        _addDevice(Resistor());
        break;
      default:
        throw ArgumentError("Invalid device kind: $kind");
    }
  }

  void removeDevice(Device element) => devices.remove(element);
  void removeDeviceAt(int index) => devices.removeAt(index);

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

  Circuit copy() {
    final newCircuit = Circuit();
    newCircuit.devices = [];
    for (Device device in devices) {
      newCircuit.devices.add(device);
    }
    newCircuit.nets = [];
    for (Net net in nets) {
      newCircuit.nets.add(net);
    }
    return newCircuit;
  }

  void replaceDeviceWith(Device newDevice, int atIndex) =>
      devices[atIndex] = newDevice;

  List<Terminal> terminals() {
    List<Terminal> terminals = [];
    for (Device device in devices) {
      terminals.addAll(device.terminals);
    }
    return terminals;
  }

  List<Segment> segments() {
    List<Segment> segments = [];
    for (Net net in nets) {
      for (Wire wire in net.wires) {
        segments.addAll(wire.segments);
      }
    }
    return segments;
  }

  List<Vertex> vertices() {
    List<Vertex> vertices = [];
    for (Net net in nets) {
      for (Wire wire in net.wires) {
        vertices.addAll(wire.vertices);
      }
    }
    return vertices;
  }

  List<Node> nodes() {
    List<Node> nodes = [];
    for (Net net in nets) {
      nodes.addAll(net.nodes);
    }
    return nodes;
  }

  Wire startWire({Terminal? terminal, Offset? position}) {
    final net = Net();
    nets.add(net);
    return net.startWire(terminal: terminal, position: position);
  }

  void endWire(Wire wire, {Terminal? terminal}) {
    // assert(nets.contains(wire.net)); // assumes shallow circuit copies
    wire.end(terminal: terminal);
  }
}
