import 'package:app/models/circuit/device.dart';
import 'package:app/models/visual/node.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/library/device_library.dart';
import 'package:app/models/visual/vertex.dart';
import 'package:app/models/circuit/net.dart';
import 'package:app/models/visual/wire.dart';
import 'package:app/models/visual/segment.dart';
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
    newCircuit.devices = devices;
    newCircuit.nets = nets;
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

  // Wire startWire({Terminal? terminal, Offset? position}) {
  //   final net = Net();
  //   nets.add(net);
  //   return net.startWire(terminal: terminal, position: position);
  // }
}
