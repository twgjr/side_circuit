import 'package:flutter/material.dart';
import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/node.dart';
import 'widgets/circuit_view.dart';
import 'widgets/diagram_controls.dart';

void main() => runApp(MainApp());

class MainApp extends StatefulWidget {
  @override
  State<StatefulWidget> createState() => MainAppState();
}

class MainAppState extends State<MainApp> {
  final Circuit circuit = Circuit();

  void addDevice(DeviceKind kind) => setState(() => circuit.addDeviceOf(kind));
  void deleteDevice(Device device) =>
      setState(() => circuit.removeDevice(device));
  void addNode() => setState(() => circuit.newNode());
  void removeNode(Node node) => setState(() => circuit.removeNode(node));

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Side Circuit',
      home: Scaffold(
        appBar: AppBar(
          title: Text('Side Circuit'),
          actions: [DiagramControls(onAddDevice: addDevice, onAddNode: () {})],
        ),
        body: CircuitView(
          devices: circuit.devices,
          onDeleteDevice: deleteDevice,
        ),
      ),
    );
  }
}
