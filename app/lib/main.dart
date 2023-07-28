import 'package:flutter/material.dart';
import 'models/circuit/circuit.dart';
import 'widgets/diagram/diagram.dart';
import 'widgets/diagram/diagram_controls.dart';
import 'models/circuit/device.dart';
import 'models/circuit/node.dart';
import 'models/view/data_obj.dart';

void main() => runApp(MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Side Circuit',
      home: MainApp(),
    ));

class MainApp extends StatefulWidget {
  @override
  State<StatefulWidget> createState() => _MainApp();
}

class _MainApp extends State<MainApp> {
  final Circuit circuit = Circuit();
  List<DataObj> dataObjs = [];

  @override
  void initState() {
    for (Device device in circuit.devices) {
      dataObjs.add(DataObj(device));
    }
    for (Node node in circuit.nodes) {
      dataObjs.add(DataObj(node));
    }
    super.initState();
  }

  void addDeviceOf(DeviceKind kind) {
    print('add device of kind $kind');
    setState(() {
      circuit.addDeviceOf(kind);
      dataObjs.add(DataObj(circuit.devices.last));
    });
  }

  void newNode() {
    setState(() => circuit.newNode());
    dataObjs.add(DataObj(circuit.nodes.last));
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Side Circuit',
      home: Scaffold(
        appBar: AppBar(
          title: Text('Side Circuit'),
          actions: [
            DiagramControls(
              onAddDevice: addDeviceOf,
              onAddNode: newNode,
            )
          ],
        ),
        body: Diagram(
          circuit: circuit,
          dataObjs: dataObjs,
        ),
      ),
    );
  }
}
