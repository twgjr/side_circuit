import 'package:flutter/material.dart';
import 'widgets/diagram/diagram_area.dart';
import 'widgets/diagram/diagram_controls.dart';
import 'models/diagram.dart';

void main() => runApp(MainApp());

class MainApp extends StatefulWidget {
  @override
  State<StatefulWidget> createState() => MainAppState();
}

class MainAppState extends State<MainApp> {
  final Diagram diagram = Diagram();

  void addItem() {
    setState(() {
      diagram.subCircuit!.addDiagramItem();
    });
  }

  void deleteItem(DiagramItem child) {
    setState(() {
      diagram.subCircuit!.remove(child);
    });
  }

  void downLevel(DiagramItem dItem) {
    setState(() {
      diagram.moveDown(dItem);
    });
  }

  void upLevel() {
    setState(() {
      diagram.moveUp();
    });
  }

  void topLevel() {
    setState(() {
      diagram.moveToTop();
    });
  }

  void solve() {
    setState(() {
      diagram.solve();
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Side Circuit',
      home: Scaffold(
        appBar: AppBar(
          title: Text('Side Circuit'),
          actions: [DiagramControls(topLevel, upLevel, addItem, solve)],
        ),
        body: DiagramArea(
          dItems: diagram.subCircuit!.children,
          depth: diagram.subCircuit!.depth(),
          downLevel: downLevel,
          deleteItem: deleteItem,
        ),
      ),
    );
  }
}
