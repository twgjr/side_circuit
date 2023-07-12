import 'package:flutter/material.dart';

import '../models/diagram.dart';
import 'diagramChildren.dart';
import 'diagramControls.dart';

class DiagramArea extends StatefulWidget {
  @override
  DiagramAreaState createState() => DiagramAreaState();
}

class DiagramAreaState extends State<DiagramArea> {
  final Diagram _mainDiagram = Diagram();

  void addItem() {
    setState(() {
      _mainDiagram.subCircuit!.addDiagramItem();
    });
  }

  void deleteItem(DiagramItem child) {
    setState(() {
      _mainDiagram.subCircuit!.remove(child);
    });
  }

  void downLevel(DiagramItem dItem) {
    setState(() {
      _mainDiagram.moveDown(dItem);
    });
  }

  void upLevel() {
    setState(() {
      _mainDiagram.moveUp();
    });
  }

  void topLevel() {
    setState(() {
      _mainDiagram.moveToTop();
    });
  }

  void solve() {
    setState(() {
      _mainDiagram.solve();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(children: [
      DiagramControls(this),
      Row(
        children: [
          Text("depth: ${_mainDiagram.subCircuit!.depth()}"),
          Text(" breadth: ${_mainDiagram.subCircuit!.breadth()}"),
        ],
      ),
      Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.max,
        children: [
          DiagramChildren(_mainDiagram.subCircuit!.children, this, context),
        ],
      ),
    ]);
  }
}
