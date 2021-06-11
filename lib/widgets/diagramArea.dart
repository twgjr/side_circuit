import 'package:flutter/material.dart';

import '../models/diagram.dart';
import 'diagramChildren.dart';
import 'diagramControls.dart';

class DiagramArea extends StatefulWidget {
  @override
  _DiagramAreaState createState() => _DiagramAreaState();
}

class _DiagramAreaState extends State<DiagramArea> {
  final Diagram _mainDiagram = Diagram();

  void _addNewDiagramItem() {
    setState(() {
      _mainDiagram.proxyRoot.children.add(DiagramItem.child(_mainDiagram.proxyRoot));
    });
    print("added new child: ${_mainDiagram.proxyRoot.children.length}");
  }

  void _downLevel(DiagramItem dItem) {
    setState(() {
      _mainDiagram.proxyRoot = dItem;
      print("down one level");
    });
  }

  void _upLevel() {
    setState(() {
      if(_mainDiagram.proxyRoot.parent != null) {
        _mainDiagram.proxyRoot = _mainDiagram.proxyRoot.parent;
        print("up one level");
      }
    });
  }

  void _topLevel() {
    setState(() {
        _mainDiagram.proxyRoot = _mainDiagram.root;
        print("top level");
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(children: [
      DiagramControls(_addNewDiagramItem,_upLevel,_topLevel),
      Text("level: ${_mainDiagram.distanceFromRoot()}"),
      DiagramChildren(_mainDiagram,_downLevel),
    ]);
  }
}
