import 'package:flutter/material.dart';

import '../models/diagram.dart';

class DiagramControls extends StatelessWidget {
  final Function _addDiagramItem;
  final Function _upLevel;
  final Function _topLevel;

  DiagramControls(this._addDiagramItem,this._upLevel,this._topLevel);

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        IconButton(
            icon: Icon(Icons.home),
            tooltip: "go to top level",
            onPressed: () {
              _topLevel();
            }),
        IconButton(
            icon: Icon(Icons.arrow_back),
            tooltip: "go up one level",
            onPressed: () {
              _upLevel();
            }),
        IconButton(
            icon: Icon(Icons.add),
            tooltip: "add new element",
            onPressed: () {
              _addDiagramItem();
            }),
      ],
    );
  }
}
