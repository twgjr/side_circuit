import 'package:flutter/material.dart';

class DiagramControls extends StatelessWidget {
  // a list of DiagramArea methods to be called by the buttons without passing
  // the DiagramArea state object
  final void Function() topLevel;
  final void Function() upLevel;
  final void Function() addItem;
  final void Function() solve;

  DiagramControls(this.topLevel, this.upLevel, this.addItem, this.solve);

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        IconButton(
          icon: Icon(Icons.home),
          tooltip: "go to top level",
          onPressed: topLevel,
        ),
        IconButton(
          icon: Icon(Icons.arrow_back),
          tooltip: "go up one level",
          onPressed: upLevel,
        ),
        IconButton(
          icon: Icon(Icons.add),
          tooltip: "add new element",
          onPressed: addItem,
        ),
        IconButton(
          icon: Icon(Icons.calculate),
          tooltip: "solve model",
          onPressed: solve,
        ),
      ],
    );
  }
}
