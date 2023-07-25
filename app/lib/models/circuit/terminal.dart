/*
Terminal represents the properties of a terminal for the given element.
*/

import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/node.dart';

class Terminal {
  Device element;
  String name;
  Node? node;

  Terminal(this.element, this.name);
}
