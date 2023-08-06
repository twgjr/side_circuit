/*
Terminal represents the properties of a terminal for the given element.
*/

import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/node.dart';
import 'package:app/models/view/visual.dart';

class Terminal {
  Device device;
  String name;
  Node? node;
  Visual visual = Visual();

  Terminal(this.device, this.name);
}
