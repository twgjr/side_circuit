import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/node.dart';
import 'package:app/models/view/visual.dart';

class Terminal {
  Device device;
  String name;
  Node? node;
  Visual visual = Visual();

  Terminal(this.device, this.name);

  Terminal copyWith({Device? device, Node? node}) {
    final terminal = Terminal(device ?? this.device, name);
    terminal.node = node ?? this.node;
    terminal.visual = visual.copy();
    return terminal;
  }
}
