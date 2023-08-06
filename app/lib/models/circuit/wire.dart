import 'package:app/models/circuit/node.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/models/view/visual.dart';

class Wire {
  Terminal terminal;
  Node node;
  Visual visual = Visual();

  Wire(this.terminal, this.node);
}
