import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/wire.dart';
import 'package:app/models/view/visual.dart';

class Terminal {
  Device device;
  String name;
  Wire? wire;
  Visual visual = Visual();

  Terminal(this.device, this.name);

  Terminal copyWith({Device? device, Wire? wire}) {
    final terminal = Terminal(device ?? this.device, name);
    terminal.wire = wire ?? this.wire;
    terminal.visual = visual.copy();
    return terminal;
  }
}
