import 'package:flutter_riverpod/flutter_riverpod.dart';

enum ModeState {
  addWire,
  addDevice,
  rotate,
  flip,
  delete,
  move,
  select,
  pan,
  zoom,
}

class ModeNotifier extends StateNotifier<Mode> {
  ModeNotifier() : super(Mode());

  void invertModeState(ModeState gestureState) {
    Mode newState = state.copy();
    newState.invert(gestureState);
    state = newState;
  }
}

final modeProvider = StateNotifierProvider<ModeNotifier, Mode>(
  (ref) => ModeNotifier(),
);

class Mode {
  bool addWire;
  bool addDevice;
  bool rotate;
  bool flip;
  bool delete;
  bool move;
  bool select;
  bool pan;
  bool zoom;

  Mode({
    this.addWire = false,
    this.addDevice = false,
    this.rotate = false,
    this.flip = false,
    this.delete = false,
    this.move = false,
    this.select = false,
    this.pan = false,
    this.zoom = false,
  });

  Mode copy() {
    return Mode(
      addWire: this.addWire,
      addDevice: this.addDevice,
      rotate: this.rotate,
      flip: this.flip,
      delete: this.delete,
      move: this.move,
      select: this.select,
      pan: this.pan,
      zoom: this.zoom,
    );
  }

  // invert the state of the specified gesture state enum
  void invert(ModeState gestureState) {
    switch (gestureState) {
      case ModeState.addWire:
        addWire = !addWire;
        break;
      case ModeState.addDevice:
        addDevice = !addDevice;
        break;
      case ModeState.rotate:
        rotate = !rotate;
        break;
      case ModeState.flip:
        flip = !flip;
        break;
      case ModeState.delete:
        delete = !delete;
        break;
      case ModeState.move:
        move = !move;
        break;
      case ModeState.select:
        select = !select;
        break;
      case ModeState.pan:
        pan = !pan;
        break;
      case ModeState.zoom:
        zoom = !zoom;
        break;
    }
  }
}
