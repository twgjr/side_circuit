import 'package:flutter_riverpod/flutter_riverpod.dart';

enum ModeStates {
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

class ModeStateNotifier extends StateNotifier<ModeState> {
  ModeStateNotifier() : super(ModeState());

  void invertModeState(ModeStates gestureState) {
    ModeState newState = state.copy();
    newState.invert(gestureState);
    state = newState;
  }
}

final modeStateProvider = StateNotifierProvider<ModeStateNotifier, ModeState>(
  (ref) => ModeStateNotifier(),
);

class ModeState {
  bool addWire;
  bool addDevice;
  bool rotate;
  bool flip;
  bool delete;
  bool move;
  bool select;
  bool pan;
  bool zoom;

  ModeState({
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

  ModeState copy() {
    return ModeState(
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
  void invert(ModeStates gestureState) {
    switch (gestureState) {
      case ModeStates.addWire:
        addWire = !addWire;
        break;
      case ModeStates.addDevice:
        addDevice = !addDevice;
        break;
      case ModeStates.rotate:
        rotate = !rotate;
        break;
      case ModeStates.flip:
        flip = !flip;
        break;
      case ModeStates.delete:
        delete = !delete;
        break;
      case ModeStates.move:
        move = !move;
        break;
      case ModeStates.select:
        select = !select;
        break;
      case ModeStates.pan:
        pan = !pan;
        break;
      case ModeStates.zoom:
        zoom = !zoom;
        break;
    }
  }
}
