import 'package:flutter_riverpod/flutter_riverpod.dart';

enum GestureStates {
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

class GestureStateNotifier extends StateNotifier<GestureState> {
  GestureStateNotifier() : super(GestureState());

  void invertGestureState(GestureStates gestureState) {
    GestureState newState = state.copy();
    newState.invert(gestureState);
    state = newState;
  }
}

final gestureStateProvider =
    StateNotifierProvider<GestureStateNotifier, GestureState>(
  (ref) => GestureStateNotifier(),
);

class GestureState {
  bool addWire = false;
  bool addDevice = false;
  bool rotate = false;
  bool flip = false;
  bool delete = false;
  bool move = false;
  bool select = false;
  bool pan = false;
  bool zoom = false;

  GestureState({
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

  GestureState copy() {
    return GestureState(
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
  void invert(GestureStates gestureState) {
    switch (gestureState) {
      case GestureStates.addWire:
        addWire = !addWire;
        break;
      case GestureStates.addDevice:
        addDevice = !addDevice;
        break;
      case GestureStates.rotate:
        rotate = !rotate;
        break;
      case GestureStates.flip:
        flip = !flip;
        break;
      case GestureStates.delete:
        delete = !delete;
        break;
      case GestureStates.move:
        move = !move;
        break;
      case GestureStates.select:
        select = !select;
        break;
      case GestureStates.pan:
        pan = !pan;
        break;
      case GestureStates.zoom:
        zoom = !zoom;
        break;
    }
  }
}
