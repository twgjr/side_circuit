// state notifier provider that tracks the state of buttons pressed in the toold bard
// such as add wire, add device, rotate, flip, etc.  The goal is to have only one state
// active at a time.  The state will be used to determine what click events do on the
// when clicking empty diagram space, devices, nodes, and terminals.  The state will
// also be used to determine what the mouse cursor looks like. The state will also be
// used to determine what the diagram controls look like.
//
//Example: when the add wire
// button is pressed, the cursor will change to a crosshair, and the diagram controls for
// adding a wire will change to active color.  The user clicks a terminal to start the wire,
// then clicks another wire, node, or terminal to end the wire.  The wire is added to the
// circuit.  The cursor changes back to the default cursor, and the diagram controls for
// adding a wire change back to the inactive color.
import 'package:flutter_riverpod/flutter_riverpod.dart';

class GestureStateNotifier extends StateNotifier<GestureState> {
  GestureStateNotifier() : super(GestureState());

  void update(GestureState gestureState) {
    state = gestureState;
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
}
