import 'package:app/models/wire.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

enum ModeState {
  addWire,
  activeWire,
  delete,
}

class ModeNotifier extends StateNotifier<Mode> {
  ModeNotifier() : super(Mode());

  void invertModeState(ModeState gestureState) {
    Mode newState = state.copy();
    newState.invert(gestureState);
    state = newState;
  }

  void reset() {
    state = Mode();
  }

  void setActiveWire(Wire wire) {
    Mode newState = state.copy();
    newState.setActiveWire(wire);
    state = newState;
  }

  void placeVertexAndContinue(Offset position) {
    Mode newState = state.copy();
    newState.activeWire!.placeAndContinue(position);
    state = newState;
  }
}

final modeProvider = StateNotifierProvider<ModeNotifier, Mode>(
  (ref) => ModeNotifier(),
);

class Mode {
  bool addWire;
  Wire? activeWire;
  bool delete;

  Mode({
    this.addWire = false,
    this.activeWire,
    this.delete = false,
  });

  Mode copy() {
    return Mode(
      addWire: this.addWire,
      activeWire: this.activeWire,
      delete: this.delete,
    );
  }

  void invert(ModeState modestate) {
    switch (modestate) {
      case ModeState.addWire:
        addWire = !addWire;
        break;
      case ModeState.activeWire:
        break;
      case ModeState.delete:
        delete = !delete;
        break;
    }
  }

  void setActiveWire(Wire wire) {
    activeWire = wire;
  }
}
