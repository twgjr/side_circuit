/*
CIRCUITS is a library for creating and analyzing electrical circuits.  It primarily
follows the same structure as the SPICE circuit simulator.  It can access the ngspice 
API.
*/

import 'dart:math';
import 'package:tuple/tuple.dart';

enum ElementKind{
  V,I,R,VC,CC,SW,L,C,VG,CG,
}

enum Quantity{
  I,V,P,
}

enum SignalClass{
  PULSE,SIN,
}

class System {
  List<Circuit> circuits = [];
  List<Element> elements = [];
  List<Node> nodes = [];

  void load(
    Map<int, double> nodePotentials, 
    Map<int, double> elementCurrents,
    double time) {
      assert(circuits.length == 1);
      circuits[0].load(nodePotentials,elementCurrents,time);
  }

  int numCircuits() {
    return circuits.length;
  }
    
  int numElements() {
    return circuits.fold(0, (sum, circuit) => sum + circuit.numElements());
  }

  Circuit addCircuit() {
    var circuit = Circuit(this);
    circuits.add(circuit);
    return circuit;
  }
    
  void removeCircuit(Circuit circuit) {
    circuits.remove(circuit);
    circuits.removeWhere((c) => c == circuit);
  }

  Node addNode(Circuit circuit, List<Element> elements) {
    assert(circuits.contains(circuit));
    var cktNode = Node(circuit, elements);
    nodes.add(cktNode);
    return cktNode;
  }
    
  void removeNode(Node node) {
    node.circuit.removeNode(node);
    nodes.remove(node);
    if (node.circuit.isEmpty()) {
      removeCircuit(node.circuit);
    }
  }
    
  Element addElementOf(ElementKind kind) {
    var circuit = addCircuit();
    var highNode = addNode(circuit, []);
    var lowNode = addNode(circuit, []);
    circuit.nodes.add(highNode);
    circuit.nodes.add(lowNode);
    var element = newElement(kind, circuit, highNode, lowNode);
    elements.add(element);
    highNode.elements.add(element);
    lowNode.elements.add(element);
    circuit.elements.add(element);
    return element;
  }
    
  Element newElement(ElementKind kind, Circuit circuit, Node high, Node low) {
    switch (kind) {
      case ElementKind.V:
        return Voltage(circuit, high, low);
      case ElementKind.I:
        return Current(circuit, high, low);
      case ElementKind.R:
        return Resistor(circuit, high, low);
      default:
        throw Exception('Not implemented');
    }
  }
    
  void removeElement(Element element) {
    element.circuit.removeElement(element);
    if (element.circuit.isEmpty()) {
      removeCircuit(element.circuit);
    }
    elements.remove(element);
  }

  Tuple2<Element, Element> addElementPair(ElementKind parentKind, ElementKind childKind) {
    assert(parentKind == ElementKind.VC || parentKind == ElementKind.CC);
    assert(
        childKind == ElementKind.SW || childKind == ElementKind.VG || childKind == ElementKind.CG);
    var parent = addElementOf(parentKind);
    var child = addElementOf(childKind);
    child.parent = parent;
    parent.child = child;
    return Tuple2(parent, child);
  }

  void connect(Node a, Node b) {
    if (a == b) {
      return;
    }
    if (a.circuit != b.circuit) {
      mergeCircuits(a.circuit, b.circuit);
    }
    mergeNodes(a, b);
  }

  void mergeNodes(Node a, Node b) {
    if (a == b) {
      return;
    }
    if (a.circuit != b.circuit) {
      throw Exception('Cannot merge nodes in different circuits');
    }
    var small = a.numElements() < b.numElements() ? a : b;
    var large = a.numElements() >= b.numElements() ? a : b;
    for (var element in small.elements) {
      if (element.high == small) {
        element.high = large;
      }
      if (element.low == small) {
        element.low = large;
      }
      large.elements.add(element);
    }
    small.elements.clear();
    small.circuit.nodes.remove(small);
    nodes.remove(small);
  }

  void mergeCircuits(Circuit a, Circuit b) {
    if (a == b) {
      return;
    }
    var small = a.numElements() < b.numElements() ? a : b;
    var large = a.numElements() >= b.numElements() ? a : b;
    for (var element in small.elements) {
      element.circuit = large;
      large.elements.add(element);
    }
    for (var node in small.nodes) {
      node.circuit = large;
      large.nodes.add(node);
    }
    small.elements.clear();
    small.nodes.clear();
    circuits.remove(small);
  }
    
  Tuple2<Circuit, Circuit> switchedResistor() {
    var childSrc = addElementOf(ElementKind.V);
    var parentSrc = addElementOf(ElementKind.V);
    var childRes = addElementOf(ElementKind.R);
    var parentRes = addElementOf(ElementKind.R);
    var pair = addElementPair(ElementKind.VC, ElementKind.SW);
    var child = pair.item1;
    var parent = pair.item2;
    connect(childSrc.high, child.high);
    connect(child.low, childRes.high);
    connect(childSrc.low, childRes.low);
    connect(parentSrc.high, parentRes.high);
    connect(parentRes.high, parent.high);
    connect(parentSrc.low, parentRes.low);
    connect(parentRes.low, parent.low);
    return Tuple2(parent.circuit, child.circuit);
  }
    
  Circuit ring(ElementKind sourceKind, ElementKind loadKind, int numLoads) {
    assert(numLoads > 0);
    var source = addElementOf(sourceKind);
    var firstLoad = addElementOf(loadKind);
    connect(source.high, firstLoad.high);
    var prevElement = firstLoad;
    for (var l = 0; l < numLoads - 1; l++) {
      var newLoad = addElementOf(loadKind);
      connect(prevElement.low, newLoad.high);
      prevElement = newLoad;
    }
    connect(source.low, prevElement.low);
    return source.circuit;
  }

  Circuit ladder(ElementKind sourceKind, ElementKind loadKind, int numLoads) {
    assert(numLoads > 0);
    var source = addElementOf(sourceKind);
    var firstLoad = addElementOf(loadKind);
    connect(source.high, firstLoad.high);
    connect(source.low, firstLoad.low);
    var prevElement = firstLoad;
    for (var l = 0; l < numLoads - 1; l++) {
      var newLoad = addElementOf(loadKind);
      connect(prevElement.high, newLoad.high);
      connect(prevElement.low, newLoad.low);
      prevElement = newLoad;
    }
    return source.circuit;
  }

  Circuit rc() {
    var v = addElementOf(ElementKind.V);
    var r = addElementOf(ElementKind.R);
    var c = addElementOf(ElementKind.C);
    connect(v.high, r.high);
    connect(r.low, c.high);
    connect(v.low, c.low);
    return v.circuit;
  }
}

class Circuit {
  System system;
  List<Node> nodes = [];
  List<dynamic> elements = [];

  Circuit(this.system);

  int index() {
    return system.circuits.indexOf(this);
  }

  bool isEmpty() {
    return elements.isEmpty && nodes.isEmpty;
  }

  void clear() {
    nodes.clear();
    elements.clear();
  }

  void removeElement(Element element) {
    elements.remove(element);
  }

  void addNode(Node node) {
    nodes.add(node);
  }

  void removeNode(Node node) {
    nodes.remove(node);
  }

  int numNodes() {
    return nodes.length;
  }

  int numElements() {
    return elements.length;
  }

  void draw() {
    // Implement the visualization logic using the drawing library of your choice
  }

  dynamic nxGraph() {
    // Implement the graph creation logic using a graph library of your choice
  }

  dynamic M({String dtype = 'float'}) {
    // Implement the conversion to incidence matrix logic using a linear algebra library of your choice
  }

  List<Element> elementsParallelTo(Element referenceElement, bool includeRef) {
    List<Element> parallels = [];
    for (var highElement in referenceElement.high.elements) {
      for (var lowElement in referenceElement.low.elements) {
        if (highElement == lowElement) {
          if (lowElement != referenceElement || includeRef) {
            parallels.add(lowElement);
          }
          break;
        }
      }
    }
    return parallels;
  }

  List<dynamic> branch_elements(Element refElement, {bool includeRef = false}) {
    List<dynamic> series = [];
    if (includeRef) {
      series.add({"element": refElement, "polarity": 1});
    }
    for (var refNode in [refElement.low, refElement.high]) {
      var node = refNode;
      var element = refElement;
      while (node.elements.length == 2) {
        element = nextSeriesElement(element, node);
        if (element == refElement) {
          return series;
        } else {
            var result = nextNodeIn(element, node);
            node = result.item1;
            var polarity = result.item2;
            series.add({"element": element, "polarity": polarity});
        }
      }
    }
    return series;
  }

  Tuple2<Node,int> nextNodeIn(Element element, Node fromNode) {
    if (fromNode == element.high) {
      return Tuple2(element.high, 1);
    } else if (fromNode == element.low) {
      return Tuple2(element.high, -1);
    } else {
      throw Exception('Node not in element');
    }
  }

  Element nextSeriesElement(Element fromElement, Node sharedNode) {
    var toElements = sharedNode.elements;
    assert(toElements.length == 2);
    for (var element in toElements) {
      if (element != fromElement) {
        return element;
      }
    }
    throw Exception('No next series element found');
  }
  

  void load(Map<int, double> nodePotentials, Map<int, double> elementCurrents, double time) {
    Node gnd = nodes[0];
    if (!gnd.data.containsKey(Quantity.P)) {
      gnd.data[Quantity.P] = {};
    }
    gnd.data[Quantity.P]![time] = 0;
    for (var n in nodePotentials.keys) {
      Node node = nodes[n];
      if (!node.data.containsKey(Quantity.P)) {
        node.data[Quantity.P] = {};
      }
      node.data[Quantity.P]![time] = nodePotentials[n]!;
    }
    for (var e in elementCurrents.keys) {
      Element element = elements[e];
      var branchCurrent = elementCurrents[e];
      var branchElements = branch_elements(element, includeRef: true);
      for (var elementPolarityDict in branchElements) {
        element = elementPolarityDict["element"];
        var polarity = elementPolarityDict["polarity"];
        if (!element.data.containsKey(Quantity.I)) {
          element.data[Quantity.I] = {};
        }
        element.data[Quantity.I]![time] = branchCurrent! * polarity;
      }
    }
    for (var element in elements) {
      var high = element.high.data[Quantity.P]![time];
      var low = element.low.data[Quantity.P]![time];
      if (!element.data.containsKey(Quantity.V)) {
        element.data[Quantity.V] = {};
      }
      element.data[Quantity.V]![time] = high - low;
    }
  }
}

    
class SignalFunction {
  late String kind;

  SignalFunction(SignalClass kind) {
    this.kind = kind.toString().split('.').last;
  }

  String toSpice() {
    List<String> args = [];
    this.toMap().forEach((key, value) {
      args.add('$key $value');
    });
    return args.join(' ');
  }

  Map<String, dynamic> toMap() {
    return {'kind': kind};
  }
}

class Sin extends SignalFunction {
  late double V0;
  late double VA;
  late double FREQ;
  late double TD;
  late double THETA;
  late double PHASE;

  Sin(double dc, double ampl, double freq,
      {double delay = 0.0, double decay = 0.0, double phase = 0.0})
      : super(SignalClass.SIN) {
    this.V0 = dc;
    this.VA = ampl;
    this.FREQ = freq;
    this.TD = delay;
    this.THETA = decay;
    this.PHASE = phase;
  }

  @override
  Map<String, dynamic> toMap() {
    var map = super.toMap();
    map.addAll({
      'V0': V0,
      'VA': VA,
      'FREQ': FREQ,
      'TD': TD,
      'THETA': THETA,
      'PHASE': PHASE,
    });
    return map;
  }
}

class Pulse extends SignalFunction {
  late double V1;
  late double V2;
  late double TD;
  late double TR;
  late double TF;
  late double PW;
  late double PER;
  late int NP;

  Pulse(double val1, double val2, double freq,
      {double delay = 0.0,
      double? riseTime,
      double? fallTime,
      double dutyRatio = 0.5,
      int numPulses = 0})
      : super(SignalClass.PULSE) {
    this.V1 = val1;
    this.V2 = val2;
    this.TD = delay;
    this.TR = riseTime ?? (fallTime != null ? fallTime : (PER / 1000));
    this.TF = fallTime ?? (riseTime != null ? riseTime : (PER / 1000));
    var period = 1 / freq;
    this.PW = period * dutyRatio;
    this.PER = period;
    this.NP = numPulses;
  }

  @override
  Map<String, dynamic> toMap() {
    var map = super.toMap();
    map.addAll({
      'V1': V1,
      'V2': V2,
      'TD': TD,
      'TR': TR,
      'TF': TF,
      'PW': PW,
      'PER': PER,
      'NP': NP,
    });
    return map;
  }
}

abstract class Element {
  Circuit circuit;
  Node high;
  Node low;
  final ElementKind kind;
  Element? parent;
  Element? child;
  final Map<Quantity, Map<double, double>> data = {};

  Element(this.circuit, this.high, this.low, this.kind);

  String behavior();

  String toSpice() {
    final id = '${kind.name}${index()}';
    final nLow = low.index.toString();
    final nHigh = high.index.toString();
    return '$id $nHigh $nLow ${behavior()}';
  }

  void addData(Quantity prop, double time, double value) {
    if (!data.containsKey(prop)) {
      data[prop] = {};
    }
    data[prop]![time] = value;
  }

  int index() {
    return circuit.elements.indexOf(this);
  }

  int? parentIndex() {
    if (parent == null) {
      return null;
    }
    return parent!.index();
  }

  int circuitIndex() {
    return circuit.index();
  }

  int? parentCircuitIndex() {
    if (parent == null) {
      return null;
    }
    return parent!.circuit.index();
  }

  String get id => '${kind.name}_${circuit.index()}_${index()}';

  @override
  String toString() {
    return id;
  }

  bool hasParent() {
    return parent != null;
  }

  bool hasParentOf(ElementKind kind) {
    assert(kind == ElementKind.VC || kind == ElementKind.CC);
    return hasParent() && parent!.kind == kind;
  }

  int get edgeKey {
    final parallels = circuit.elementsParallelTo(this, true);
    return parallels.indexOf(this);
  }
}

class IndependentSource extends Element {
  double? dc;
  double? acMag;
  double? acPhase;
  SignalFunction? sigFunc;

  IndependentSource(Circuit circuit, Node high, Node low, ElementKind kind)
      : super(circuit, high, low, kind);
  
  String behavior() {
    var args = <String>[];
    if (dc != null) {
      args.add('DC');
      args.add(dc.toString());
    }
    if (acMag != null) {
      args.add('AC');
      args.add(acMag.toString());
    }
    if (acPhase != null) {
      args.add(acPhase.toString());
    }
    if (sigFunc != null) {
      args.add(sigFunc!.toSpice());
    }
    return args.join(' ');
  }
}

class Voltage extends IndependentSource {
  Voltage(Circuit circuit, Node high, Node low)
      : super(circuit, high, low, ElementKind.V);
}

class Current extends IndependentSource {
  Current(Circuit circuit, Node high, Node low)
      : super(circuit, high, low, ElementKind.I);
}
    
class Resistor extends Element {
  double? parameter;

  Resistor(Circuit circuit, Node high, Node low) : super(circuit, high, low, ElementKind.R) {
    parameter = null;
  }

  String behavior() {
    assert(parameter != null);
    assert(parameter! > 0);
    return parameter.toString();
  }
}
    
class Node {
  Circuit circuit;
  List<Element> elements;
  Map<Quantity, Map<double, double>> data = {};

  Node(this.circuit, this.elements);

  @override
  String toString() {
    var elStr = elements.join(',');
    return 'Node($index:$elStr)';
  }

  int numElements() {
    return elements.length;
  }

  int get index => circuit.nodes.indexOf(this);

  void addElement(Element element) {
    if (!elements.contains(element)) {
      elements.add(element);
    }
  }

  void removeElement(Element element) {
    if (elements.contains(element)) {
      elements.remove(element);
    }
  }
}

class Signal {
  Map<double, double> _data;
  double lower;
  double upper;
  double max;
  bool isPeriodic;

  Signal(Map<double, double> data):_data = data, isPeriodic = false, max = 0.0, lower = 0.0, upper = 0.0 ;

  Iterable<double> values() => _data.values;

  Iterable<double> keys() => _data.keys;

  Iterable<MapEntry<double, double>> items() => _data.entries;

  @override
  String toString() => _data.toString();

  int length() => _data.length;

  double operator [](double time) {
    if (isPeriodic) {
      if (max > 0.0) {
        time = time % max;
      }
    }
    if (time < lower || upper < time) {
      setWindow(time);
    }
    if (!isPeriodic && upper < time) {
      assert(lower <= upper);
      return _data[max]!;
    }
    if (_data.containsKey(time)) {
      assert(lower <= upper);
      return _data[time]!;
    }
    return interpolate(time);
  }

  double interpolate(double time) {
    final lowVal = _data[lower];
    final highVal = _data[upper];
    final ratio = (time - lower) / (upper - lower);
    assert(lower <= upper);
    return lowVal! + ratio * (highVal! - lowVal);
  }

  void setWindow(double time) {
    var prevKey = 0.0;
    for (final timeKey in _data.keys) {
      if (prevKey <= time && time <= timeKey) {
        lower = prevKey;
        upper = timeKey;
        break;
      }
      prevKey = timeKey;
    }
  }

  void operator []=(double key, double value) {
    _data[key] = value;
    if (key > max) {
      max = key;
    }
  }

  Iterator<double> iterator() => _data.keys.iterator;

  bool operator ==(Object signal) {
    if (!(signal is Signal)) {
      return false;
    }
    if (length() != signal.length()) {
      return false;
    }
    for (final time in _data.keys) {
      if (!signal._data.containsKey(time)) {
        return false;
      }
      if (!closeTo(signal[time], this[time], 1e-6)) {
        return false;
      }
    }
    return true;
  }

  Signal operator -() {
    final data = <double, double>{};
    for (final entry in _data.entries) {
      data[entry.key] = -entry.value;
    }
    return Signal(data);
  }

  Signal operator +(Signal signal) {
    assert(length() == signal.length() || length() == 1 || signal.length() == 1);
    final sigSum = <double, double>{};
    if (length() == 1 && signal.length() > 1) {
      for (final time in _data.keys) {
        sigSum[time] = this[0.0] + signal[time];
      }
    }
    if (length() > 1 && signal.length() == 1) {
      for (final time in _data.keys) {
        sigSum[time] = this[time] + signal[0.0];
      }
    }
    if (length() == signal.length()) {
      for (final time in _data.keys) {
        sigSum[time] = this[time] + signal[time];
      }
    }
    return Signal(sigSum);
  }

  Signal operator -(Signal signal) {
    assert(length() == signal.length() || length() == 1 || signal.length() == 1);
    final sigSum = <double, double>{};
    if (length() == 1 && signal.length() > 1) {
      for (final time in _data.keys) {
        sigSum[time] = this[0.0] - signal[time];
      }
    }
    if (length() > 1 && signal.length() == 1) {
      for (final time in _data.keys) {
        sigSum[time] = this[time] - signal[0.0];
      }
    }
    if (length() == signal.length()) {
      for (final time in _data.keys) {
        sigSum[time] = this[time] - signal[time];
      }
    }
    return Signal(sigSum);
  }

  Signal operator *(Signal signal) {
    assert(length() == signal.length() || length() == 1 || signal.length() == 1);
    final sigSum = <double, double>{};
    if (length() == 1 && signal.length() > 1) {
      for (final time in _data.keys) {
        sigSum[time] = this[0.0] * signal[time];
      }
    }
    if (length() > 1 && signal.length() == 1) {
      for (final time in _data.keys) {
        sigSum[time] = this[time] * signal[0.0];
      }
    }
    if (length() == signal.length()) {
      for (final time in _data.keys) {
        sigSum[time] = this[time] * signal[time];
      }
    }
    return Signal(sigSum);
  }

  Signal operator /(Signal signal) {
    assert(length() == signal.length() || length() == 1 || signal.length() == 1);
    final sigSum = <double, double>{};
    if (length() == 1 && signal.length() > 1) {
      for (final time in _data.keys) {
        sigSum[time] = this[0.0] / signal[time];
      }
    }
    if (length() > 1 && signal.length() == 1) {
      for (final time in _data.keys) {
        sigSum[time] = this[time] / signal[0.0];
      }
    }
    if (length() == signal.length()) {
      for (final time in _data.keys) {
        sigSum[time] = this[time] / signal[time];
      }
    }
    return Signal(sigSum);
  }

  double rms() {
    final sumOfSquares = _data.values.map((value) => pow(value, 2)).reduce((a, b) => a + b);
    final count = _data.length;
    return sqrt(sumOfSquares / count);
  }
}

bool closeTo(double a, double b, double tolerance) {
  return (a - b).abs() <= tolerance;
}