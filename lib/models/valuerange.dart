import 'values.dart';

/// empty values is "empty range" indicating no solution is possible
class Range {
  List<Value> values = [];

  Range.empty();

  Range.pair(Value lower, Value upper) {
    this.values.add(lower);
    this.values.add(upper);
  }

  Range.initNumber() {
    this.values.add(Value.negInf());
    this.values.add(Value.posInf());
  }

  Range.initLogic() {
    this.values.add(Value.logicLowBound());
    this.values.add(Value.logicHighBound());
  }

  Range.copy(Range copy) {
    for (Value value in copy.values) {
      this.values.add(Value.copy(value));
    }
  }

  Range.shiftLeft(num shift, Range toCopy) {
    Value newLower;
    Value newUpper;
    if(toCopy.highest.equals(toCopy.lowest).stored) {
      newLower = Value.copy(toCopy.lowest);
      newUpper = Value.copy(toCopy.highest);
    } else {
      newLower = Value.copyShiftLeft(shift, toCopy.highest);
      newUpper = Value.copyShiftLeft(shift, toCopy.lowest);
    }
    values.add(newLower);
    values.add(newUpper);
  }

  Range.satisfyAdd(Range parent, Range sibling) {
    values.add(Value.subtract(parent.lowest, sibling.highest, false));
    values.add(Value.subtract(parent.highest, sibling.lowest, true));
  }

  Range.singleNum(num value) {
    this.values.add(Value.lowerBound(value, false));
    this.values.add(Value.upperBound(value, false));
  }

  Range.upperBoundNum(num value, bool isExclusive) {
    this.values.add(Value.negInf());
    this.values.add(Value.upperBound(value, isExclusive));
  }

  Range.lowerBoundNum(num value, bool isExclusive) {
    this.values.add(Value.lowerBound(value, isExclusive));
    this.values.add(Value.posInf());
  }

  Range.singleLogic(bool value) {
    if (value) {
      this.values.add(Value.logicHighBound());
    } else {
      this.values.add(Value.logicLowBound());
    }
  }

  num midVal() {
    return (this.highest.stored + this.lowest.stored) / 2;
  }

  Value get lowest {
    return this.values.first;
  }

  set lowest(Value value) {
    this.values.first = value;
  }

  Value get highest {
    return values.last;
  }

  set highest(Value value) {
    this.values.last = value;
  }

  void insert(Value value) {
    for (Value listVal in values) {
      if (value.isBelow(listVal)) {
        values.insert(values.indexOf(listVal), value);
        return;
      }
    }
    values.add(value);
  }

  num width() {
    return this.highest.stored - this.lowest.stored;
  }

  bool get isEmpty {
    return this.values.isEmpty;
  }

  bool get isNotEmpty {
    return this.values.isNotEmpty;
  }

  bool isLogic() {
    return this.highest.isLogic();
  }

  /// generates list of ranges that only contain one valid upper and lower
  /// boundary pair in each range
  List<Range> validRanges() {
    List<Range> ranges = [];
    for (int i = 0; i < this.values.length; i++) {
      int j = i + 1;
      if (j < this.values.length) {
        Value left = this.values[i];
        Value right = this.values[j];
        if (left.isLower && right.isUpper) {
          ranges.add(Range.pair(left, right));
        }
      }
    }
    return ranges;
  }

  Range closestTo(Value value){
    List<Range> validPairs = this.validRanges();
    for (Range pair in validPairs) {
      if(pair.contains(value)){
        return pair;
      }
    }
    return Range.empty();
  }

  void clear(){
    this.values = [];
  }

  bool contains(Value value) {
    if (this.isEmpty) {
      return false;
    }

    if (value.isLogic()) {
      for(Value val in this.values) {
        if (val.stored == value.stored) {
          return true;
        }
      }
      return false;
    } else {
      assert(this.values.length > 1, "range too small for num type");

      for (int i = 1; i < values.length; i++) {
        Value left = values[i - 1];
        Value right = values[i];

        if (left.boundaryContains(value) && right.boundaryContains(value)) {
          return true;
        }
      }
    }

    return false;
  }

  String toString() {
    String rangeString = "{ ";
    for (Value value in values) {
      rangeString += value.toString() + ", ";
    }
    rangeString += "}";
    return rangeString;
  }
}
