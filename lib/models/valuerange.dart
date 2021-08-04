import 'values.dart';

class Range {
  List<Value> values = [];
  bool useDiscrete = false;

  Range.empty();

  Range.boundary(Value lower, Value upper){
    this.values.add(lower);
    this.values.add(upper);
  }

  Range.num(){
    this.values.add(Value.posInf());
    this.values.add(Value.negInf());
  }

  Range.logic(){
    this.values.add(Value.logic(true));
    this.values.add(Value.logic(false));
  }

  Range.copy(Range copy){
    for(Value value in copy.values){
      this.values.add(Value.copy(value));
    }
  }

  /// midpoint of the range becomes the upper bound, lower bound unchanged
  Range.splitLeft(Range rangeToSplit) {
    this.lowest = rangeToSplit.lowest; // lower bound
    this.highest = Value.upperBound(rangeToSplit.midVal(),true);
    if(this.useDiscrete) {
      // @todo remove values greater than the new upper bound
    }
  }

  /// midpoint of the range becomes the lower bound, upper bound unchanged
  Range.splitRight(Range rangeToSplit) {
    this.highest = rangeToSplit.highest; //upper bound
    this.lowest = Value.lowerBound(rangeToSplit.midVal(),true);
    if(this.useDiscrete) {
      // @todo remove values lesser than the new lower bound
    }
  }

  Range.shiftLeft(num shift, Range toCopy){
    values.add(Value.copyShiftLeft(shift, toCopy.highest));
    values.add(Value.copyShiftLeft(shift, toCopy.lowest));
  }

  Range.satisfyAdd(Range variable, Range parent, Range sibling){
    values.add(Value.subtract(parent.lowest, sibling.highest, false));
    values.add(Value.subtract(parent.highest, sibling.lowest, true));
  }

  Range.singleNum(num value){
    this.values.add(Value.lowerBound(value, false));
    this.values.add(Value.upperBound(value, false));
  }

  num midVal() {
    return (this.highest.stored + this.lowest.stored) / 2;
  }

  Value get lowest {
    return this.values.first;
  }

  set lowest(Value value){
    this.values.first = value;
  }

  Value get highest {
    return values.last;
  }

  set highest(Value value){
    this.values.last = value;
  }

  void insert(Value value){
    if(value.isLowerThan(this.lowest)){
      values.insert(0,value);
    }
    if(value is ){

    }
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

  void setUpper(Value newUpper) {
    if (newUpper.isExclusive && this.highest.isExclusive) {
      if (newUpper.stored < this.highest.stored) {
        this.highest = newUpper;
      }
    }
    if (newUpper.isNotExclusive && this.highest.isNotExclusive) {
      if (newUpper.stored < this.highest.stored) {
        this.highest = newUpper;
      }
    }
    if (newUpper.isExclusive && this.highest.isNotExclusive) {
      if (newUpper.stored <= this.highest.stored) {
        this.highest = newUpper;
      }
    }
    if (newUpper.isNotExclusive && this.highest.isExclusive) {
      if (newUpper.stored < this.highest.stored) {
        this.highest = newUpper;
      }
    }
  }

  void setLower(Value newLower) {
    if (newLower.isExclusive && this.lowest.isExclusive) {
      if (newLower.stored > this.lowest.stored) {
        this.lowest = newLower;
      }
    }
    if (newLower.isNotExclusive && this.lowest.isNotExclusive) {
      if (newLower.stored > this.lowest.stored) {
        this.lowest = newLower;
      }
    }
    if (newLower.isExclusive && this.lowest.isNotExclusive) {
      if (newLower.stored >= this.lowest.stored) {
        this.lowest = newLower;
      }
    }
    if (newLower.isNotExclusive && this.lowest.isExclusive) {
      if (newLower.stored > this.lowest.stored) {
        this.lowest = newLower;
      }
    }
  }

  bool contains(var value) {
    if (value is bool) {
      return value;
    }

    //print("${this.lower.value.value}<${value}<${this.upper.value.value}");
    // otherwise value is num
    if (this.lowest.isNotExclusive && this.highest.isNotExclusive) {
      if ((this.lowest.stored <= value) &&
          (value <= this.highest.stored)) {
        return true;
      }
    }
    if (this.lowest.isNotExclusive && this.highest.isExclusive) {
      if ((this.lowest.stored <= value) &&
          (value < this.highest.stored)) {
        return true;
      }
    }
    if (this.lowest.isExclusive && this.highest.isNotExclusive) {
      if ((this.lowest.stored < value) &&
          (value <= this.highest.stored)) {
        return true;
      }
    }
    if (this.lowest.isExclusive && this.highest.isExclusive) {
      if ((this.lowest.stored < value) &&
          (value < this.highest.stored)) {
        return true;
      }
    }
    return false;
  }

  printRange(){
    String rangeString = "{";
    for(Value value in values) {
      rangeString += value.valueString();
    }
    rangeString += "}";
  }
}
