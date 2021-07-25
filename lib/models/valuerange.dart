import 'values.dart';

class Range {
  List<Value> values = [];
  bool useDiscrete = false;

  Range.empty();

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
    this.lower = rangeToSplit.lower; // lower bound
    this.upper = Value.upperBound(rangeToSplit.midVal(),true);
    if(this.useDiscrete) {
      // @todo remove values greater than the new upper bound
    }
  }

  /// midpoint of the range becomes the lower bound, upper bound unchanged
  Range.splitRight(Range rangeToSplit) {
    this.upper = rangeToSplit.upper; //upper bound
    this.lower = Value.lowerBound(rangeToSplit.midVal(),true);
    if(this.useDiscrete) {
      // @todo remove values lesser than the new lower bound
    }
  }

  Range.copyShiftLeft(num shift, Range toCopy){
    values.add(Value.copyShiftLeft(shift, toCopy.upper));
    values.add(Value.copyShiftLeft(shift, toCopy.lower));
  }

  num midVal() {
    return (this.upper.stored + this.lower.stored) / 2;
  }

  Value get lower {
    return this.values.first;
  }

  set lower(Value value){
    this.values.first = value;
  }

  Value get upper {
    return values.last;
  }

  set upper(Value value){
    this.values.last = value;
  }

  num width() {
    return this.upper.stored - this.lower.stored;
  }

  bool get isEmpty {
    return this.upper.isZeroExclusive() && this.lower.isZeroExclusive();
  }

  bool get isNotEmpty {
    return !(this.upper.isZeroExclusive() && this.lower.isZeroExclusive());
  }

  bool isLogic() {
    return this.upper.isLogic();
  }

  void setUpper(Value newUpper) {
    if (newUpper.isExclusive && this.upper.isExclusive) {
      if (newUpper.stored < this.upper.stored) {
        this.upper = newUpper;
      }
    }
    if (newUpper.isNotExclusive && this.upper.isNotExclusive) {
      if (newUpper.stored < this.upper.stored) {
        this.upper = newUpper;
      }
    }
    if (newUpper.isExclusive && this.upper.isNotExclusive) {
      if (newUpper.stored <= this.upper.stored) {
        this.upper = newUpper;
      }
    }
    if (newUpper.isNotExclusive && this.upper.isExclusive) {
      if (newUpper.stored < this.upper.stored) {
        this.upper = newUpper;
      }
    }
  }

  void setLower(Value newLower) {
    if (newLower.isExclusive && this.lower.isExclusive) {
      if (newLower.stored > this.lower.stored) {
        this.lower = newLower;
      }
    }
    if (newLower.isNotExclusive && this.lower.isNotExclusive) {
      if (newLower.stored > this.lower.stored) {
        this.lower = newLower;
      }
    }
    if (newLower.isExclusive && this.lower.isNotExclusive) {
      if (newLower.stored >= this.lower.stored) {
        this.lower = newLower;
      }
    }
    if (newLower.isNotExclusive && this.lower.isExclusive) {
      if (newLower.stored > this.lower.stored) {
        this.lower = newLower;
      }
    }
  }

  bool contains(var value) {
    if (value is bool) {
      return value;
    }
    //print("${this.lower.value.value}<${value}<${this.upper.value.value}");
    // otherwise value is num
    if (this.lower.isNotExclusive && this.upper.isNotExclusive) {
      if ((this.lower.stored <= value) &&
          (value <= this.upper.stored)) {
        return true;
      }
    }
    if (this.lower.isNotExclusive && this.upper.isExclusive) {
      if ((this.lower.stored <= value) &&
          (value < this.upper.stored)) {
        return true;
      }
    }
    if (this.lower.isExclusive && this.upper.isNotExclusive) {
      if ((this.lower.stored < value) &&
          (value <= this.upper.stored)) {
        return true;
      }
    }
    if (this.lower.isExclusive && this.upper.isExclusive) {
      if ((this.lower.stored < value) &&
          (value < this.upper.stored)) {
        return true;
      }
    }
    return false;
  }

  // Range containing(Range range) {
  //   if (range.isLogic()) {
  //     return range;
  //   }
  //   // otherwise value is num
  //
  //   // the given range is contained in this range
  //   if (this.contains(range.lower.stored) &&
  //       this.contains(range.upper.stored)) {
  //     return range;
  //   }
  //
  //   // this range is contained in the given range
  //   if (range.contains(this.lower.stored) &&
  //       range.contains(this.upper.stored)) {
  //     return Range.copy(this);
  //   }
  //
  //   // only given range upper is contained in this range
  //   if (!this.contains(range.lower.stored) &&
  //       this.contains(range.upper.stored)) {
  //     return Range(Boundary.copy(this.lower), Boundary.copy(range.upper));
  //   }
  //
  //   // only given range lower is contained in this range
  //   if (this.contains(range.lower.stored) &&
  //       !this.contains(range.upper.stored)) {
  //     return Range(Boundary.copy(range.lower), Boundary.copy(this.upper));
  //   }
  //
  //   // return empty range
  //   return Range(Boundary.excludes(Value.number(0),true),
  //       Boundary.excludes(Value.number(0),false));
  // }
}
