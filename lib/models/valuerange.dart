import 'values.dart';

class ValueRange {
  Boundary upper;
  Boundary lower;

  ValueRange(this.lower, this.upper);
  ValueRange.dynamic(var lower, var upper) {
    this.lower = Boundary.includes(Values.dynamic(lower));
    this.upper = Boundary.includes(Values.dynamic(upper));
  }

  ValueRange.copy( ValueRange toCopy ) {
    this.upper = Boundary.copy(toCopy.upper);
    this.lower = Boundary.copy(toCopy.lower);
  }

  bool contains( var value ) {
    if(lower.isInclusive() && upper.isInclusive()){
      if( (lower.value.value <= value) && (value <= upper.value.value) ) {
        return true;
      }
    }
    if(lower.isInclusive() && upper.isExclusive()){
      if( (lower.value.value <= value) && (value < upper.value.value) ) {
        return true;
      }
    }
    if(lower.isExclusive() && upper.isInclusive()){
      if( (lower.value.value < value) && (value <= upper.value.value) ) {
        return true;
      }
    }
    if(lower.isExclusive() && upper.isExclusive()){
      if( (lower.value.value < value) && (value < upper.value.value) ) {
        return true;
      }
    }
    return false;
  }

  ValueRange.startingTarget(Values target){
    if(target == null) {
      target = Values.number(0);
    }
    this.upper = Boundary.includes(target);
    this.lower = Boundary.includes(target);
  }

  Values mid() {
    var upperVal = upper.value.value;
    var lowerVal = lower.value.value;
    return (upperVal+lowerVal)/2;
  }

  Values min() {
    return lower.value.value;
  }

  Values max() {
    return upper.value.value;
  }

  void setUpper(Boundary newUpper) {
    if (newUpper.isExclusive() && this.upper.isExclusive()) {
      if (newUpper.value.value < this.upper.value.value) {
        this.upper = newUpper;
      }
    }
    if (newUpper.isInclusive() && this.upper.isInclusive()) {
      if (newUpper.value.value < this.upper.value.value) {
        this.upper = newUpper;
      }
    }
    if (newUpper.isExclusive() && this.upper.isInclusive()) {
      if (newUpper.value.value <= this.upper.value.value) {
        this.upper = newUpper;
      }
    }
    if (newUpper.isInclusive() && this.upper.isExclusive()) {
      if (newUpper.value.value < this.upper.value.value) {
        this.upper = newUpper;
      }
    }
  }

  void setLower(Boundary newLower) {
    if (newLower.isExclusive() && this.lower.isExclusive()) {
      if (newLower.value.value > this.lower.value.value) {
        this.lower = newLower;
      }
    }
    if (newLower.isInclusive() && this.lower.isInclusive()) {
      if (newLower.value.value > this.lower.value.value) {
        this.lower = newLower;
      }
    }
    if (newLower.isExclusive() && this.lower.isInclusive()) {
      if (newLower.value.value >= this.lower.value.value) {
        this.lower = newLower;
      }
    }
    if (newLower.isInclusive() && this.lower.isExclusive()) {
      if (newLower.value.value > this.lower.value.value) {
        this.lower = newLower;
      }
    }
  }
}

class Boundary {
  int type;
  Values value;

  Boundary.logicLow() {
    this.type = 0;
    this.value = Values.logic(false);
  }

  Boundary.logicHigh() {
    this.type = 1;
    this.value = Values.logic(true);
  }

  Boundary.includes(this.value) {
    this.type = 2;
  }

  Boundary.excludes(this.value) {
    this.type = 3;
  }

  Boundary.copy(Boundary toCopy) {
    this.type = toCopy.type;
    this.value = Values.copy(toCopy.value);
  }

  void includes(Values value){
    this.type = 2;
    this.value = value;
  }

  void excludes(Values value){
    this.type = 3;
    this.value = value;
  }

  bool isLogicLow() => this.type == 0;
  bool isLogicHigh() => this.type == 1;
  bool isInclusive() => this.type == 2;
  bool isExclusive() => this.type == 3;

}
