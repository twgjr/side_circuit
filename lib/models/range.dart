import 'values.dart';

class Range {
  Boundary upper;
  Boundary lower;

  Range(this.upper,this.lower);

  void setUpper(Boundary newUpper){
    if(newUpper.isExclusive() && this.upper.isExclusive()) {
      if(newUpper.value.value < this.upper.value.value){
        this.upper = newUpper;
      }
    }
    if(newUpper.isInclusive() && this.upper.isInclusive()) {
      if(newUpper.value.value < this.upper.value.value){
        this.upper = newUpper;
      }
    }
    if(newUpper.isExclusive() && this.upper.isInclusive()) {
      if(newUpper.value.value <= this.upper.value.value){
        this.upper = newUpper;
      }
    }
    if(newUpper.isInclusive() && this.upper.isExclusive()) {
      if(newUpper.value.value < this.upper.value.value){
        this.upper = newUpper;
      }
    }
  }

  void setLower(Boundary newLower){
    if(newLower.isExclusive() && this.lower.isExclusive()) {
      if(newLower.value.value > this.lower.value.value){
        this.lower = newLower;
      }
    }
    if(newLower.isInclusive() && this.lower.isInclusive()) {
      if(newLower.value.value > this.lower.value.value){
        this.lower = newLower;
      }
    }
    if(newLower.isExclusive() && this.lower.isInclusive()) {
      if(newLower.value.value >= this.lower.value.value){
        this.lower = newLower;
      }
    }
    if(newLower.isInclusive() && this.lower.isExclusive()) {
      if(newLower.value.value > this.lower.value.value){
        this.lower = newLower;
      }
    }
  }
}

class Boundary {
  int type;
  Values value;

  Boundary.includes(this.value){
    this.type = 0;
  }
  Boundary.excludes(this.value){
    this.type = 1;
  }

  bool isInclusive()=> this.type == 0;
  bool isExclusive()=> this.type == 1;
}