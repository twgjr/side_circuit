///  Dynamic and abstract numbers. Used throughout solver and model
///  Needed to help handle abstract numbers such as infinity
class Values {
  var value;

  Values.number(num val) {
    this.value = val;
  }
  Values.negInf(){
    this.value = -1073741824;  // dart VM minimum small integer on 32 bit system
  }
  Values.posInf(){
    this.value = 1073741823;  // dart VM maximum small integer on 32 bit system
  }

  //Values operator <=(Values values) => this.value <= values.value;

  bool isLogic() => this.value is bool;
  bool isNumber() => this.value is num;
}