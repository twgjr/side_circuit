import 'dart:math';

class DataObj<T> {
  T obj;
  Point<double> position = Point<double>(0, 0);

  DataObj(this.obj);
}
