function createSpriteObjects(elementType) {
    var component = Qt.createComponent(elementType);
    var sprite = component.createObject(elementRectId);
}
