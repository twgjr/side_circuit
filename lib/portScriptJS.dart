function createPortObjects(side,position) {
    var component = Qt.createComponent("Port.qml");

    var {portX,portY} = setCoodinates(side,position)

//    console.log(portX)
//    console.log(portY)

    var port = component.createObject(itemRectId, {x: portX, y: portY});

    if (port === null) {
        // Error Handling
        console.log("Error creating object");
    }
}

function setCoodinates(side,position) {
    var portX=0
    var portY=0
    switch(side) {
    case 0:
        // top
        portX = position
        portY = 0
        return {portX,portY}
        //break;
    case 1:
        // bottom
        portX = position
        portY = itemRectId.height
        return {portX,portY}
    case 2:
        // left
        portX = 0
        portY = position
        return {portX,portY}
    case 3:
        // right
        portX = itemRectId.width
        portY = position
        return {portX,portY}
    default:
        return {portX,portY}
    }
}
