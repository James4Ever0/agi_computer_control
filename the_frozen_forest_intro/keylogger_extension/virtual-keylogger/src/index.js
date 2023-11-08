console.log('Starting keylogger for browser');
const serverPort = 4471
const baseUrl = `http://localhost:${serverPort}`
const backendUrl = `${baseUrl}/browserInputEvent`;
// const identifierUrl = `${baseUrl}/getIdentifier`;

// function generateUUIDFallback() {
//   var d = new Date().getTime();
//   if (window.performance && typeof window.performance.now === "function") {
//     d += performance.now(); // use high-precision timer if available
//   }
//   var uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
//     var r = (d + Math.random() * 16) % 16 | 0;
//     d = Math.floor(d / 16);
//     return (c == 'x' ? r : (r & 0x3 | 0x8)).toString(16);
//   });
//   return uuid;
// }
// function generateUUID() {
//   const crypto = window.crypto || window.msCrypto;
//   if (crypto) {
//     const array = new Uint32Array(4);
//     crypto.getRandomValues(array);
//     return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
//       const r = (array[0] & 0x0f) / 0x0f;
//       const v = c === 'x' ? r : (r & 0x3 | 0x8);
//       array = array.slice(1);
//       return v.toString(16);
//     });
//   } else {
//     console.error('crypto API not available');
//     return generateUUIDFallback();
//   }
// }

// const pageIdentifier = generateUUID();
// fetch(identifierUrl, {
//   method: "GET"
// }).then(response => { const pageIdentifier = response.json()['client_id'] })
// https://developer.mozilla.org/en-US/docs/Web/Events
// noe we have click/dblclick, keypress events
// how to handle them?
const eventTypes = ['keydown', 'keyup', 'mousedown', 'mouseup', 'mousemove'];
// const eventTypes = ['keydown', 'keyup', 'mousedown', 'mouseup', 'mousemove', 'resize'];
var keylogger_timestamp_private = null; // you can check if this thing still works.
// var pageIdentifier = null;

function sendHIDEvent(event, e) {
  let event_keys = Object.keys(e.__proto__);
  let event_data = {}
  for (let k of event_keys) {
    event_data[k] = e[k];
  }
  // debugger
  keylogger_timestamp_private = new Date();
  const inputEvent = {
    // eventType: event,
    // timestamp: keylogger_timestamp_private,
    // data: JSON.stringify(event_data),

    // "eventType": event,
    "timestamp": keylogger_timestamp_private,
    "client_id": pageIdentifier,
    "payload": { 'eventType': event, 'data': event_data },
    // "data": JSON.stringify(event_data),

    // data: JSON.stringify(e),
    // data: e,
  };
  // console.log(inputEvent);
  // debugger;
  // console.log('payload:', JSON.stringify(inputEvent))
  fetch(backendUrl, {
    method: "POST",
    // or you could remove the 'mode' parameter
    mode: "cors", // you must use cors or the content will be unprocessable.
    // mode: "no-cors",
    headers: { "Content-Type": "application/json" },
    // json: {browserEvent: inputEvent}
    // body: inputEvent,
    // body: "hello world",
    // body: { body: JSON.stringify(inputEvent) },
    body: JSON.stringify(inputEvent),
  }).then((res) => {
    // console.log(`posted ${event} event`, res);
    // console.log(`posted ${event} event`, res.json());
  }).catch((err) => {
    console.log(`error posting ${event} event`, err);
  });
}
const pageIdentifierPrefix = "pageIdentifier_";
function getPageIdentifierFromExposedFunctionName() {
  let wk = Object.keys(window)
  let candidate_keys = [];
  for (let k of wk) {
    if (k.startsWith(pageIdentifierPrefix)) {
      let myIdentifier = k.replace(pageIdentifierPrefix, "").replace(/\_/g, "-");
      candidate_keys.push(myIdentifier);
    }
  }
  if (candidate_keys.length == 1) {
    return candidate_keys[0];
  } else {
    console.error('Invalid page identifier candidates:', candidate_keys)
  }
  return 'unknown'
}
const pageIdentifier = getPageIdentifierFromExposedFunctionName();
function addSpecificEventListener(event) {
  document.addEventListener(event, (e) => {
    // console.log('event', event, e);
    // debugger
    // if (isVariableEmpty(pageIdentifier)) {
    //   pageIdentifier = getPageIdentifierFromExposedFunctionName()
    // not working for content script

    // fetch(identifierUrl, {
    //   method: "GET"
    // }).then(response => {
    //   pageIdentifier = response.json()['client_id'];
    //   sendHIDEvent(event, e);
    // })
    // window.generateUUID(JSON.stringify({})).then(r => {
    //   pageIdentifier = r;
    //   sendHIDEvent(event, e);
    // });
    // } 
    // else {
    sendHIDEvent(event, e);
    // }
  });
}

// window.generateUUID().then(pageIdentifier => {
for (const event of eventTypes) {
  addSpecificEventListener(event, pageIdentifier);
}
// }); // this can be expected from playwright.


function setElementAttributeAsCursorReady(element, elementId) {
  element.style.position = "absolute";
  element.style.pointerEvents = "none";
  element.id = elementId;
}


function createOmniscentPointerElement(elementId) {
  // var divElement = document.createElement("div");

  // divElement.style.width = "10px";
  // divElement.style.height = "10px";
  // divElement.style.backgroundColor = "red";
  // divElement.style.borderRadius = "50%";

  // setElementAttributeAsCursorReady(divElement, elementId)

  // Create the img element
  var imgElement = document.createElement("img");
  imgElement.src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAUCAQAAAD8O3+kAAAA+ElEQVR4nHXOsSsEcBjG8S/nkOIGk+EyXHGuDIazyMRikbKgLC4ZZGAysPEnmC9lvOnKZjJcWQyUOqWQYpS7zrnjvpa7Y/jdO71Pn3qfFw7oOD/sd5C407IXpEmfnWmwE6CkNd9M/7AVoLL64lSdjSDpk5M11oOkD6aqrARJ7x2rsBwkvTNRZjFIeuPoBwtB0qLj78xBz19riSM+gSgjsWKetSZ9cck811ycUKALiABJKzbcdNiS53L27/mUVXclS/bUuskSiTZNuC05oqRnv/VYDtvULXkGgEi0cOurg4/EWpZjqLlllly1t0C8Rf3tAyNckaGvFX8Bxhaqb4UTp4MAAAAASUVORK5CYII=";

  setElementAttributeAsCursorReady(imgElement, elementId);
  // Append the img element to the div element
  // divElement.appendChild(imgElement);

  // Insert the div element into the document body
  document.body.appendChild(imgElement);
  // document.body.appendChild(divElement);
  return imgElement;
  // return divElement;
}
const pointerElementId = 'omniscent_pointer';
function isVariableEmpty(v) {
  return v === undefined | v == null
}
function getOmniscentPointer() {
  // while (isVariableEmpty(vpointer)) {
  var pointer = document.getElementById(pointerElementId);
  if (isVariableEmpty(pointer)) {
    console.log('pointer is undefined');
    console.log('creating pointer element');
    pointer = createOmniscentPointerElement(pointerElementId);
    // pointer = createOmniscentPointerElement(pointerElementId);
  }
  // }
  return pointer;
}

// sometimes this pointer is misaligned.
function addMouseEventTracer(eventName) {
  document.addEventListener(eventName, function (event) {
    const x = event.clientX;
    const y = event.clientY;
    // const x = event.layerX;
    // const y = event.layerY;
    // const x = event.x;
    // const y = event.y;
    // var pointer = getOmniscentPointer();
    const pointer = getOmniscentPointer();
    // debugger;

    // Set the position of the pointer element
    pointer.style.left = x + 'px';
    pointer.style.top = y + 'px';

    // pointer.style.marginLeft = x + 'px';
    // pointer.style.marginTop = y + 'px';
  });
}

// it is this page creating havoc.
// https://darkreader.org/help/zh-CN/

// a little bit of fucked up.

// let's not do this.
// instead, render the mouse cursor later. could be more accurate and precise.
// document.addEventListener('DOMContentLoaded', function () {

// const mouseEvents = ['mousedown', 'mouseup', 'mousemove'];

// for (let e of mouseEvents) {
//   // Event listener for mouse movement
//   addMouseEventTracer(e)
// }

// });

