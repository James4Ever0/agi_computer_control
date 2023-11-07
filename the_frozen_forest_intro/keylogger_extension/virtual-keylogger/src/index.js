console.log('Starting keylogger for browser');
const serverPort = 4471
const backendUrl = `http://localhost:${serverPort}/browserInputEvent`;
// https://developer.mozilla.org/en-US/docs/Web/Events
// noe we have click/dblclick, keypress events
// how to handle them?
const eventTypes = ['keydown', 'keyup', 'mousedown', 'mouseup', 'mousemove'];
var keylogger_timestamp_private = null; // you can check if this thing still works.
function addSpecificEventListener(event) {
  document.addEventListener(event, (e) => {
    // console.log('event', event, e);
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
      "payload": {'eventType':event, 'data':event_data},
      // "data": JSON.stringify(event_data),

      // data: JSON.stringify(e),
      // data: e,
    };
    // console.log(inputEvent);
    // console.log('payload:', JSON.stringify(inputEvent))
    fetch(`${backendUrl}`, {
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
      console.log(`posted ${event} event`, res);
      // console.log(`posted ${event} event`, res.json());
    }).catch((err) => {
      console.log(`error posting ${event} event`, err);
    });

  });
}
for (const event of eventTypes) {
  addSpecificEventListener(event);
}