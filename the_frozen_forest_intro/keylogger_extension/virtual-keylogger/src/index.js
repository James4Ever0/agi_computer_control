const serverPort = 4471
const backendUrl = `http://localhost:${serverPort}/browserInputEvent`;
const eventTypes = ['keydown', 'keyup', 'mousedown', 'mouseup', 'mousemove'];
var keylogger_timestamp_private = null; // you can check if this thing still works.

function addSpecificEventListener(event){
  document.addEventListener(event, (e) => {
    keylogger_timestamp_private = new Date();
    const inputEvent = {
      eventType: event,
      timestamp: timestamp,
      data: e,
    };
    fetch(`${backendUrl}/`, {
      method: "POST",
      mode: "cors",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(inputEvent),
    }).then((res) => {
      console.log(`posted ${event} event`, res);
    }).catch((err) => {
      console.log(`error posting ${event} event`, err);
    });

  });
}
for (const event of eventTypes) {
  addSpecificEventListener(event);
}