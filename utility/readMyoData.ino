// This #include statement was automatically added by the Particle IDE.
#include "Adafruit_BNO055/Adafruit_BNO055.h"

// This #include statement was automatically added by the Particle IDE.
#include "Adafruit_Sensor/Adafruit_Sensor.h"

SYSTEM_THREAD(ENABLED);

int devicesHandler(String data); // forward declaration
void sendData(void);

const unsigned long REQUEST_WAIT_MS = 5000;
const unsigned long RETRY_WAIT_MS = 5000;
const unsigned long SEND_WAIT_MS = 40;

// Sensors

enum State { STATE_REQUEST, STATE_REQUEST_WAIT, STATE_CONNECT, STATE_SEND_DATA, STATE_RETRY_WAIT };
State state = STATE_REQUEST;
unsigned long stateTime = 0;
IPAddress serverAddr;
int serverPort;
char nonce[34];
TCPClient client;

Adafruit_BNO055 bno = Adafruit_BNO055(55);

void setup() {
	Serial.begin(9600);
	Particle.function("devices", devicesHandler);

	Serial.println("Orientation Sensor Test");
    /* Initialise the sensor */
    if(!bno.begin())
    {
        /* There was a problem detecting the BNO055 ... check your connections */
        Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
        while(1);
    }
    
    bno.setExtCrystalUse(true);
}

void loop() {

	switch(state) {
	case STATE_REQUEST:
		if (Particle.connected()) {
			Serial.println("sending devicesRequest");
			Particle.publish("devicesRequest", WiFi.localIP().toString().c_str(), 10, PRIVATE);
			state = STATE_REQUEST_WAIT;
			stateTime = millis();
		}
		break;

	case STATE_REQUEST_WAIT:
		if (millis() - stateTime >= REQUEST_WAIT_MS) {
			state = STATE_RETRY_WAIT;
			stateTime = millis();
		}
		break;

	case STATE_CONNECT:
		if (client.connect(serverAddr, serverPort)) {
			client.println("POST /devices HTTP/1.0");
			client.printlnf("Authorization: %s", nonce);
			client.printlnf("Content-Length: 99999999");
		    client.println();
		    state = STATE_SEND_DATA;
		}
		else {
			state = STATE_RETRY_WAIT;
			stateTime = millis();
		}
		break;

	case STATE_SEND_DATA:
		// In this state, we send data until we lose the connection to the server for whatever
		// reason. We'll to the server again.
		if (!client.connected()) {
			Serial.println("server disconnected");
			client.stop();
			state = STATE_RETRY_WAIT;
			stateTime = millis();
			break;
		}

		if (millis() - stateTime >= SEND_WAIT_MS) {
			stateTime = millis();

			sendData();
		}
		break;

	case STATE_RETRY_WAIT:
		if (millis() - stateTime >= RETRY_WAIT_MS) {
			state = STATE_REQUEST;
		}
		break;
	}
}

void sendData(void) {
	// Called periodically when connected via TCP to the server to update data.
	// Unlike Particle.publish you can push a very large amount of data through this connection,
	// theoretically up to about 800 Kbytes/sec, but really you should probably shoot for something
	// lower than that, especially with the way connection is being served in the node.js server.

    sensors_event_t event; 
    bno.getEvent(&event);

	// Use printf and manually added a \n here. The server code splits on LF only, and using println/
	// printlnf adds both a CR and LF. It's easier to parse with LF only, and it saves a byte when
	// transmitting.
	int lo=3700;
	int hi=3850;
	int val[5] = {map(analogRead(A1),lo,hi,0,255),map(analogRead(A2),lo,hi,0,255),map(analogRead(A3),lo,hi,0,255),map(analogRead(A4),lo,hi,0,255),map(analogRead(A5),lo,hi,0,255)};
	
	client.printf("%.1f,%.1f,%.1f,%3d,%3d,%3d,%3d\n",
			event.orientation.x,event.orientation.y,event.orientation.z,val[0],val[1],val[2],val[3]);

	// roll,pitch,heading,altitude,pressure,temperature
	// Example:
	// 0.000,-0.449,-49.091,263.317,982.02,27.5
	// -0.224,-0.224,-48.955,262.719,982.09,27.5
	// 0.000,-0.449,-48.704,262.719,982.09,27.5
	// 0.000,-0.449,-48.704,262.890,982.07,27.5
}

int devicesHandler(String data) {
	Serial.printlnf("devicesHandler data=%s", data.c_str());
	int addr[4];

	if (sscanf(data, "%u.%u.%u.%u,%u,%32s", &addr[0], &addr[1], &addr[2], &addr[3], &serverPort, nonce) == 6) {
		serverAddr = IPAddress(addr[0], addr[1], addr[2], addr[3]);
		Serial.printlnf("serverAddr=%s serverPort=%u nonce=%s", serverAddr.toString().c_str(), serverPort, nonce);
		state = STATE_CONNECT;
	}
	return 0;
}
