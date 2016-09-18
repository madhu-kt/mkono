import processing.serial.*;

Serial myPort;
int xPos = 1;
int lastxPos=1;
int lastyPos=0;

void setup()
{
  size(600,400);
  println(Serial.list());
  myPort = new Serial(this, Serial.list()[0], 115200);
  background(0);
}

void draw()
{
}

void serialEvent(Serial myPort)
{
  String inString = myPort.readStringUntil('\n');
  if(inString != null)
  {
    inString = trim(inString);
    float inByte = float(inString);
    inByte = map(inByte, 0, 1023, 0, height);
    stroke(127,34,255);
    strokeWeight(4);
    line(lastxPos, lastyPos, xPos, height - inByte);
    lastyPos = int(height-inByte);
    if(xPos >= width)
    {
      xPos = 0;
      lastxPos = 0;
      background(0);
    }
    else
    {
      xPos++;
    }
  }
}