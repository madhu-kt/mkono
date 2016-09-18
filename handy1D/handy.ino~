#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;

#define N_SMOOTHING_SAMPLES 4
#define N_MAX_TRIALS 100000
#define N_TRAINING_SAMPLES 4
#define OUTPUT_SCALER 10.0

#define REST_ADC 200.0
#define N_SENSORS 5
#define LOGGING_WAIT_TIME 1000 //ms
#define CALIBRATION_TIME 100 //ms
#define WAIT_TIME 3000 //ms

void trainEvent(void);
void activeEvent(void);

int trainPin = D3;
int activePin = D4;

int ledRedPin = D2;
int ledGreenPin = D5;

int sense1Pin = A1;
int sense2Pin = A2;
int sense3Pin = A3;
int sense4Pin = A4;
int sense5Pin = A5;

struct Connection
{
  double weight;
  double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ****************** class Neuron ******************
class Neuron
{
public:
  Neuron(unsigned numOutputs, unsigned myIndex);
  void setOutputVal(double val) { m_outputVal = val; }
  double getOutputVal(void) const { return m_outputVal; }
  void feedForward(const Layer &prevLayer);
  void calcOutputGradients(double targetVal);
  void calcHiddenGradients(const Layer &nextLayer);
  void updateInputWeights(Layer &prevLayer);

private:
  static double eta;   // [0.0..1.0] overall net training rate
  static double alpha; // [0.0..n] multiplier of last weight change (momentum)
  static double transferFunction(double x);
  static double transferFunctionDerivative(double x);
  static double randomWeight(void) { return rand() / (10.0*double(RAND_MAX)); }
  double sumDOW(const Layer &nextLayer) const;
  double m_outputVal;
  vector<Connection> m_outputWeights;
  unsigned m_myIndex;
  double m_gradient;
};

double Neuron::eta = 0.15;    // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.5;   // momentum, multiplier of last deltaWeight, [0.0..1.0]


void Neuron::updateInputWeights(Layer &prevLayer)
{
  // The weights to be updated are in the Connection container
  // in the neurons in the preceding layer

  for (unsigned n = 0; n < prevLayer.size(); ++n) {
    Neuron &neuron = prevLayer[n];
    double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
	  // Individual input, magnified by the gradient and train rate:
                eta
	  * neuron.getOutputVal()
                * m_gradient
	  // Also add momentum = a fraction of the previous delta weight;
                + alpha
	  * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
  }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
  double sum = 0.0;

  // Sum our contributions of the errors at the nodes we feed.

  for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
    sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
  }

  return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
  double dow = sumDOW(nextLayer);
  m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
  double delta = targetVal - m_outputVal;
  m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
  // tanh - output range [-1.0..1.0]

  return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
  // tanh derivative
  return 1.0 - tanh(x) * tanh(x);
}

void Neuron::feedForward(const Layer &prevLayer)
{
  double sum = 0.0;

  // Sum the previous layer's outputs (which are our inputs)
  // Include the bias node from the previous layer.

  for (unsigned n = 0; n < prevLayer.size(); ++n) {
    sum += prevLayer[n].getOutputVal() *
      prevLayer[n].m_outputWeights[m_myIndex].weight;
  }

  m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
  for (unsigned c = 0; c < numOutputs; ++c) {
    m_outputWeights.push_back(Connection());
    m_outputWeights.back().weight = randomWeight();
  }

  m_myIndex = myIndex;
}


// ****************** class Net ******************
class Net
{
public:
    Net();
  Net(const vector<unsigned> &topology);
  void feedForward(const vector<double> &inputVals);
  void backProp(const vector<double> &targetVals);
  void getResults(vector<double> &resultVals) const;
  double getNetError(void) const { return m_error; }
  double getRecentAverageError(void) const { return m_recentAverageError; }
  
private:
  vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
  double m_error;
  double m_recentAverageError;
  static double m_recentAverageSmoothingFactor;
};

Net::Net()
{
    Serial.println("Creating new neural network...");
}

double Net::m_recentAverageSmoothingFactor = N_SMOOTHING_SAMPLES; // Number of training samples to average over


void Net::getResults(vector<double> &resultVals) const
{
  resultVals.clear();

  for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
    resultVals.push_back(m_layers.back()[n].getOutputVal());
  }
}

void Net::backProp(const vector<double> &targetVals)
{
  // Calculate overall net error (RMS of output neuron errors)

  Layer &outputLayer = m_layers.back();
  m_error = 0.0;

  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    double delta = targetVals[n] - outputLayer[n].getOutputVal();
    m_error += delta * delta;
  }
  m_error /= outputLayer.size() - 1; // get average error squared
  m_error = sqrt(m_error); // RMS

  // Implement a recent average measurement
  
  
  m_recentAverageError =
    (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
    / (m_recentAverageSmoothingFactor + 1.0);
  
  // Calculate output layer gradients
  
  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    outputLayer[n].calcOutputGradients(targetVals[n]);
  }
  
  // Calculate hidden layer gradients
  
  for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
    Layer &hiddenLayer = m_layers[layerNum];
    Layer &nextLayer = m_layers[layerNum + 1];
    
    for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
      hiddenLayer[n].calcHiddenGradients(nextLayer);
    }
  }
  
  // For all layers from outputs to first hidden layer,
  // update connection weights
  
  for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
    Layer &layer = m_layers[layerNum];
    Layer &prevLayer = m_layers[layerNum - 1];
    
    for (unsigned n = 0; n < layer.size() - 1; ++n) {
      layer[n].updateInputWeights(prevLayer);
    }
  }
}

void Net::feedForward(const vector<double> &inputVals)
{
  if(inputVals.size() != m_layers[0].size() - 1)
  {
      Serial.println("MISMATCH");
  }

  // Assign (latch) the input values into the input neurons
  for (unsigned i = 0; i < inputVals.size(); ++i) {
    m_layers[0][i].setOutputVal(inputVals[i]);
  }

  // forward propagate
  for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
    Layer &prevLayer = m_layers[layerNum - 1];
    for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
      m_layers[layerNum][n].feedForward(prevLayer);
    }
  }
}

Net::Net(const vector<unsigned> &topology)
{
  unsigned numLayers = topology.size();
  for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
    m_layers.push_back(Layer());
    unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

    // We have a new layer, now fill it with neurons, and
    // add a bias neuron in each layer.
    for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
      m_layers.back().push_back(Neuron(numOutputs, neuronNum));
    }

    // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
    m_layers.back().back().setOutputVal(1.0);
    }
}

void showVectorVals(string label, vector<double> &v)
{
    Serial.printf("%s ",label.c_str());
    for (unsigned i = 0; i < v.size(); ++i) {
        Serial.print(v[i]);
        Serial.print(" ");
    }
    Serial.println();
}

void showApproxVectorVals(string label, vector<double> &v)
{
    Serial.printf("%s ",label.c_str());
    for (unsigned i = 0; i < v.size(); ++i) {
        Serial.print(round(v[i]*10.0)/10.0);
        Serial.print(" ");
    }
    Serial.println();
}

//calculates the standard deviation of a vector's elements
double getStDev(vector<double> data, double &mean)
{
    mean=0;
    float sum_deviation=0.0;
    int i;
    for(i=0; i<data.size();++i)
    {
        mean+=data[i];
    }
    mean=mean/data.size();
    for(i=0; i<data.size();++i)
    sum_deviation+=(data[i]-mean)*(data[i]-mean);
    return sqrt(sum_deviation/data.size());    
}

vector<unsigned> topology;
vector<double> targetVals, inputVals, resultVals;
vector< vector<double> > targetSet;
vector< vector<double> > trainingSet;

void readCalibs(void);
//setup a timer that calibrates the sensors
Timer timerCalibs(CALIBRATION_TIME, readCalibs);
//setup a timer that waits between calibration stages
void pleaseWait(void);
Timer timerWait(WAIT_TIME, pleaseWait);

void readTrain(void);
//setup a timer that gets sensor data after 1 second
Timer timerTrain(LOGGING_WAIT_TIME, readTrain);

void readActive(void);
//setup a timer that gets sensor data every 1 second
Timer timerActive(LOGGING_WAIT_TIME, readActive);

double activeVal = 0.;
String sysState = 0;
void setup()
{
    pinMode(D7, OUTPUT); //heart beat!
    pinMode(trainPin, INPUT_PULLDOWN);
    pinMode(activePin, INPUT_PULLDOWN);
    pinMode(ledRedPin, OUTPUT);
    pinMode(ledGreenPin, OUTPUT);
    
    topology.push_back(N_SENSORS);
    topology.push_back(3);
    topology.push_back(1);
    
    Serial.begin(9600);
    
    //attachInterrupt(trainPin, trainEvent, FALLING);
    attachInterrupt(activePin, activeEvent, FALLING);
    
    Particle.function("startCalib",startCalib);
    Particle.function("startTrain",startTrain);
    Particle.function("startActive",startActive);
    Particle.function("linkAction",linkAction);
    Particle.variable("sysState",sysState);
    sysState="init...";
}

volatile bool ledState = LOW;
bool state = LOW;

//starts the timer that begins the calibration cycle
String calibStr;
int startCalib(String str)
{
    //toggle LED
    ledState=!ledState;
    digitalWrite(D7,ledState);
    sysState = "lo calib...";
    
    timerCalibs.start();
    return 1;
}

//starts the timer that begins the training cycle & sets the target (expected) values
String targetStr;
int startTrain(String str)
{
    //toggle LED
    ledState=!ledState;
    digitalWrite(D7,ledState);
    sysState = "training...";
    
    String s = "Training gesture: "+str+"...";
    Serial.println(s);
    targetStr=str;
    timerTrain.start();
    return 1;
}

int startActive(String str)
{
    //toggle LED
    ledState=!ledState;
    digitalWrite(D7,ledState);
    sysState = "active...";
    
    String s = "Trigger: "+str+". Starting active mode...";
    timerActive.start();
    return 1;
}

volatile bool activateActive = LOW;
vector<double> rawVals[N_SENSORS]; //array of 5 vectors
double calibrations[2][N_SENSORS]={{0},{0}};
double stdevs[2][N_SENSORS]={{0},{0}};
volatile int calibCount=0;
void readCalibs(void)
{
    ++calibCount; //keep track of how many times the sensors are sampled
    
    //Serial.print(calibCount);
    //Serial.print(".");
    //read values from all sensors every 100ms and add to vector
    vector<double> currentVals={analogRead(sense1Pin),analogRead(sense2Pin),analogRead(sense3Pin),analogRead(sense4Pin),analogRead(sense5Pin)};
    //showVectorVals("values: ",currentVals);
    
    //feed values into vector array
    for(int i=0;i<N_SENSORS;++i)
    {
        rawVals[i].push_back(currentVals[i]);
    }
    
    if(calibCount==50) //if 5 seconds have passed, calculate low baseline
    {
        for(int i=0;i<N_SENSORS;++i)
        {
            stdevs[0][i]=getStDev(rawVals[i],calibrations[0][i]); //rawVals[i] is a vector of all the readings for the i'th sensor
            rawVals[i].clear(); //prepare vector set for next set of inputs
        }
        /*
        String s=String(calibrations[0][0])+" "+String(calibrations[0][1])+" "+String(calibrations[0][2])+" "+String(calibrations[0][3])+" "+String(calibrations[0][4]);
        Particle.publish(s);
        s=String(stdevs[0][0])+" "+String(stdevs[0][1])+" "+String(stdevs[0][2])+" "+String(stdevs[0][3])+" "+String(stdevs[0][4]);
        Particle.publish(s);
        */
        
        timerCalibs.stop();
        timerWait.start();
        
        //toggle LED
        ledState=!ledState;
        digitalWrite(D7,ledState);
        sysState = "make a fist!";
    }
    
    
    else if(calibCount==100) //if 10 seconds have passed, calculate high baseline & end calibration
    {
        activateActive = HIGH;
        calibCount=0;
        
        for(int i=0;i<N_SENSORS;++i)
        {
            stdevs[1][i]=getStDev(rawVals[i],calibrations[1][i]);
            rawVals[i].clear(); //prepare vector set for next set of inputs
            
            if(calibrations[0][i]>calibrations[1][i])
            {
                Serial.println("Calibration failed. Try again!");
                //toggle LED
                ledState=!ledState;
                digitalWrite(D7,ledState);
                sysState = "calib failed";
                break;
            }
        }
        
        //toggle LED
        ledState=!ledState;
        digitalWrite(D7,ledState);
        sysState = "calib done!";
        /*
        String s=String(calibrations[1][0])+" "+String(calibrations[1][1])+" "+String(calibrations[1][2])+" "+String(calibrations[1][3])+" "+String(calibrations[1][4]);
        Particle.publish(s);
        s=String(stdevs[1][0])+" "+String(stdevs[1][1])+" "+String(stdevs[1][2])+" "+String(stdevs[1][3])+" "+String(stdevs[1][4]);
        Particle.publish(s);
        */
        timerCalibs.stop(); 
    }
    
}

void pleaseWait(void)
{
    //toggle LED
    ledState=!ledState;
    digitalWrite(D7,ledState);
    sysState = "hi calib...";
    timerWait.stop();
    timerCalibs.start();
}

volatile int readCount=0;
volatile bool enableActiveMode = false;
volatile bool fubar = true;
Net myNet;

void readTrain(void)
{
    activeVal=-999; //idiot-check
    timerActive.stop();
    //clear vectors
    inputVals.clear();
    targetVals.clear();
    //pass in 'normalized' input
    int inputVal[N_SENSORS]={analogRead(sense1Pin),analogRead(sense2Pin),analogRead(sense3Pin),analogRead(sense4Pin),analogRead(sense5Pin)};
    
    showVectorVals("Raw: ", inputVals);
    
    //'normalization' of NN input
    for(int i=0;i<N_SENSORS;++i)
    {
        if(inputVal[i]<calibrations[1][i] && inputVal[i]>calibrations[0][i])
        {
            inputVals.push_back(map(inputVal[i],calibrations[0][i],calibrations[1][i],0,10000)/10000.0);
        }
        else if(inputVal[i]<calibrations[0][i])
        {
            inputVals.push_back(0);
        }
        else if(inputVal[i]>calibrations[1][i])
        {
            inputVals.push_back(1);
        }
        //inputVals.push_back( (inputVal[i]>calibrations[0][i]) ? (inputVal[i]<calibrations[1][i] ? map(inputVal[i],calibrations[0][i],calibrations[1][i],0,10000)/10000.0 : 1 ) : 0 );
    }
    
    targetVals.push_back(targetStr.toInt()/OUTPUT_SCALER);
    
    trainingSet.push_back(inputVals);
    targetSet.push_back(targetVals);

    String s_d = "Added <"+String(trainingSet.back()[0])+" "+String(trainingSet.back()[1])+" "+String(trainingSet.back()[2])+" "+String(trainingSet.back()[3])+" "+String(trainingSet.back()[4])+" to training set.>";
    Serial.println(s_d);
    s_d = "Added <"+String(targetSet.back()[0])+"> to target set.";
    Serial.println(s_d);
    
    if(readCount>N_TRAINING_SAMPLES)
    {
        readCount=0;
        timerTrain.stop();
        Serial.println("Starting neural network training...");
        
        /////////////////////////////////////////////////////////////////////////////////////////
        ///////////////
        ///////////////         NEURAL NET TRAINING
        ///////////////
        /////////////////////////////////////////////////////////////////////////////////////////
        
        myNet = Net(topology);
        
        unsigned n=0;
        
        do
        {
            unsigned i = n%trainingSet.size();
            
        	//Get new input data from vector and feed it forward:
        	myNet.feedForward(trainingSet[i]);
        	//showVectorVals("Feeding in: ",trainingSet[i]);
        	// Collect the net's actual output results:
        	myNet.getResults(resultVals);
            /*  
            Serial.printf("Pass %d :: Round %d",n,i);
            showVectorVals(": Inputs:", trainingSet[i]);
            showVectorVals("Outputs:", resultVals);
            // Report how well the training is working, average over recent samples:
            Serial.print("Net recent average error: ");
            Serial.print(myNet.getNetError());
            Serial.println();
        	
        	showVectorVals("Targets:", targetSet[i]);
        	*/
        	if(targetSet[i].size() != topology.back())
        	{
        	    Serial.println("TARGET VALS & TOPOLOGY MISMATCH");
        	    break;
        	}
        	
        	// Tell the net what the outputs should have been:
        	myNet.backProp(targetSet[i]);
        
        	if(n==N_MAX_TRIALS)
        	{
        	    Serial.println("Maximum training limit reached. Neural net did not converge :(");
        	    break;
        	}
        	++n;
        }
        while(myNet.getNetError() > 0.0001);
        
        //toggle LED
        ledState=!ledState;
        digitalWrite(D7,ledState);
        sysState = "done!";
        
        Particle.publish("Training complete!");
        Serial.println("Training complete");
        enableActiveMode = true;
        //could add a timer that waits about 5 seconds and automatically starts active-mode
    }
    else
    {
        ++readCount;
    }
}

void readActive(void)
{
    inputVals.clear();
    resultVals.clear();
    //pass in 'normalized' input
    int inputVal[N_SENSORS]={analogRead(sense1Pin),analogRead(sense2Pin),analogRead(sense3Pin),analogRead(sense4Pin),analogRead(sense5Pin)};
    for(int i=0;i<N_SENSORS;++i)
    {
        if(inputVal[i]<calibrations[1][i] && inputVal[i]>calibrations[0][i])
        {
            inputVals.push_back(map(inputVal[i],calibrations[0][i],calibrations[1][i],0,10000)/10000.0);
        }
        else if(inputVal[i]<calibrations[0][i])
        {
            inputVals.push_back(0);
        }
        else if(inputVal[i]>calibrations[1][i])
        {
            inputVals.push_back(1);
        }
        //inputVals.push_back( (inputVal[i]>calibrations[0][i]) ? (inputVal[i]<calibrations[1][i] ? map(inputVal[i],calibrations[0][i],calibrations[1][i],0,10000)/10000.0 : 1 ) : 0 );
    }
    //String s_i = "Added <"+String(inputVals[0])+" "+String(inputVals[1])+" "+String(inputVals[2])+" "+String(inputVals[3])+" "+String(inputVals[4])+" to input set.>";
    //Serial.println(s_i);
    // Feed real-time input
    showVectorVals("Neural input:", inputVals);
    myNet.feedForward(inputVals);
    // Collect the net's actual output results:
	myNet.getResults(resultVals);
	showVectorVals("Neural output:", resultVals);
	showApproxVectorVals("Approx. neural output:", resultVals);
	
	//since you know in advance that your NN has only one output you can just update a cloud variable (remember that there's a limit to how many cloud variables are allowed); figure out a better way later!
	sysState=String(resultVals[0]);
}

//loads linked values to vector of ints
int linkAction(String str)
{
    vector<int> v;
    Serial.println(str);
    int from=0;
    for(int i=0;i<str.length();i++)
    {
        int count=0;
        if(str.charAt(i)=='_')
        {
            v.push_back(str.substring(from,i).toInt());
            from=i+1;
        }
    }
    v.push_back(str.substring(from,str.length()).toInt());
    Particle.publish(str);
    state=!state;
    digitalWrite(D7,state);
    return 1;
}
/*
//Debouncing inside ISR
long lastTrainDebounceTime = 0;

void trainEvent(void)   //this ISR triggers when the train button is pressed. each time you press the switch,
{
  if ((millis() - lastTrainDebounceTime) > debounceDelay) //if current time minus the last trigger time is greater than
  {                                                  //the delay (debounce) time, button is completley closed.
    lastTrainDebounceTime = millis();
    
    //switch was pressed, do whatever you need to here
    Serial.println("Starting training mode...");
    timerTrain.startFromISR();
  }
}*/

//Debouncing inside ISR
long debounceDelay = 50; // 50 mS
long lastActiveDebounceTime = 0;
void activeEvent(void)   //this ISR triggers when the active button is pressed. each time you press the switch,
{
  if ((millis() - lastActiveDebounceTime) > debounceDelay) //if current time minus the last trigger time is greater than
  {                                                  //the delay (debounce) time, button is completley closed.
    lastActiveDebounceTime = millis();
    
    //switch was pressed, do whatever you need to here
    if(enableActiveMode)
    {
        Serial.println("Starting active mode...");
        timerActive.startFromISR();
    }
  }
}
