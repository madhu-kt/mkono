#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

#define N_SIM 100 // no. of loops
#define N_GEST 10 // no. of gestures to simulate
#define N_INPUT 5 // no. of sensor values
#define N_OUTPUT 1 // no. of outputs
int main()
{
  srand(time(NULL));

  vector< vector <double> > input_vect;
  vector< vector <double> > output_vect;
  input_vect.resize( N_GEST, vector<double>( N_INPUT , 0. ) );
  output_vect.resize( N_GEST, vector<double>( N_OUTPUT , 0. ) );

  ofstream f;
  f.open("../../neural2d/build/xinputData.txt");
  
  for(unsigned n=0;n<input_vect.size();n++)
    {
      for(unsigned i=0;i<input_vect[n].size();++i)
	{
	  input_vect[n][i]=1-2.*(double)rand()/RAND_MAX;
	}

      for(unsigned i=0;i<output_vect[n].size();i++)
	{
	  output_vect[n][i]=-1.+(double)n*2./N_GEST; //1-2.*(double)rand()/RAND_MAX;
	  cout<<output_vect[n][i]<<endl;
	}
    }

  for(unsigned sim=0;sim<N_SIM;sim++)
    {
      for(unsigned n=0;n<N_GEST;n++)
	{
	  f<<"{ ";
	  for(unsigned i=0;i<N_INPUT;i++)
	    {
	      f << setprecision(2) << fixed << input_vect[n][i] << " ";
	    }
	  f<<"} ";
	  for(unsigned i=0;i<N_OUTPUT;i++)
	    {
	      f << setprecision(2) << fixed << output_vect[n][i] << " ";
	    }
	  f << "\n";
	}
    }

  return 0;
}
