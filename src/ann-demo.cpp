#include "ann-demo.h"

void XOR::generate(int n)
{
	for (int i = 0; i < n / 4; i++)
	{
		for (double j = 0; j < 2; j++) {
			for (double k = 0; k < 2; k++) {
				double * input = new double[2];
				input[0] = j;
				input[1] = k;
				double * output = new double[2];
				output[0] = j == k;
				output[1] = j != k;
				addSample({ input,output });
			}
		}
	}
}

void xor_sample(){
  Topology *topology = new Topology();
	topology->addLayer(2);
	topology->addLayer(5);
	topology->addLayer(4);
	topology->addLayer(2);



	AnnSerialDBL  *SerialDBL=new AnnSerialDBL();

	double alpha = 0.7;
  double eta = 0.25;
	SerialDBL -> prepare(topology, alpha, eta);

	SerialDBL->init(NULL);

	XOR xo;
	xo.generate(5000);
	SerialDBL->train(xo.getInput(0), xo.getOutput(0));


	// file("()a_arr.csv", SerialDBL->a_arr, sum);
	// file("()z_arr.csv", SerialDBL->z_arr, sum);
	// file("()w_arr.csv", SerialDBL->w_arr, mult);
	// file("()dw_arr.csv", SerialDBL->dw_arr, mult);
	// file("()g_arr.csv", SerialDBL->gjl, sum);

	for (int i = 1; i < xo.getNumberOfSamples(); i++) {
		SerialDBL->train(xo.getInput(i), xo.getOutput(i));
	}

	//Checking results(all combinations 0 and 1)
	for (double i = 0; i < 2; i++) {
		for (double j = 0; j < 2; j++) {
			double input[] = { i ,j };
			double output[] = { 0,0 };
			SerialDBL->feedForward(input, output);

			cout << endl << "input : " << input[0] << "   " << input[1] << endl;
			cout << endl << "output: " << output[0] << "   " << output[1] << endl << endl;
			cout << "---------------------------------------------------" << endl;
		}
	}

	SerialDBL->destroy();
	delete SerialDBL;

}

int main (int c, char *v[]) {

  printf("ANN - demo\n\n");

  if(run_tests() == false) return 0;

  xor_sample();

  run_cuda_sample();


 return 0;
}
