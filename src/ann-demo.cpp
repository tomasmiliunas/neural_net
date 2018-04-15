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

int XOR::getResult(int index){

		double max = 0;
		int index1 = 0;

		double* ats=getOutput(index);

		for (int i = 0; i<outputs; i++) {
		 	if (max < ats[i]) {
		 		max = ats[i];
		 		index1 = i;
		 	}
		 }
		 return index1;
}

void XOR_Float::generate(int n)
{
	for (int i = 0; i < n / 4; i++)
	{
		for (float j = 0; j < 2; j++) {
			for (float k = 0; k < 2; k++) {
				float * input = new float[2];
				input[0] = j;
				input[1] = k;
				float * output = new float[2];
				output[0] = j == k;
				output[1] = j != k;
				addSample({ input,output });
			}
		}
	}
}

int XOR_Float::getResult(int index){

		float max = 0;
		int index1 = 0;

		float* ats=getOutput(index);

		for (int i = 0; i<outputs; i++) {
		 	if (max < ats[i]) {
		 		max = ats[i];
		 		index1 = i;
		 	}
		 }
		 return index1;
}

void xor_sample(){
  Topology *topology = new Topology();
	topology->addLayer(2);
	topology->addLayer(5);
	topology->addLayer(4);
	topology->addLayer(2);



	AnnSerialDBL* SerialDBL=new AnnSerialDBL();

	double alpha = 0.7;
  double eta = 0.25;
	SerialDBL -> prepare(topology, alpha, eta);

	SerialDBL->init(NULL);

	XOR xo;
	int dataCount=5000;
	xo.generate(dataCount);
	SerialDBL->train(xo.getInput(0), xo.getOutput(0));


	for (int i = 1; i < xo.getNumberOfSamples(); i++) {
		SerialDBL->train(xo.getInput(i), xo.getOutput(i));
	}

	//Checking results(all combinations 0 and 1)
	for (double i = 0; i < 2; i++) {
		for (double j = 0; j < 2; j++) {
			double input[] = { i ,j };
			double output[] = { 0,0 };

			SerialDBL->feedForward(input, output);
			Sample_Double temp={input,output};
			xo.addSample(temp);
			printf("inout:  %.2f  %.2f\n",xo.getInput(dataCount+i*2+j)[0],xo.getInput(dataCount+i*2+j)[1] );
			printf("output: %.2f  %.2f\n",xo.getOutput(dataCount+i*2+j)[0],xo.getOutput(dataCount+i*2+j)[1] );
			printf("Result: %d\n", xo.getResult(dataCount+i*2+j));
			printf("---------------------------------\n");
		}
	}

	SerialDBL->destroy();
	delete SerialDBL;
}


void xor_sample_Float(){
  Topology *topology = new Topology();
	topology->addLayer(2);
	topology->addLayer(5);
	topology->addLayer(4);
	topology->addLayer(2);



	AnnSerialFLT* serialFlt=new AnnSerialFLT();

	float alpha = 0.7;
  float eta = 0.25;
	serialFlt -> prepare(topology, alpha, eta);

	serialFlt->init(NULL);

	XOR_Float xo;
	int dataCount=5000;
	xo.generate(dataCount);
	serialFlt->train(xo.getInput(0), xo.getOutput(0));


	for (int i = 1; i < xo.getNumberOfSamples(); i++) {
		serialFlt->train(xo.getInput(i), xo.getOutput(i));
	}

	//Checking results(all combinations 0 and 1)
	for (float i = 0; i < 2; i++) {
		for (float j = 0; j < 2; j++) {
			float input[] = { i ,j };
			float output[] = { 0,0 };

			serialFlt->feedForward(input, output);
			Sample_Float temp={input,output};
			xo.addSample(temp);
			printf("inout:  %.2f  %.2f\n",xo.getInput(dataCount+i*2+j)[0],xo.getInput(dataCount+i*2+j)[1] );
			printf("output: %.2f  %.2f\n",xo.getOutput(dataCount+i*2+j)[0],xo.getOutput(dataCount+i*2+j)[1] );
			printf("Result: %d\n", xo.getResult(dataCount+i*2+j));
			printf("---------------------------------\n");
		}
	}

	serialFlt->destroy();
	delete serialFlt;
}

int main (int c, char *v[]) {

  printf("ANN - demo\n\n");

  if(run_tests() == false) return 0;


	printf("\n\n\nDouble rezultatai: \n");
  xor_sample();

	printf("\n\n\nFloat rezultatai: \n");
	xor_sample_Float();

  run_cuda_sample();


 return 0;
}
