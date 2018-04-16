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

//************************************************************************
//                           Paveiksliukai
//************************************************************************

//Nuskaito inputus is duomenu failu
void PictureData::readMnist(string filename, vector<double*> &arr)
{
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        for(int i = 0; i < number_of_images; ++i)
        {
            double* tp = new double[784];
            int index;
            for(int r = 0; r < n_rows; r++)
            {
                for(int c = 0; c < n_cols; c++)
                {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    index = r * n_cols + c;
                    tp[index] = ((double)temp / 255 * 1.0);
                }
            }
            arr.push_back(tp);
        }
    }
}

//Nuskaito labelius is failo
void PictureData::readMnistLabel(string filename, vector<double> &vec)
{
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec.push_back((double)temp);
        }
    }
}


//int PictureData::printMaxValue(int index, AnnSerialDBL &SerialDBL) {
//  int ind = 0;
//  double max = 0;
//  for (int i = 0; i < SerialDBL.l[SerialDBL.L - 1]; i++) {
//      if (SerialDBL.a_arr[SerialDBL.s[SerialDBL.L - 1]] >= max) {
//          max = SerialDBL.a_arr[SerialDBL.s[SerialDBL.L - 1]];
//          ind = i;
//      }
//  }
//  return ind;
//}

//************************************************************************

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

void pic_sample() {
	string train_labels = "train-labels.idx1-ubyte";
  string train_images = "train-images.idx3-ubyte";
  string test_labels = "t10k-labels.idx1-ubyte";
  string test_images = "t10k-images.idx3-ubyte";

	Topology *topology = new Topology();
	topolygy->addLayer(784);
	topolygy->addLayer(16);
	topolygy->addLayer(10);

	AnnSerialDBL* SerialDBL=new AnnSerialDBL();

	float alpha = 0.2;
  float eta = 0.5;
	serialFlt -> prepare(topology, alpha, eta);

	serialFlt->init(NULL);

	PictureData pictures;

	trainData(pictures, train_images, train_labels, SerialDBL);
  testNet(pictures, test_images, test_labels, SerialDBL);
}

int main (int c, char *v[]) {

  printf("ANN - demo\n\n");

  if(run_tests() == false) return 0;


	printf("\n\n\nDouble rezultatai: \n");
  xor_sample();

	printf("\n\n\nFloat rezultatai: \n");
	xor_sample_Float();

	pic_sample();

  //run_cuda_sample();


 return 0;
}
