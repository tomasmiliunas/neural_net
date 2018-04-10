//#include "ann.h"


// Ann.cpp : Defines the entry point for the console application.
//



#include "ann.h"

using namespace std;
//Generating random number: either 0 or 1, uniform distribution, for XOR operation. Can remove later if using data from files.
int randint();
double f(double x);
double f_deriv(double x);
double gL(double a, double z, double t);
double w_gradient(int layer_id, int w_i, int w_j, double *a_arr, int *s, double *gll);

void calc_gjl(double *a_arr, double *z_arr, double *t_arr, double *w_arr, int *s, int *sw, int L, int *l, double *gll);
// const double ETA = 0.3;
// const double ALPHA = 0.7;
void file(string filename, double *a, int dydis);

//
// Topology
//

Topology::Topology(){
	ml = new vector<int>();
}

Topology::~Topology(){
	ml->clear();
	delete ml;
}

void Topology::addLayer(int size){
	ml->push_back(size);
}

int Topology::getLayerCount(){
	return ml->size();
}

int Topology::getLayerSize(int index){
	return (*ml)[index];
}

int Topology::obtainNeuronCount(){
	int count = 0;
	for (int i = 0; i < ml->size(); i++)
		count += (*ml)[i] + 1;
	return count;
}

int Topology::obtainWeightCount(){
	int count = 0;
	for (int i = 0; i < ml->size()-1; i++)
		count += ((*ml)[i] + 1)*(*ml)[i+1];
	return count;
}





void print_all(AnnSerialDBL *SerialDBL, int sum, int mult, int i);
void read_W(string filename, double *w_arr);



//returns random int, either 0 or 1
int randint() {
	double r = ((double)rand() / (RAND_MAX));
	int a = 0;
	if (r > 0.5) {
		a = 1;
	}
	else
	{
		a = 0;
	}
	return a;
}

/* sudeti funkcijas i private */

double f(double x) {
	double y = 1 + exp(-x);
	return 1 / y;
}

double f_deriv(double x) {
	return exp(-x) / pow((1 + exp(-x)), 2);
}

double gL(double a, double z, double t) {
	double w = f_deriv(z) * (a - t);
	return w;
}

double w_gradient(int layer_id, int w_i, int w_j, double *a_arr, int *s, double *gll) {
	return a_arr[s[layer_id] + w_i] * gll[s[layer_id + 1] + w_j];
}



void calc_gjl(double *a_arr, double *z_arr, double *t_arr, double *w_arr, int *s, int *sw, int L, int *l, double *gll)
{
	for (int i = L - 1; i >= 0; i--) {
		for (int j = 0; j < l[i]-1; j++) {
			if (L - 1 == i) {
				gll[s[i] + j] = gL(a_arr[s[i] + j], z_arr[s[i] + j], t_arr[j]);
			}
			else {
				gll[s[i] + j] = f_deriv(z_arr[s[i] + j]);
				double sum = 0;
				for (int k = 0; k < l[i + 1] - 1; k++) {
					sum += w_arr[sw[i] + j*(l[i + 1] - 1) + k] * gll[s[i + 1] + k];
				}
				gll[s[i] + j] *= sum;
			}
		}
	}
}

//*********
double * Data::getInput(int index)
{
	return data[index].input;
}

double * Data::getOutput(int index)
{
	return data[index].output;
}

void Data::addSample(Sample sample)
{
	data.push_back(sample);
	samples++;
}

void Data::setSizes(int input_size, int output_size)
{
	inputs = input_size;
	outputs = output_size;
}

//****************


//****************
void AnnSerialDBL::prepare(Topology *top, double alpha, double eta)
{
	cTopology = top;
	mAlpha = alpha;
	mEta = eta;

	input_count = top->getLayerSize(0);
	output_count = top->getLayerSize(top->getLayerCount() - 1);

	l = new int[top->getLayerCount()];
	s = new int[top->getLayerCount()];

	int neuronCount = cTopology->obtainNeuronCount();
	int weightCount = cTopology->obtainWeightCount();

	a_arr = new double[neuronCount];
	z_arr = new double[neuronCount];

	W = new int[top->getLayerCount()];
	sw = new int[top->getLayerCount()];

	w_arr = new double[weightCount];
	dw_arr = new double[weightCount];

	t_arr = new double[top->getLayerSize(top->getLayerCount() - 1)];

	gjl = new double[neuronCount];
}

void AnnSerialDBL::init(double w_arr_1[] = NULL)
{
	L = cTopology->getLayerCount();
	Topology *top = cTopology;

	//Neuronu kiekiai sluoksnyje
	for (int i = 0; i < L; i++) {
		l[i] = top->getLayerSize(i) + 1;
	}

	//Sluoksniu pradzios indeksai
	for (int i = 0; i < L; i++) {
		s[i] = 0;
		for (int j = i; j > 0; j--) {
			s[i] += l[j - 1];
		}
	}

	//Bias neuronai
	for (int i = 0; i < L - 1; i++) {
		a_arr[s[i + 1] - 1] = 1;
	}

	//Svoriu kiekiai l-ame sluoksnyje
	for (int i = 0; i < L - 1; i++) {
		W[i] = l[i] * (l[i + 1] - 1);
		sw[i] = 0;
		if (i != 0) {
			for (int j = 0; j < i; j++) {
				sw[i] += W[j];
			}
		}
		if (w_arr_1 == NULL) {
			for (int j = 0; j < W[i]; j++) {
				w_arr[sw[i] + j] = (double)rand() / double(RAND_MAX);
				dw_arr[sw[i] + j] = 0;
			}
		}
		else {
			for (int j = 0; j < W[i]; j++) {
				w_arr[sw[i] + j] = w_arr_1[sw[i] + j];
				dw_arr[sw[i] + j] = 0;
			}
		}

	}
}

void AnnSerialDBL::train(double *a, double *b)
{
	for (int i = 0; i < input_count; i++) {
		a_arr[i] = a[i];
	}

	for (int j = 0; j < z_count; j++) {
		z_arr[j] = 0;
	}

	calc_feedForward();

	for (int i = 0; i < output_count; i++) {
		t_arr[i] = b[i];
	}
	calc_gjl(a_arr, z_arr, t_arr, w_arr, s, sw, L, l, gjl);

	//back propogation:
	for (int i = 0; i <L - 1; i++) {//per sluoksnius
		for (int j = 0; j < l[i]; j++) {//per neuronus
			for (int k = 0; k < l[i + 1] - 1; k++) {//per kito sluoksnio neuronus
				dw_arr[sw[i] + k + j*(l[i + 1] - 1)] = delta_w(w_gradient(i, j, k, a_arr, s, gjl), dw_arr[sw[i] + k + j*(l[i + 1] - 1)]);
				w_arr[sw[i] + k + j*(l[i + 1] - 1)] += dw_arr[sw[i] + k + j*(l[i + 1] - 1)];
			}
		}
	}
}

void AnnSerialDBL::feedForward(double *a, double *b)
{
	for (int i = 0; i < input_count; i++) {
		a_arr[i] = a[i];
	}

	for (int j = 0; j < z_count; j++) {
		z_arr[j] = 0;
	}

	calc_feedForward();

	double max = 0;
	int index = 0;

	for (int i = 0; i<output_count; i++)
		b[i] = a_arr[s[L - 1] + i];

	// for (int i = 0; i<output_count; i++) {
	// 	cout << " a reiksmes: " << a_arr[s[L - 1] + i] << endl;
	// 	if (max < a_arr[s[L - 1] + i]) {
	// 		max = a_arr[s[L - 1] + i];
	// 		index = i;
	// 	}
	// }
	// for (int i = 0; i < output_count; i++) {
	// 	if (i == index) {
	// 		b[i] = 1;
	// 	}
	// 	else {
	// 		b[i] = 0;
	// 	}
	// }
}

void AnnSerialDBL::calc_feedForward()
{
	for (int i = 0; i < L - 1; i++) {//per sluoksnius einu+
		for (int j = 0; j < l[i]; j++) { //kiek neuronu sluoksnyje+
			for (int k = 0; k < l[i + 1] - 1; k++) {//per sekancio sluoksnio z+
				z_arr[s[i + 1] + k] += w_arr[sw[i] + k + j*(l[i + 1] - 1)] * a_arr[s[i] + j];
			}
		}
		for (int k = 0; k < l[i + 1] - 1; k++) {//per sekancio sluoksnio z
			a_arr[s[i + 1] + k] = f(z_arr[s[i + 1] + k]);
		}
	}
}

void AnnSerialDBL::destroy()
{
	delete[] l;
	l = NULL;
	delete[] s;
	s = NULL;

	delete[] a_arr;
	a_arr = NULL;
	delete[] z_arr;
	z_arr = NULL;

	delete[] W;
	W = NULL;
	delete[] sw;
	sw = NULL;

	delete[] w_arr;
	w_arr = NULL;
	delete[] dw_arr;
	dw_arr = NULL;

	delete[] t_arr;
	t_arr = NULL;

	delete[] gjl;
	gjl = NULL;
}

double* AnnSerialDBL::getWeights(){
	return w_arr;
}

double AnnSerialDBL::delta_w(double grad, double dw) {
	return -mEta*grad + mAlpha*dw;
}

// void read_W(string filename, double *w_arr) {
// 	ifstream myReadFile;
// 	myReadFile.open(filename);
// 	string a;
// 	if (myReadFile.is_open()) {
// 		int i = 0;
// 		while (!myReadFile.eof()) {
// 			myReadFile >> a;
// 			w_arr[i++] = stod(a);
// 		}
// 	}
// 	myReadFile.close();
// }



// void file(string filename, double *a, int dydis) {
// 	ofstream myfile;
// 	myfile.open(filename);
// 	myfile << filename << ";" << endl;
// 	for (int i = 0; i < dydis; i++) {
// 		myfile <<setprecision(16)<< a[i] << ";" << endl;
// 	}
// 	myfile.close();
// }
