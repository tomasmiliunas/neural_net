//#include "ann.h"


// Ann.cpp : Defines the entry point for the console application.
//
#include "ann.h"

using namespace std;

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

int Topology::getInputNeuronCount(){
	return (*ml)[0];
}

int Topology::getOutputNeuronCount(){
	return (*ml)[ml->size()-1];
}

/* sudeti funkcijas i private */

double AnnSerialDBL::f(double x) {
	double y = 1 + exp(-x);
	return 1 / y;
}

double AnnSerialDBL::f_deriv(double x) {
	return exp(-x) / pow((1 + exp(-x)), 2);
}

double AnnSerialDBL::gL(double a, double z, double t) {
	double w = f_deriv(z) * (a - t);
	return w;
}

double AnnSerialDBL::w_gradient(int layer_id, int w_i, int w_j) {
	return a_arr[s[layer_id] + w_i] * gjl[s[layer_id + 1] + w_j];
}



void AnnSerialDBL::calc_gjl()
{
	for (int i = L - 1; i >= 0; i--) {
		for (int j = 0; j < l[i]-1; j++) {
			if (L - 1 == i) {
				gjl[s[i] + j] = gL(a_arr[s[i] + j], z_arr[s[i] + j], t_arr[j]);
			}
			else {
				gjl[s[i] + j] = f_deriv(z_arr[s[i] + j]);
				double sum = 0;
				for (int k = 0; k < l[i + 1] - 1; k++) {
					sum += w_arr[sw[i] + j*(l[i + 1] - 1) + k] * gjl[s[i + 1] + k];
				}
				gjl[s[i] + j] *= sum;
			}
		}
	}
}

//*********
double * Data_Double::getInput(int index)
{
	return data[index].input;
}

double * Data_Double::getOutput(int index)
{
	return data[index].output;
}

void Data_Double::addSample(Sample_Double sample)
{
	data.push_back(sample);
	samples++;
}

void Data_Double::setSizes(int input_size, int output_size)
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

	inputCount = top->getLayerSize(0);
	outputCount = top->getLayerSize(top->getLayerCount() - 1);

	l = new int[top->getLayerCount()];
	s = new int[top->getLayerCount()];

	neuronCount = cTopology->obtainNeuronCount();
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

	//Neuronu kiekiai sluoksnyje
	for (int i = 0; i < L; i++) {
		l[i] = cTopology->getLayerSize(i) + 1;
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
	for (int i = 0; i < inputCount; i++) {
		a_arr[i] = a[i];
	}

	for (int j = 0; j < neuronCount; j++) {
		z_arr[j] = 0;
	}

	calc_feedForward();

	for (int i = 0; i < outputCount; i++) {
		t_arr[i] = b[i];
	}
	calc_gjl();

	//back propogation:
	for (int i = 0; i <L - 1; i++) {//per sluoksnius
		for (int j = 0; j < l[i]; j++) {//per neuronus
			for (int k = 0; k < l[i + 1] - 1; k++) {//per kito sluoksnio neuronus
				dw_arr[sw[i] + k + j*(l[i + 1] - 1)] = delta_w(w_gradient(i, j, k), dw_arr[sw[i] + k + j*(l[i + 1] - 1)]);
				w_arr[sw[i] + k + j*(l[i + 1] - 1)] += dw_arr[sw[i] + k + j*(l[i + 1] - 1)];
			}
		}
	}
}

void AnnSerialDBL::feedForward(double *a, double *b)
{
	for (int i = 0; i < inputCount; i++) {
		a_arr[i] = a[i];
	}

	for (int j = 0; j < neuronCount; j++) {
		z_arr[j] = 0;
	}

	calc_feedForward();

	for (int i = 0; i<outputCount; i++)
		b[i] = a_arr[s[L - 1] + i];
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










//*************

float AnnSerialFLT::f(float x) {
	float y = 1 + exp(-x);
	return 1 / y;
}

float AnnSerialFLT::f_deriv(float x) {
	return exp(-x) / pow((1 + exp(-x)), 2);
}

float AnnSerialFLT::gL(float a, float z, float t) {
	float w = f_deriv(z) * (a - t);
	return w;
}

float AnnSerialFLT::w_gradient(int layer_id, int w_i, int w_j) {
	return a_arr[s[layer_id] + w_i] * gjl[s[layer_id + 1] + w_j];
}



void AnnSerialFLT::calc_gjl()
{
	for (int i = L - 1; i >= 0; i--) {
		for (int j = 0; j < l[i]-1; j++) {
			if (L - 1 == i) {
				gjl[s[i] + j] = gL(a_arr[s[i] + j], z_arr[s[i] + j], t_arr[j]);
			}
			else {
				gjl[s[i] + j] = f_deriv(z_arr[s[i] + j]);
				float sum = 0;
				for (int k = 0; k < l[i + 1] - 1; k++) {
					sum += w_arr[sw[i] + j*(l[i + 1] - 1) + k] * gjl[s[i + 1] + k];
				}
				gjl[s[i] + j] *= sum;
			}
		}
	}
}

//*********
float * Data_Float::getInput(int index)
{
	return data[index].input;
}

float * Data_Float::getOutput(int index)
{
	return data[index].output;
}

void Data_Float::addSample(Sample_Float sample)
{
	data.push_back(sample);
	samples++;
}

void Data_Float::setSizes(int input_size, int output_size)
{
	inputs = input_size;
	outputs = output_size;
}

//****************


//****************
void AnnSerialFLT::prepare(Topology *top, float alpha, float eta)
{
	cTopology = top;
	mAlpha = alpha;
	mEta = eta;

	inputCount = top->getLayerSize(0);
	outputCount = top->getLayerSize(top->getLayerCount() - 1);

	l = new int[top->getLayerCount()];
	s = new int[top->getLayerCount()];

	neuronCount = cTopology->obtainNeuronCount();
	int weightCount = cTopology->obtainWeightCount();

	a_arr = new float[neuronCount];
	z_arr = new float[neuronCount];

	W = new int[top->getLayerCount()];
	sw = new int[top->getLayerCount()];

	w_arr = new float[weightCount];
	dw_arr = new float[weightCount];

	t_arr = new float[top->getLayerSize(top->getLayerCount() - 1)];

	gjl = new float[neuronCount];
}

void AnnSerialFLT::init(float w_arr_1[] = NULL)
{
  L = cTopology->getLayerCount();

	//Neuronu kiekiai sluoksnyje
	for (int i = 0; i < L; i++) {
		l[i] = cTopology->getLayerSize(i) + 1;
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
				w_arr[sw[i] + j] = (float)rand() / double(RAND_MAX);
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

void AnnSerialFLT::train(float *a, float *b)
{
	for (int i = 0; i < inputCount; i++) {
		a_arr[i] = a[i];
	}

	for (int j = 0; j < neuronCount; j++) {
		z_arr[j] = 0;
	}

	calc_feedForward();

	for (int i = 0; i < outputCount; i++) {
		t_arr[i] = b[i];
	}
	calc_gjl();

	//back propogation:
	for (int i = 0; i <L - 1; i++) {//per sluoksnius
		for (int j = 0; j < l[i]; j++) {//per neuronus
			for (int k = 0; k < l[i + 1] - 1; k++) {//per kito sluoksnio neuronus
				dw_arr[sw[i] + k + j*(l[i + 1] - 1)] = delta_w(w_gradient(i, j, k), dw_arr[sw[i] + k + j*(l[i + 1] - 1)]);
				w_arr[sw[i] + k + j*(l[i + 1] - 1)] += dw_arr[sw[i] + k + j*(l[i + 1] - 1)];
			}
		}
	}
}

void AnnSerialFLT::feedForward(float *a, float *b)
{
	for (int i = 0; i < inputCount; i++) {
		a_arr[i] = a[i];
	}

	for (int j = 0; j < neuronCount; j++) {
		z_arr[j] = 0;
	}

	calc_feedForward();

	for (int i = 0; i<outputCount; i++)
		b[i] = a_arr[s[L - 1] + i];
}

void AnnSerialFLT::calc_feedForward()
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

void AnnSerialFLT::destroy()
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

float* AnnSerialFLT::getWeights(){
	return w_arr;
}

float AnnSerialFLT::delta_w(float grad, float dw) {
	return -mEta*grad + mAlpha*dw;
}
