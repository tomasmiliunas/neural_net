//#include "ann.h"


// Ann.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cmath>
#include <math.h> 
#include <vector>
#include <iostream>	
#include <iomanip>
#include <fstream>
#include <string>
#include <array>

using namespace std;
//Generating random number: either 0 or 1, uniform distribution, for XOR operation. Can remove later if using data from files.
int randint();
double f(double x);
double f_deriv(double x);
double gL(double a, double z, double t);
double w_gradient(int layer_id, int w_i, int w_j, double *a_arr, int *s, double *gll);
double delta_w(double grad, double dw);
void calc_gjl(double *a_arr, double *z_arr, double *t_arr, double *w_arr, int *s, int *sw, int L, int *l, double *gll);
const double ETA = 0.2;
const double ALPHA = 0.5;
void file(string filename, double *a, int dydis);

struct Topology
{
	std::vector<int> l;//kiekiai sluoksnyje
} topolygy;

struct Sample
{
	double * input;
	double * output;
};
class Data
{
public:
	int getNumberOfInputs() { return inputs; }
	int getNumberOfOutputs() { return outputs; }

	double * getInput(int index);

	double * getOutput(int index);

	int getNumberOfSamples() { return samples; }

	void addSample(Sample sample);

	void setSizes(int input_size, int output_size);

protected:
	std::vector<Sample> data;
	int inputs;
	int outputs;
	int samples = 0;
};

class XOR : public Data
{
public:
	void generate(int n);

	XOR()
	{
		inputs = 2;
		outputs = 2;
		samples = 0;
	}
	void printInputs(int index);

	void printOutputs(int index);
};

class AnnBase {
public:
	virtual void prepare(Topology top) = 0;
	virtual void init(Topology top, double w_arr_1[]) = 0;
	virtual void train(double *a, double *b) = 0;
	virtual void feedForward(double *a, double *b) = 0;
	virtual void destroy() = 0;
private:
	virtual void calc_feedForward() = 0;
};

class AnnSerialDBL : public AnnBase {
public:
	void prepare(Topology top);
	void init(Topology top, double w_arr_1[]);
	void train(double *a, double *b);
	void feedForward(double *a, double *b);
	void destroy();
	AnnSerialDBL() {};
private:
	void calc_feedForward();
public:
	int z_count;//temp var to keep the length of z, so z could be reset for calcs.
	int input_count;
	int output_count;
	int L;
	int * l;
	int * s;
	double * a_arr;
	double * z_arr;
	int * W;
	int * sw;
	double * w_arr;
	double * dw_arr;
	double * t_arr;
	double * gjl;
};
void print_all(AnnSerialDBL *SerialDBL, int sum, int mult, int i);
void read_W(string filename, double *w_arr);
void calc_sizes(int &sum, int &mult, Topology top);
void test();
void setTopology(Topology &top, int a[], size_t length);
int main()
{
	//test();
	int arr[] = {2,5,4,2};
	setTopology(topolygy, arr, sizeof(arr)/sizeof(arr[0]));

	AnnSerialDBL  *SerialDBL=new AnnSerialDBL();

	int sum = 0;
	int mult = 0;
	calc_sizes(sum, mult, topolygy);

	SerialDBL -> prepare(topolygy);

	double * a1 = new double[mult];
	//read_W("W_ARR.txt", a1);
	SerialDBL->init(topolygy, NULL);
	delete[] a1;
	a1 = NULL;

	XOR xo;
	xo.generate(10000);
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

			cout << endl << "input : " << input[0] << "   " << input[1] << endl;
			cout << endl << "output: " << output[0] << "   " << output[1] << endl << endl;
			cout << "---------------------------------------------------" << endl;
		}
	}

	SerialDBL->destroy();
	delete SerialDBL;

	return 0;
}
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

double delta_w(double grad, double dw) {
	return (-ETA)*grad + ALPHA*dw;
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

//****************
void AnnSerialDBL::prepare(Topology top)
{
	input_count = top.l.at(0);
	output_count = top.l.at(top.l.size() - 1);

	l = new int[top.l.size()];
	s = new int[top.l.size()];

	int sum = 0;
	int mult = 0;
	for (int i = 0; i < top.l.size(); i++) {
		sum += top.l.at(i) + 1;
	}
	z_count = sum;
	for (int i = 0; i < top.l.size() - 1; i++) {
		mult += (top.l.at(i) + 1)*top.l.at(i + 1);
	}
	a_arr = new double[sum];
	z_arr = new double[sum];

	W = new int[top.l.size()];
	sw = new int[top.l.size()];

	w_arr = new double[mult];
	dw_arr = new double[mult];

	t_arr = new double[top.l.at(top.l.size() - 1)];

	gjl = new double[sum];
}

void AnnSerialDBL::init(Topology top, double w_arr_1[] = NULL)
{
	L = top.l.size();
	//Neuronu kiekiai sluoksnyje
	for (int i = 0; i < top.l.size(); i++) {
		l[i] = top.l.at(i) + 1;
	}

	//Sluoksniu pradzios indeksai
	for (int i = 0; i < top.l.size(); i++) {
		s[i] = 0;
		for (int j = i; j > 0; j--) {
			s[i] += l[j - 1];
		}
	}

	//Bias neuronai
	for (int i = 0; i < top.l.size() - 1; i++) {
		a_arr[s[i + 1] - 1] = 1;
	}

	//Svoriu kiekiai l-ame sluoksnyje
	for (int i = 0; i < top.l.size() - 1; i++) {
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

	for (int i = 0; i<output_count; i++) {
		cout << " a reiksmes: " << a_arr[s[L - 1] + i] << endl;
		if (max < a_arr[s[L - 1] + i]) {
			max = a_arr[s[L - 1] + i];
			index = i;
		}
	}
	for (int i = 0; i < output_count; i++) {
		if (i == index) {
			b[i] = 1;
		}
		else {
			b[i] = 0;
		}
	}
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

void read_W(string filename, double *w_arr) {
	ifstream myReadFile;
	myReadFile.open(filename);
	string a;
	if (myReadFile.is_open()) {
		int i = 0;
		while (!myReadFile.eof()) {
			myReadFile >> a;
			w_arr[i++] = stod(a);
		}
	}
	myReadFile.close();
}

void calc_sizes(int &sum, int &mult, Topology top) {
	for (int i = 0; i < topolygy.l.size(); i++) {
		sum += topolygy.l.at(i) + 1;
	}
	for (int i = 0; i < topolygy.l.size() - 1; i++) {
		mult += (topolygy.l.at(i) + 1)*topolygy.l.at(i + 1);
	}
}
void setTopology(Topology &top, int a[], size_t length){
	for(int i=0;i<length;i++){
		top.l.push_back(a[i]);
	}
}
