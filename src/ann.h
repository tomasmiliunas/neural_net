#ifndef ANN_HEADER
#define ANN_HEADER

#include <helper_cuda.h>
#include <cmath>
#include <cstdlib>

#include <cmath>
#include <math.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <array>

class Topology {
	private:
		std::vector<int> *ml;
	public:
		Topology();
		~Topology();
		void addLayer(int size);

		int getLayerCount();
		int getLayerSize(int index);

		int obtainNeuronCount();
		int obtainWeightCount();

		// int getInputNeuronCount();
		// int getOutputNeuronCount();
};

class AnnBase {
public:
	virtual void prepare(Topology *top, double alpha, double eta) = 0;
	virtual void init(double w_arr_1[]) = 0;
	virtual void train(double *a, double *b) = 0;
	virtual void feedForward(double *a, double *b) = 0;
	virtual void destroy() = 0;
private:
	virtual void calc_feedForward() = 0;
};

class AnnSerialDBL : public AnnBase {
private:
	Topology* cTopology;
	double mAlpha;
	double mEta;

	int neuronCount;
	int inputCount;
	int outputCount;
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


public:
	void prepare(Topology *top, double alpha, double eta);
	void init(double w_arr_1[]);
	void train(double *a, double *b);
	void feedForward(double *a, double *b);
	void destroy();
	AnnSerialDBL() {};

	double* getWeights();


private:
	void calc_feedForward();
	double delta_w(double grad, double dw);
	double f(double x);
	double f_deriv(double x);
	double gL(double a, double z, double t);
	double w_gradient(int layer_id, int w_i, int w_j);
	void calc_gjl();
};


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




/* Class definitions here. */
void run_cuda_sample();

#endif /* ANN_HEADER */
