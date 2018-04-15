#ifndef ANN_DEMO_HEADER
#define ANN_DEMO_HEADER

#include <stdio.h>
#include <iostream>
#include "ann.h"
#include "tests.h"

using namespace std;

class XOR : public Data_Double
{
public:
	void generate(int n);

	int getResult(int index);

	XOR()
	{
		inputs = 2;
		outputs = 2;
		samples = 0;
	}
	void printInputs(int index);

	void printOutputs(int index);
};

class XOR_Float : public Data_Float
{
public:
	void generate(int n);

	int getResult(int index);

	XOR_Float()
	{
		inputs = 2;
		outputs = 2;
		samples = 0;
	}
	void printInputs(int index);

	void printOutputs(int index);
};

#endif /* ANN_DEMO_HEADER */
