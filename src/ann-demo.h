#ifndef ANN_DEMO_HEADER
#define ANN_DEMO_HEADER

#include <stdio.h>
#include <iostream>
#include "ann.h"
#include "tests.h"

using namespace std;

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

#endif /* ANN_DEMO_HEADER */
