

#define TCLAP_SETBASE_ZERO 1

#include "tclap/CmdLine.h"
#include <iostream>
#include <string>


using namespace TCLAP;
using namespace std;

int main(int argc, char** argv)
{

	try {

	CmdLine cmd("this is a message", ' ', "0.99" );

	ValueArg<int> itest("i", "intTest", "integer test", true, 5, "int");
	cmd.add( itest );

	//
	// Parse the command line.
	//
	cmd.parse(argc,argv);

	//
	// Set variables
	//
	int _intTest = itest.getValue();
	cout << "found int: " << _intTest << endl;

	} catch ( ArgException& e )
	{ cout << "ERROR: " << e.error() << " " << e.argId() << endl; }
}



