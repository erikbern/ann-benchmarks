

#include "tclap/CmdLine.h"
#include <iostream>
#include <string>

using namespace TCLAP;
using namespace std;

int _intTest;
float _floatTest;
bool _boolTestA;
bool _boolTestB;
bool _boolTestC;
string _stringTest;
string _utest;

void parseOptions(int argc, char** argv);

int main(int argc, char** argv)
{

	parseOptions(argc,argv);

	cout << "for float we got : " << _floatTest << endl
		 << "for int we got : " << _intTest<< endl
		 << "for string we got : " << _stringTest<< endl
		 << "for ulabeled we got : " << _utest << endl
		 << "for bool A we got : " << _boolTestA << endl
		 << "for bool B we got : " << _boolTestB << endl
		 << "for bool C we got : " << _boolTestC << endl;

}


void parseOptions(int argc, char** argv)
{
	try {

	CmdLine cmd("this is a message", ' ', "0.99" );

	//
	// Define arguments
	//

	SwitchArg btest("B","existTestB", "tests for the existence of B", false);
	cmd.add( btest );

	SwitchArg ctest("C","existTestC", "tests for the existence of C", false);
	cmd.add( ctest );

	SwitchArg atest("A","existTestA", "tests for the existence of A", false);
	cmd.add( atest );

	ValueArg<string> stest("s","stringTest","string test",true,"homer",
					       "string");
	cmd.add( stest );

	ValueArg<int> itest("i", "intTest", "integer test", true, 5, "int");
	cmd.add( itest );

	ValueArg<double> ftest("f", "floatTest", "float test", false, 3.7, "float");
	cmd.add( ftest );

	UnlabeledValueArg<string> utest("unTest","unlabeld test", true,
					                "default","string");
	cmd.add( utest );

	UnlabeledMultiArg<string> mtest("fileName", "file names", false, "string");
	cmd.add( mtest );

	//
	// Parse the command line.
	//
	cmd.parse(argc,argv);

	//
	// Set variables
	//
	_intTest = itest.getValue();
	_floatTest = ftest.getValue();
	_stringTest = stest.getValue();
	_boolTestB = btest.getValue();
	_boolTestC = ctest.getValue();
	_boolTestA = atest.getValue();
	_utest = utest.getValue();

	vector<string> v = mtest.getValue();
	for ( int i = 0; static_cast<unsigned int>(i) < v.size(); i++ )
		cout << i << "  " <<  v[i] << endl;

	} catch ( ArgException& e )
	{ cout << "ERROR: " << e.error() << " " << e.argId() << endl; }
}



