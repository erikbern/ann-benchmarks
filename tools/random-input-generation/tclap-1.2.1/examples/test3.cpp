

#include "tclap/CmdLine.h"
#include <iostream>
#include <string>

using namespace TCLAP;
using namespace std;

bool _boolTestB;
string _stringTest;
string _utest;
string _ztest;

void parseOptions(int argc, char** argv);

int main(int argc, char** argv)
{

	parseOptions(argc,argv);

	cout << "for string we got : " << _stringTest<< endl
		 << "for ulabeled one we got : " << _utest << endl
		 << "for ulabeled two we got : " << _ztest << endl
		 << "for bool B we got : " << _boolTestB << endl;

}


void parseOptions(int argc, char** argv)
{
	try {

	CmdLine cmd("this is a message", '=', "0.99" );

	//
	// Define arguments
	//

	SwitchArg btest("B","existTestB", "exist Test B", false);
	cmd.add( btest );

	ValueArg<string> stest("", "stringTest", "string test", true, "homer",
					       "string");
	cmd.add( stest );

	UnlabeledValueArg<string> utest("unTest1","unlabeled test one", true,
					                "default","string");
	cmd.add( utest );

	UnlabeledValueArg<string> ztest("unTest2","unlabeled test two", true,
					                "default","string");
	cmd.add( ztest );

	MultiArg<int> itest("i", "intTest", "multi int test", false,"int" );
	cmd.add( itest );

	MultiArg<float> ftest("f", "floatTest", "multi float test", false,"float" );
	cmd.add( ftest );

	UnlabeledMultiArg<string> mtest("fileName","file names",false,
					                "fileNameString");
	cmd.add( mtest );
	//
	// Parse the command line.
	//
	cmd.parse(argc,argv);


	//
	// Set variables
	//
	_stringTest = stest.getValue();
	_boolTestB = btest.getValue();
	_utest = utest.getValue();
	_ztest = ztest.getValue();

	vector<int> vi = itest.getValue();
	for ( int i = 0; static_cast<unsigned int>(i) < vi.size(); i++ )
		cout << "[-i] " << i << "  " <<  vi[i] << endl;

	vector<float> vf = ftest.getValue();
	for ( int i = 0; static_cast<unsigned int>(i) < vf.size(); i++ )
		cout << "[-f] " << i << "  " <<  vf[i] << endl;

	vector<string> v = mtest.getValue();
	for ( int i = 0; static_cast<unsigned int>(i) < v.size(); i++ )
		cout << "[  ] " << i << "  " <<  v[i] << endl;

	} catch ( ArgException& e )
	{ cout << "ERROR: " << e.error() << " " << e.argId() << endl; }
}



