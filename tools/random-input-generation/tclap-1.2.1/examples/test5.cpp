

#include "tclap/CmdLine.h"
#include <iostream>
#include <string>

using namespace TCLAP;
using namespace std;

string _orTest;
string _orTest2;
string _testc;
bool _testd;

void parseOptions(int argc, char** argv);

int main(int argc, char** argv)
{

	parseOptions(argc,argv);

	cout << "for A OR B we got : " << _orTest<< endl
		 << "for string C we got : " << _testc << endl
		 << "for string D we got : " << _testd << endl
		 << "for E or F or G we got: " << _orTest2 << endl;

}


void parseOptions(int argc, char** argv)
{
	try {

	CmdLine cmd("this is a message", ' ', "0.99" );

	//
	// Define arguments
	//

	ValueArg<string> atest("a", "aaa", "or test a", true, "homer", "string");
	ValueArg<string> btest("b", "bbb", "or test b", true, "homer", "string");
	cmd.xorAdd( atest, btest );

	ValueArg<string> ctest("c", "ccc", "c test", true, "homer", "string");
	cmd.add( ctest );

	SwitchArg dtest("", "ddd", "d test", false);
	cmd.add( dtest );

	ValueArg<string> etest("", "eee", "e test", false, "homer", "string");
	ValueArg<string> ftest("", "fff", "f test", false, "homer", "string");
	ValueArg<string> gtest("g", "ggg", "g test", false, "homer", "string");
	vector<Arg*> xorlist;
	xorlist.push_back(&etest);
	xorlist.push_back(&ftest);
	xorlist.push_back(&gtest);
	cmd.xorAdd( xorlist );

	MultiArg<string> itest("i", "iii", "or test i", true, "string");
	MultiArg<string> jtest("j", "jjj", "or test j", true, "string");
	cmd.xorAdd( itest, jtest );

	//
	// Parse the command line.
	//
	cmd.parse(argc,argv);


	//
	// Set variables
	//

	if ( atest.isSet() )
		_orTest = atest.getValue();
	else if ( btest.isSet() )
		_orTest = btest.getValue();
	else
		// Should never get here because TCLAP will note that one of the
		// required args above has not been set.
		throw("very bad things...");

	_testc = ctest.getValue();
	_testd = dtest.getValue();

	if ( etest.isSet() )
		_orTest2 = etest.getValue();
	else if ( ftest.isSet() )
		_orTest2 = ftest.getValue();
	else if ( gtest.isSet() )
		_orTest2 = gtest.getValue();
	else
		throw("still bad");

    if ( jtest.isSet() )
    {
        cout << "for J:" << endl;
        vector<string> v = jtest.getValue();
        for ( int z = 0; static_cast<unsigned int>(z) < v.size(); z++ )
            cout << " " << z << "  " << v[z] << endl;
    }
    else if ( itest.isSet() )
    {
        cout << "for I:" << endl;
        vector<string> v = itest.getValue();
        for ( int z = 0; static_cast<unsigned int>(z) < v.size(); z++ )
            cout << " " << z << "  " << v[z] << endl;
    }
    else
		throw("yup, still bad");



	} catch ( ArgException& e )
	{ cout << "ERROR: " << e.error() << " " << e.argId() << endl; }
}



